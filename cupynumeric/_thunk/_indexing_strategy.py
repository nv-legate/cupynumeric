# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Strategy ADT for advanced-indexing dispatch.

Strategies are named by KEY SHAPE, not by which task they currently use.
Each strategy OWNS its execution via ``execute_get`` / ``execute_set``; the
backend (Layer C, i.e. ``DeferredArray._dispatch_gather`` /
``_dispatch_scatter``) picks the runtime tier.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from legate.core import get_legate_runtime

from ..config import CuPyNumericOpCode
from ..runtime import runtime
from ..settings import settings
from ._advanced_indexing import _execute_boolean_indexing_task

if TYPE_CHECKING:
    from .deferred import DeferredArray


legate_runtime = get_legate_runtime()


def _shape_mismatch_error(
    rhs_shape: tuple[int, ...], selection_shape: tuple[int, ...]
) -> ValueError:
    return ValueError(
        f"shape mismatch: value array of shape {rhs_shape!r} could not "
        f"broadcast to indexing result of shape {selection_shape!r}"
    )


def _check_broadcast_shapes(
    rhs_shape: tuple[int, ...], selection_shape: tuple[int, ...]
) -> None:
    try:
        np.broadcast_shapes(rhs_shape, selection_shape)
    except ValueError:
        raise _shape_mismatch_error(rhs_shape, selection_shape)


class Strategy(Protocol):
    """Abstract base for all strategies.

    Concrete subclasses are frozen dataclasses; each declares its own
    ``array`` field plus variant-specific fields, and owns its execution
    via ``execute_get`` / ``execute_set``.
    """

    @abstractmethod
    def execute_get(self) -> DeferredArray:
        """Run the GET (read) operation. Called from ``a[key]`` after
        ``_lower_to_strategy``."""
        ...

    @abstractmethod
    def execute_set(self, rhs: DeferredArray) -> None:
        """Run the SET (write) operation. Called from ``a[key] = rhs`` after
        ``_lower_to_strategy``. ``rhs`` is the write payload."""
        ...


@dataclass(frozen=True)
class Basic(Strategy):
    """slices / ints / None / Ellipsis only — view ops both directions.

    Not "advanced indexing": no integer or boolean arrays in the key.
    """

    array: DeferredArray
    key: tuple[Any, ...]

    def execute_get(self) -> DeferredArray:
        result = self.array._get_view(self.key)
        if Ellipsis not in self.key and result.shape == ():  # type: ignore[operator]
            view = result
            result = runtime.create_deferred_thunk((), self.array.base.type)
            task = legate_runtime.create_auto_task(
                self.array.library, CuPyNumericOpCode.READ
            )
            task.add_input(view.base)
            task.add_output(result.base)
            task.execute()
        return result

    def execute_set(self, rhs: DeferredArray) -> None:
        view = self.array._get_view(self.key)
        if view.size == 0:
            return
        if view.shape == ():
            assert rhs.size == 1
            task = legate_runtime.create_auto_task(
                self.array.library, CuPyNumericOpCode.WRITE
            )
            # Pass the view with write-discard privilege so the mapper
            # either creates a fresh instance for this one-element view or
            # picks an existing valid instance for the parent.
            task.add_output(view.base)
            task.add_input(rhs.base)
            task.execute()
            return
        # ``arr[key] op= value`` calls __setitem__ after __iop__ already wrote
        # through the view; skip the copy. Store/Storage have no __eq__, so
        # compare underlying Legion handles via equal_storage.
        if view.base.equal_storage(rhs.base):
            return
        view.copy(rhs, deep=False)


@dataclass(frozen=True)
class BoolMask(Strategy):
    """``a[mask]`` — single boolean array, any leading-axis shape.

    Handles both full-shape masks (``mask.shape == a.shape``) and
    leading-axis partial masks (``mask.shape == a.shape[:mask.ndim]``).

    ``transpose_index`` is always 0 in the current implementation; it is
    reserved for a future transpose fast-path that will split per mask shape.

    execute_get → ADVANCED_INDEXING task.
    execute_set → ADVANCED_INDEXING + scatter; PUTMASK fast path on scalar rhs.
    """

    array: DeferredArray
    transformed_array: DeferredArray
    bool_key: DeferredArray
    transpose_index: int

    def execute_get(self) -> DeferredArray:
        if self.bool_key.size == 0 or self.transformed_array.size == 0:
            if self.transformed_array.size == 0 and self.bool_key.size != 0:
                s = self.bool_key.nonzero()[0].size
            else:
                s = 0
            out_shape = (s,) + self.transformed_array.shape[
                self.bool_key.ndim :
            ]
            out = runtime.create_deferred_thunk(
                out_shape, self.transformed_array.base.type
            )
            return out
        return _execute_boolean_indexing_task(
            self.transformed_array, self.bool_key, is_set=False
        )

    def execute_set(self, rhs: DeferredArray) -> None:
        if self.bool_key.size == 0:
            # NumPy validates shape even for empty selections.
            selection_shape = (0,) + self.transformed_array.shape[
                self.bool_key.ndim :
            ]
            _check_broadcast_shapes(rhs.shape, selection_shape)
            return

        if self.transformed_array.size == 0:
            # Zero-sized target: scatter is a no-op, but shape validation must
            # cover the selected-count dimension too.  a.shape==(3,0) with an
            # all-true mask selects shape (3,0); (2,0) or (4,0) rhs must raise.
            s = self.bool_key.nonzero()[0].size
            trailing = self.transformed_array.shape[self.bool_key.ndim :]
            selection_shape = (s,) + trailing
            _check_broadcast_shapes(rhs.shape, selection_shape)
            return

        # Scalar rhs fast path: putmask avoids the gather-scatter round-trip.
        # transpose_index is always 0 in the current implementation.
        if rhs.size == 1 and self.transpose_index == 0:
            from .deferred import DeferredArray  # circular-import guard

            self.transformed_array.putmask(
                DeferredArray(base=self.bool_key.base), rhs
            )
            return

        index_array = _execute_boolean_indexing_task(
            self.transformed_array, self.bool_key, is_set=True
        )
        # ``_prepare_boolean_array_indexing`` only emits this strategy when
        # ``transformed_array`` is a view of ``array`` (identity or transpose).
        # Writes propagate to the original through the shared backing store.
        self.array._perform_scatter(
            rhs, index_array, destination=self.transformed_array
        )


@dataclass(frozen=True)
class IntArraySingleAxis(Strategy):
    """``a[:, idx]`` — one int array on one axis; all other elements are
    ``slice(None)``, non-trivial slices, integer scalars, or ``np.newaxis``.

    Before the take: scalar subscripts are projected out (``store.project``);
    partial slices are applied as zero-copy view transforms (``_slice_store``).
    After the take: ``store.promote`` re-inserts singleton dims at the
    positions in ``promote_dims`` (one entry per ``np.newaxis`` in the key).

    Absorbs ``Take``. GET goes through the TAKE task.
    """

    array: DeferredArray  # post-projection/slice view (scalars and partial slices already applied)
    indices: DeferredArray
    axis: int
    promote_dims: tuple[
        int, ...
    ]  # output dims to promote after the take (for np.newaxis)

    def execute_get(self) -> DeferredArray:
        result = self.array._advanced_indexing_using_take(
            self.axis, self.indices
        )
        if self.promote_dims:
            from .deferred import DeferredArray  # circular-import guard

            for dim in self.promote_dims:
                result = DeferredArray(base=result.base.promote(dim, 1))
        return result

    def execute_set(self, rhs: DeferredArray) -> None:
        # PUT-task not yet landed; add a SET arm in _lower_to_strategy when it does.
        raise NotImplementedError(
            "IntArraySingleAxis SET not yet implemented; "
            "_lower_to_strategy should route SET to General"
        )


@dataclass(frozen=True)
class General(Strategy):
    """Everything else.

    int + non-trivial slice, bool reduced via nonzero, None/newaxis mixed
    with advanced, non-adjacent advanced axes — the residual bucket once
    the fast-path variants have been ruled out.

    Carries the post-transformation view in ``array`` (transpose / slice /
    project / promote already applied by ``_lower_to_strategy``); executors
    run ZIP → gather/scatter against it. ``original`` is the indexed array
    as the caller saw it; SET writes back through ``original`` when the
    transformation made ``array`` a distinct buffer.
    """

    array: DeferredArray  # post-transformation view (transpose/slice/project/promote already applied)
    original: DeferredArray  # the array as the caller saw it; SET writes back through this
    index_arrays: tuple[DeferredArray, ...]
    start_index: int

    def execute_get(self) -> DeferredArray:
        check_bounds = settings.bounds_check_enabled("indexing")
        if (
            fused_result := self.array._zipgather(
                self.start_index, self.index_arrays, check_bounds=check_bounds
            )
        ) is not None:
            return fused_result
        # Multi-GPU fused path: feed per-dim index arrays directly into the
        # NCCL all-to-all gather, skipping the standalone ZIP task.
        if (
            settings.use_nccl_gather()
            and runtime.num_gpus > 1
            and self.array.base.ndim > 0
        ):
            nccl_fused = self.array._nccl_zipgather(
                self.start_index, self.index_arrays, check_bounds=check_bounds
            )
            if nccl_fused is not None:
                return nccl_fused
        index_array = self.array._zip_indices(
            self.start_index, self.index_arrays, check_bounds=check_bounds
        )
        return self.original._perform_gather(self.array, index_array)

    def execute_set(self, rhs: DeferredArray) -> None:
        check_bounds = settings.bounds_check_enabled("indexing")
        if self.array._zipscatter(
            rhs,
            self.start_index,
            self.index_arrays,
            self.original,
            check_bounds,
        ):
            return
        # Multi-GPU fused path: feed per-dim index arrays directly into the
        # NCCL all-to-all scatter, skipping the standalone ZIP task.
        if (
            settings.use_nccl_scatter()
            and runtime.num_gpus > 1
            and self.array.base.ndim > 0
            and self.array._nccl_zipscatter(
                rhs,
                self.start_index,
                self.index_arrays,
                self.original,
                check_bounds,
            )
        ):
            return
        index_array = self.array._zip_indices(
            self.start_index, self.index_arrays, check_bounds=check_bounds
        )
        self.original._perform_scatter(
            rhs, index_array, destination=self.array, user_target=self.original
        )
