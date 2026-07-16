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
from __future__ import annotations

from typing import TYPE_CHECKING

import legate.core.types as ty
import numpy as np
from legate.core import LogicalStore, align, broadcast, get_legate_runtime

from ..config import CuPyNumericOpCode
from ..runtime import runtime

if TYPE_CHECKING:
    from .deferred import DeferredArray


legate_runtime = get_legate_runtime()


def _project_scalar(
    store: LogicalStore, dim: int, k: int | np.integer
) -> LogicalStore:
    """Project ``store`` along ``dim`` at scalar index ``k``.

    Removes dimension ``dim`` by selecting its ``k``-th element (NumPy-style
    integer indexing, e.g. ``a[k]``). Negative ``k`` is wrapped relative to
    the extent of ``dim``; bounds checking is the caller's responsibility.
    """
    k = int(k)
    if k < 0:
        k += store.shape[dim]
    return store.project(dim, k)


def _slice_store(k: slice, store: LogicalStore, dim: int) -> LogicalStore:
    # slice.indices() clamps to [0, size] and resolves None/negatives.
    # Legate raises for out-of-bounds stop, so we must clamp first.
    start, stop, step = k.indices(store.shape[dim])
    if step == 1 and start >= stop:
        # Legate's store.slice() does not correctly produce a zero-extent
        # dimension for empty ranges. Use project+promote instead: project
        # removes the dimension (using index 0, which is always valid when
        # the original size > 0), then promote adds it back with extent 0.
        # If the dimension is already empty (size == 0), the store is already
        # zero-extent and no transformation is needed.
        if store.shape[dim] > 0:
            store = store.project(dim, 0)
            store = store.promote(dim, 0)
        return store
    return store.slice(dim, slice(start, stop, k.step))


def _execute_boolean_indexing_task(
    rhs: DeferredArray, key: DeferredArray, is_set: bool
) -> DeferredArray:
    """
    Execute the ADVANCED_INDEXING task for boolean indexing.

    This is shared logic for both get and set operations.

    Parameters
    ----------
    rhs : DeferredArray
        The input array (after transformations and transpose)
    key : DeferredArray
        The boolean mask (will be promoted to match rhs dimensionality)
    is_set : bool
        Whether this is a set operation

    Returns
    -------
    DeferredArray
        The raw output from the ADVANCED_INDEXING task
    """
    rhs_store = rhs.base
    key_store = key.base

    key_dims = key_store.ndim
    for i in range(key_dims, rhs_store.ndim):
        key_store = key_store.promote(i, rhs_store.shape[i])

    out_dtype = (
        ty.point_type(rhs_store.ndim) if is_set else rhs_store.type
    )  # set: Point<N> for indirect copy

    # Boolean-indexed dims are flattened into one output dim.
    out = runtime.create_unbound_thunk(
        out_dtype, ndim=rhs_store.ndim - key.ndim + 1
    )

    task = legate_runtime.create_auto_task(
        rhs.library, CuPyNumericOpCode.ADVANCED_INDEXING
    )
    task.add_output(out.base)
    p_rhs = task.add_input(rhs_store)
    p_key = task.add_input(key_store)
    task.add_scalar_arg(is_set, ty.bool_)

    task.add_constraint(align(p_rhs, p_key))
    if rhs_store.ndim > 1:
        # key_store is promoted to rhs_store.ndim above, so partitioning is
        # always along dim 0 only; dims 1..N-1 are broadcast regardless of
        # the original key.ndim.
        task.add_constraint(broadcast(p_rhs, range(1, rhs_store.ndim)))

    task.execute()

    return out
