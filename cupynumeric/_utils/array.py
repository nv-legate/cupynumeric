# Copyright 2024 NVIDIA Corporation
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

from functools import reduce
from typing import TYPE_CHECKING, Any, cast

import legate.core.types as ty
import numpy as np
from legate.core import StoreTarget

from ..types import NdShape

if TYPE_CHECKING:
    from legate.core import PhysicalStore

SUPPORTED_DTYPES = {
    np.dtype(bool): ty.bool_,
    np.dtype(np.int8): ty.int8,
    np.dtype(np.int16): ty.int16,
    np.dtype(np.int32): ty.int32,
    np.dtype(np.int64): ty.int64,
    np.dtype(np.uint8): ty.uint8,
    np.dtype(np.uint16): ty.uint16,
    np.dtype(np.uint32): ty.uint32,
    np.dtype(np.uint64): ty.uint64,
    np.dtype(np.float16): ty.float16,
    np.dtype(np.float32): ty.float32,
    np.dtype(np.float64): ty.float64,
    np.dtype(np.complex64): ty.complex64,
    np.dtype(np.complex128): ty.complex128,
}


def is_supported_dtype(dtype: str | np.dtype[Any]) -> bool:
    """
    Whether a NumPy dtype is supported by cuPyNumeric

    Parameters
    ----------
    dtype : data-type
        The dtype to query

    Returns
    -------
    res : bool
        True if `dtype` is a supported dtype
    """
    return np.dtype(dtype) in SUPPORTED_DTYPES


def to_core_type(dtype: str | np.dtype[Any]) -> ty.Type:
    core_dtype = SUPPORTED_DTYPES.get(np.dtype(dtype))
    if core_dtype is None:
        raise TypeError(f"cuPyNumeric does not support dtype={dtype}")
    return core_dtype


def is_advanced_indexing(key: Any) -> bool:
    if key is Ellipsis or key is None:  # np.newdim case
        return False
    if _is_scalar_boolean_index(key):
        return True
    if np.isscalar(key):
        return False
    if isinstance(key, slice):
        return False
    if isinstance(key, tuple):
        return any(is_advanced_indexing(k) for k in key)
    # Any other kind of thing leads to advanced indexing
    return True


def _is_scalar_boolean_index(key: Any) -> bool:
    """Whether ``key`` is a scalar boolean indexing component."""
    if isinstance(key, (bool, np.bool_)):
        return True
    dtype = getattr(key, "dtype", None)
    return (
        dtype is not None
        and np.dtype(dtype) == np.dtype(bool)
        and getattr(key, "ndim", None) == 0
    )


def _contains_scalar_boolean_index(key: Any) -> bool:
    """Whether an indexing key contains a scalar boolean component."""
    if isinstance(key, tuple):
        return any(_is_scalar_boolean_index(k) for k in key)
    return _is_scalar_boolean_index(key)


def _expand_ellipsis(key_tuple: tuple[Any, ...], ndim: int) -> tuple[Any, ...]:
    """Expand a single ``Ellipsis`` into explicit ``slice(None)`` co-keys.

    Shared by the indexing thunks and the Doctor predicate; it lives in this
    dependency-free util so both layers can use one implementation.  Raises
    ``ValueError`` for more than one ``Ellipsis``.

    Each entry consumes one source axis except a boolean mask (one axis per
    dimension) and ``np.newaxis`` (none), matching NumPy: ``a[..., mask2d]``
    fills zero slices and stays the solo-mask case ``a[mask2d]``.
    """
    num_ellipsis = sum(k is Ellipsis for k in key_tuple)
    if num_ellipsis == 0:
        return key_tuple
    if num_ellipsis > 1:
        raise ValueError("Only a single ellipsis must be present")
    consumed = 0
    for k in key_tuple:
        if k is Ellipsis or k is np.newaxis:
            continue
        # Scalar booleans add an advanced-index dimension without consuming
        # a source axis.
        if _is_scalar_boolean_index(k):
            continue
        dtype = getattr(k, "dtype", None)
        if dtype is not None and dtype == np.dtype(bool):
            consumed += getattr(k, "ndim", 1)
        else:
            consumed += 1
    free_dims = ndim - consumed
    fill = (slice(None),) * max(free_dims, 0)
    idx = next(i for i, k in enumerate(key_tuple) if k is Ellipsis)
    return key_tuple[:idx] + fill + key_tuple[idx + 1 :]


def is_true_unoptimized_advanced_indexing(
    key: Any, ndim: int, is_set: bool = False
) -> bool:
    """
    Return True for advanced indexing patterns that use the unoptimized
    gather/scatter path.

    As gather/scatter paths are optimized, the corresponding checks here
    should be removed so the function returns False for those cases.

    Returns False for:

    * Basic indexing (scalar, slice, ellipsis, None)
    * Solo boolean array — uses the ADVANCED_INDEXING task directly,
      no gather or scatter (optimized path)
    * Single integer-array, all other dims ``slice(None)``, ndim < 5
      — routed through einsum (optimized path)
    * Single boolean array with all co-keys ``slice(None)`` (or ``Ellipsis``,
      normalized to ``slice(None)``, so ``a[..., mask]`` == ``a[:, mask]``),
      where the mask is on the leading axis or is 1-D —
      ``_prepare_boolean_array_indexing`` routes through BoolMask (transpose +
      ADVANCED_INDEXING task), no gather or scatter

    Returns True for:

    * Boolean array alongside non-``slice(None)`` co-keys —
      ``_prepare_boolean_array_indexing`` falls through to nonzero →
      ZIP + gather/scatter
    * Multidimensional boolean array on a non-leading axis (e.g.
      ``a[:, mask2d]``) — ``_prepare_boolean_array_indexing`` returns None,
      falling through to nonzero → ZIP + gather/scatter
    * ``is_set`` into a boolean mask on a non-leading axis (e.g.
      ``a[:, mask] = v``) — the multi-process scalar-RHS path still runs
      nonzero → ZIP → scatter (conservative; see body note)
    * Multiple advanced index components — ZIP + gather/scatter
    * Advanced index mixed with a non-trivial slice — gather/scatter
    * Single integer array with ndim >= 5 — einsum not applied, gather used

    Intended for use by the Doctor tool to detect genuinely expensive
    indexing operations.

    Args:
        key:  The index key passed to ``__getitem__`` or ``__setitem__``.
        ndim: Number of dimensions of the array being indexed.
        is_set: True for ``__setitem__``. A non-leading boolean-mask SET
            (``a[:, mask] = v``) is conservatively classified as unoptimized
            (see the note in the body).

    Returns:
        True if the key will trigger an unoptimized gather or scatter operation.
    """
    if not is_advanced_indexing(key):
        return False

    key_tuple = key if isinstance(key, tuple) else (key,)

    # Mirror ``_prepare_boolean_array_indexing``: expand a single Ellipsis into
    # explicit ``slice(None)`` co-keys so ``a[..., mask]`` is classified the
    # same as the explicit ``a[:, mask]`` spelling (both take the optimized
    # BoolMask route).
    if any(k is Ellipsis for k in key_tuple):
        try:
            key_tuple = _expand_ellipsis(key_tuple, ndim)
        except ValueError:
            # More than one Ellipsis is a user error; the real indexing call
            # will raise, so don't emit a (misleading) optimization warning.
            return False

    adv_components = [k for k in key_tuple if is_advanced_indexing(k)]
    non_trivial_slices = [
        k for k in key_tuple if isinstance(k, slice) and k != slice(None)
    ]

    # Multiple advanced components or a non-trivial slice alongside an
    # advanced component cannot be routed to einsum — always gather/scatter.
    if len(adv_components) > 1 or non_trivial_slices:
        return True

    # Exactly one advanced component from here.
    single = adv_components[0]
    if _is_scalar_boolean_index(single):
        # Scalar booleans lower through General (gather/scatter), not one of
        # the optimized array-indexing paths below.
        return True
    # Access .dtype directly — both numpy and cupynumeric arrays expose it.
    # Plain Python sequences (lists) have no .dtype and are always integer
    # indices, so treat them as non-boolean without any conversion.
    dtype = getattr(single, "dtype", None)
    if dtype is not None and dtype.kind == "b":
        # Solo boolean array (no co-keys) → ADVANCED_INDEXING task only: optimized.
        if len(key_tuple) == 1:
            return False
        # With co-keys, only all-slice(None) stays on the optimized BoolMask
        # route (Ellipsis was expanded above); any other co-key falls through
        # to nonzero + ZIP + gather/scatter.
        bool_pos = next(i for i, k in enumerate(key_tuple) if k is single)
        co_keys = [k for i, k in enumerate(key_tuple) if i != bool_pos]
        if not all(isinstance(k, slice) and k == slice(None) for k in co_keys):
            return True
        # A non-leading mask stays optimized only when 1-D (a multidim mask
        # there routes to None → gather/scatter). SET is conservative: a
        # multi-process scalar-RHS ``a[:, mask] = v`` still scatters, so we
        # over-warn until the dedicated SET task lands.
        single_ndim = getattr(single, "ndim", None)
        return bool_pos > 0 and (single_ndim != 1 or is_set)

    # Single integer-array index, all other dims slice(None).
    # cuPyNumeric uses einsum for ndim < 5: not expensive.
    return ndim >= 5


def calculate_volume(shape: NdShape) -> int:
    if len(shape) == 0:
        return 0
    return reduce(lambda x, y: x * y, shape)


def max_identity(
    ty: np.dtype[Any],
) -> int | np.floating[Any] | bool | np.complexfloating[Any, Any]:
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).min
    elif ty.kind == "f":
        return cast(np.floating[Any], np.finfo(ty).min)
    elif ty.kind == "c":
        return np.finfo(np.float64).min + np.finfo(np.float64).min * 1j
    elif ty.kind == "b":
        return False
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


def min_identity(
    ty: np.dtype[Any],
) -> int | np.floating[Any] | bool | np.complexfloating[Any, Any]:
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).max
    elif ty.kind == "f":
        return cast(np.floating[Any], np.finfo(ty).max)
    elif ty.kind == "c":
        return np.finfo(np.float64).max + np.finfo(np.float64).max * 1j
    elif ty.kind == "b":
        return True
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


def local_task_array(store: PhysicalStore) -> Any:
    """
    Generate an appropriate local-memory ndarray object, that is backed by the
    portion of a Legate store that was passed to a task.

    Parameters
    ----------
    store : PhysicalStore
        A Legate physical store to adapt.

    Returns
    -------
    arr : cupy.ndarray or np.ndarray
        If the store is located on GPU, then this function will return a CuPy
        array. Otherwise, a NumPy array is returned.

    """
    if store.target in {StoreTarget.FBMEM, StoreTarget.ZCMEM}:
        # cupy is only a dependency for GPU packages -- but we should
        # only hit this import in case the store is located on a GPU
        import cupy  # type: ignore

        return cupy.asarray(store)
    else:
        return np.asarray(store)
