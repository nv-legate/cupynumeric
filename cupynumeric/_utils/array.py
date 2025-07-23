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
from typing import TYPE_CHECKING, Any

import legate.core.types as ty
import numpy as np
from legate.core import PhysicalArray, StoreTarget

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
    if np.isscalar(key):
        return False
    if isinstance(key, slice):
        return False
    if isinstance(key, tuple):
        return any(is_advanced_indexing(k) for k in key)
    # Any other kind of thing leads to advanced indexing
    return True


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
        return np.finfo(ty).min
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
        return np.finfo(ty).max
    elif ty.kind == "c":
        return np.finfo(np.float64).max + np.finfo(np.float64).max * 1j
    elif ty.kind == "b":
        return True
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


def local_task_array(obj: PhysicalArray | PhysicalStore) -> Any:
    """
    Generate an appropriate local-memory ndarray object, that is backed by the
    portion of a Legate array or store that was passed to a task.

    Parameters
    ----------
    obj : PhysicalArray | PhysicalStore
        A Legate physical array or store to adapt.

    Returns
    -------
    arr : cupy.ndarray or np.ndarray
        If the array or store is located on GPU, then this function will return
        a CuPy array. Otherwise, a NumPy array is returned.

    """
    store = obj.data() if isinstance(obj, PhysicalArray) else obj

    if store.target in {StoreTarget.FBMEM, StoreTarget.ZCMEM}:
        # cupy is only a dependency for GPU packages -- but we should
        # only hit this import in case the store is located on a GPU
        import cupy  # type: ignore

        return cupy.asarray(store)
    else:
        return np.asarray(store)
