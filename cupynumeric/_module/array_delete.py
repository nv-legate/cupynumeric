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

from typing import TYPE_CHECKING, Any

import numpy as np

from .._utils import is_np2

if is_np2:
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index  # type: ignore[no-redef]


from .._array.util import add_boilerplate
from .array_dimension import atleast_1d
from .creation_shape import full
from .indexing import compress
from .ssc_searching import where

if TYPE_CHECKING:
    from .._array.array import ndarray


def _normalize_and_check_indices(
    obj: ndarray, axis_size: int, axis: int | None
) -> ndarray:
    # Wrap indices to fit numpy's behavior with negative indices
    wrapped = where(obj < 0, obj + axis_size, obj)

    # Check if any indices are out of bounds after wrapping
    if ((wrapped < 0) | (wrapped >= axis_size)).any():
        # Find the first index that is out of bounds
        idx = obj[(wrapped < 0) | (wrapped >= axis_size)][0]

        axis_name = "0" if axis is None else str(axis)
        raise IndexError(
            f"index {idx} is out of bounds for axis {axis_name} with size {axis_size}"
        )

    return wrapped


@add_boilerplate("arr")
def delete(arr: ndarray, obj: Any, axis: int | None = None) -> ndarray:
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by `arr[obj]`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int, array-like of ints or bools
        Indicate indices of sub-arrays to remove along the specified axis.
    axis : int, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    See Also
    --------
    numpy.delete

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """

    if axis is None:
        arr = arr.ravel()
    else:
        axis = normalize_axis_index(axis, arr.ndim)

    if not isinstance(obj, slice):
        obj = atleast_1d(obj)
        if obj.size == 0:
            return arr.copy()
        if not np.issubdtype(obj.dtype, bool):
            size = arr.size if axis is None else arr.shape[axis]
            obj = _normalize_and_check_indices(obj, size, axis)

    if axis is None:
        if not isinstance(obj, slice) and np.issubdtype(obj.dtype, bool):
            if obj.size != arr.size:
                raise ValueError(
                    "boolean array argument obj to delete"
                    " must be one dimensional and match "
                    f"the axis length of {arr.size}"
                )
            return arr[~obj]

        mask = full(arr.size, True, dtype=bool)
        mask[obj] = False  # type: ignore [assignment]
        return arr[mask]
    else:
        if axis < -arr.ndim or axis >= arr.ndim:
            raise IndexError(
                f"axis {axis} is out of bounds for array of dimension {arr.ndim}"
            )

        if not isinstance(obj, slice) and np.issubdtype(obj.dtype, bool):
            if obj.size != arr.shape[axis]:
                raise ValueError(
                    "boolean array argument obj to delete"
                    " must be one dimensional and match "
                    f"the axis length of {arr.size}"
                )
            return compress(~obj, arr, axis=axis)

        mask = full(arr.shape[axis], True, dtype=bool)
        mask[obj] = False  # type: ignore [assignment]
        return compress(mask, arr, axis=axis)
