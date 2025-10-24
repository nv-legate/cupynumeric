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

from typing import Literal, Any

from .._array.util import add_boilerplate
from .._array.array import ndarray
import numpy as np

_builtin_any = any


@add_boilerplate("ar")
def unique(
    ar: ndarray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
) -> ndarray:
    """

    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:
    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique
        array.
        Currently not supported.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
        Currently not supported.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
        Currently not supported.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.
        Currently not supported.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    numpy.unique

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    Notes
    --------
    Keyword arguments for optional outputs are not yet supported.
    `axis` is also not handled currently.

    """
    if (
        _builtin_any((return_index, return_inverse, return_counts))
        or axis is not None
    ):
        raise NotImplementedError(
            "Keyword arguments for `unique` are not yet supported"
        )
    return ar.unique()


@add_boilerplate("ar2")
def in1d(
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    invert: bool = False,
    kind: Literal["sort", "table"] | None = None,
) -> "ndarray":
    """
    Test whether each element of this 1-D array is also present in a second
    array.

    Parameters
    ----------
    ar2 : array_like
        The values against which to test each value of ar1.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which can
        speed up the calculation. Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of ar1 is in ar2 and True otherwise).
        Default is False.
    kind : {None, 'sort', 'table'}, optional
        The algorithm to use. This will not affect the final result, but
        will affect the speed and memory use. The default, None, will
        select automatically based on memory considerations.

    Returns
    -------
    in1d : ndarray, bool
        The values ar1[in1d] are in ar2.

    See Also
    --------
    numpy.in1d : NumPy equivalent function

    Availability
    ------------
    Multiple GPUs, Multiple CPUs

    Notes
    --------
    When `kind` is None (default), the 'sort' algorithm is used to leverage
    GPU acceleration for optimal performance.
    """
    # Ciruclar import
    from .._module.creation_shape import (
        full,
        ones,
        ones_like,
        zeros,
        zeros_like,
    )

    # Check kind
    if kind not in (None, "sort", "table"):
        raise ValueError("kind must be None, 'sort', or 'table'")

    ar1 = ar1.ravel()
    ar2 = ar2.ravel()

    # Handle empty arrays
    if ar1.size == 0 or ar2.size == 0:
        if invert:
            return ones(ar1.shape, dtype=bool)
        else:
            return zeros(ar1.shape, dtype=bool)

    # Handle object arrays
    if ar2.dtype == object:
        ar2 = ar2.reshape(-1, 1)

    # Handle NaNs
    supports_nan = np.issubdtype(ar1.dtype, np.floating) or np.issubdtype(
        ar1.dtype, np.complexfloating
    )
    if supports_nan:
        ar2 = ar2[~np.isnan(ar2)]

    # Check dtype compatibility
    common_dtype = np.result_type(ar1.dtype, ar2.dtype)
    if ar1.dtype != common_dtype:
        ar1 = ar1._maybe_convert(common_dtype, (ar1,))
    if ar2.dtype != common_dtype:
        ar2 = ar2._maybe_convert(common_dtype, (ar2,))

    # Handle kind
    is_int_arrays = ar2.dtype.kind in ("u", "i", "b")
    use_table_method = is_int_arrays and kind == "table"

    ar2_min = 0
    ar2_max = 0

    if kind is None:
        kind = "sort"

    if use_table_method:
        if ar2.size == 0:
            if invert:
                return ones_like(ar1, dtype=bool)
            else:
                return zeros_like(ar1, dtype=bool)

        ar2_min = ar2.min()
        ar2_max = ar2.max()

        ar2_range = ar2_max - ar2_min

        # Constraints on whether we can actually use the table method:
        #  1. Assert memory usage is not too large
        below_memory_constraint = ar2_range <= 6 * (ar1.size + ar2.size)
        #  2. Check overflows for (ar2 - ar2_min); dtype=ar2.dtype
        range_safe_from_overflow = ar2_range <= np.iinfo(ar2.dtype).max
        if range_safe_from_overflow and (
            below_memory_constraint or kind == "table"
        ):
            kind = "table"

        elif kind == "table":
            raise RuntimeError(
                "You have specified kind='table', "
                "but the range of values in `ar2` or `ar1` exceed the "
                "maximum integer of the datatype. "
                "Please set `kind` to None or 'sort'."
            )
        else:
            kind = "sort"

    elif kind == "table":
        raise ValueError(
            "The 'table' method is only "
            "supported for boolean or integer arrays. "
            "Please select 'sort' or None for kind."
        )
    else:
        kind = "sort"

    result_thunk = ar1._thunk.in1d(
        ar2._thunk,
        assume_unique=assume_unique,
        invert=invert,
        kind=kind,
        ar2_min=ar2_min,
        ar2_max=ar2_max,
    )

    result = ndarray._from_thunk(result_thunk)

    if supports_nan:
        mask = np.isnan(ar1)
        result[mask] = full(mask.sum(), False, dtype=bool)

    return result


@add_boilerplate("element", "test_elements")
def isin(
    element: Any,
    test_elements: Any,
    assume_unique: bool = False,
    invert: bool = False,
    kind: Literal["sort", "table"] | None = None,
) -> ndarray:
    """
    Calculates ``element in test_elements``, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.

    Parameters
    ----------
    element : array_like
        Input array.
    test_elements : array_like
        The values against which to test each value of `element`.
        This argument is flattened if it is an array or array_like.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted, as if
        calculating `element not in test_elements`. Default is False.
        ``np.isin(a, b, invert=True)`` is equivalent to (but faster
        than) ``np.invert(np.isin(a, b))``.
    kind : {None, 'sort', 'table'}, optional
        The algorithm to use. This will not affect the final result,
        but will affect the speed and memory use. The default, None,
        will select automatically based on memory considerations.

        * If 'sort', will use a mergesort-based approach. This will have
          a memory usage of roughly 6 times the sum of the sizes of
          `element` and `test_elements`, not accounting for size of dtypes.
        * If 'table', will use a lookup table approach similar to a
          counting sort. This is only available for boolean and integer
          arrays. This will have a memory usage of the size of
          `element` plus the max-min value of `test_elements`. This tends
          to be the faster method if the following relationship is true:
          ``len(element) >> (max(test_elements)-min(test_elements))``

    Returns
    -------
    isin : ndarray, bool
        Has the same shape as `element`. The values `element[isin]`
        are in `test_elements`.

    See Also
    --------
    in1d                  : Flattened version of this function.
    numpy.isin            : NumPy equivalent function.

    Availability
    ------------
    Multiple GPUs, Multiple CPUs

    """
    return in1d(
        element,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
        kind=kind,
    ).reshape(element.shape)
