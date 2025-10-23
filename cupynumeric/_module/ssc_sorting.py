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

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate
from .creation_shape import empty

if TYPE_CHECKING:
    from ..types import SelectKind, SortType


@add_boilerplate("a")
def argsort(
    a: ndarray,
    axis: int | None = -1,
    kind: SortType = "quicksort",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Returns the indices that would sort an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis to sort. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'quicksort', 'mergesort', 'heapsort', 'stable'}``, optional
        Default is 'quicksort'. The underlying sort algorithm might vary.
        The code basically supports 'stable' or *not* 'stable'.
    order : str or list[str], optional
        Currently not supported

    Returns
    -------
    index_array : ndarray[int]
        Array of indices that sort a along the specified axis. It has the
        same shape as `a.shape` or is flattened in case of `axis` is None.

    See Also
    --------
    numpy.argsort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    result = ndarray(a.shape, np.int64)
    result._thunk.sort(
        rhs=a._thunk, argsort=True, axis=axis, kind=kind, order=order
    )
    return result


@add_boilerplate("a")
def sort(
    a: ndarray,
    axis: int | None = -1,
    kind: SortType = "quicksort",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Returns a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis to sort. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'quicksort', 'mergesort', 'heapsort', 'stable'}``, optional
        Default is 'quicksort'. The underlying sort algorithm might vary.
        The code basically supports 'stable' or *not* 'stable'.
    order : str or list[str], optional
        Currently not supported

    Returns
    -------
    out : ndarray
        Sorted array with same dtype and shape as `a`. In case `axis` is
        None the result is flattened.


    See Also
    --------
    numpy.sort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.sort(rhs=a._thunk, axis=axis, kind=kind, order=order)
    return result


@add_boilerplate("a")
def sort_complex(a: ndarray) -> ndarray:
    """

    Returns a sorted copy of an array sorted along the last axis. Sorts the
    real part first, the imaginary part second.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    out : ndarray, complex
        Sorted array with same shape as `a`.

    See Also
    --------
    numpy.sort_complex

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    result = sort(a)
    # force complex result upon return
    if np.issubdtype(result.dtype, np.complexfloating):
        return result
    elif (
        np.issubdtype(result.dtype, np.integer) and result.dtype.itemsize <= 2
    ):
        return result.astype(np.complex64, copy=True)
    else:
        return result.astype(np.complex128, copy=True)


# partition


@add_boilerplate("a")
def argpartition(
    a: ndarray,
    kth: int | Sequence[int],
    axis: int | None = -1,
    kind: SelectKind = "introselect",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Perform an indirect partition along the given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    kth : int or Sequence[int]
    axis : int or None, optional
        Axis to partition. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'introselect'}``, optional
        Currently not supported.
    order : str or list[str], optional
        Currently not supported.

    Returns
    -------
    out : ndarray[int]
        Array of indices that partitions a along the specified axis. It has the
        same shape as `a.shape` or is flattened in case of `axis` is None.


    Notes
    -----
    The current implementation falls back to `cupynumeric.argsort`.

    See Also
    --------
    numpy.argpartition

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, np.int64)
    result._thunk.partition(
        rhs=a._thunk,
        argpartition=True,
        kth=kth,
        axis=axis,
        kind=kind,
        order=order,
    )
    return result


@add_boilerplate("a")
def partition(
    a: ndarray,
    kth: int | Sequence[int],
    axis: int | None = -1,
    kind: SelectKind = "introselect",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Returns a partitioned copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    kth : int or Sequence[int]
    axis : int or None, optional
        Axis to partition. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'introselect'}``, optional
        Currently not supported.
    order : str or list[str], optional
        Currently not supported.

    Returns
    -------
    out : ndarray
        Partitioned array with same dtype and shape as `a`. In case `axis` is
        None the result is flattened.

    Notes
    -----
    The current implementation falls back to `cupynumeric.sort`.

    See Also
    --------
    numpy.partition

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.partition(
        rhs=a._thunk, kth=kth, axis=axis, kind=kind, order=order
    )
    return result


@add_boilerplate("keys")
def lexsort(keys: Sequence[ndarray], axis: int | None = -1) -> ndarray:
    """
    Perform a stable lexicographical sort along multiple keys.

    Parameters
    ----------
    keys : Sequence[ndarray]
        Sequence of arrays to sort. The last key is the primary (most significant) sort key.
    axis : int, optional
        Axis to be indirectly sorted. By default, sort over the last axis of each
        sequence. Separate slices along axis sorted over independently.
        Note: axis=None is not supported and will raise TypeError.

    Returns
    -------
    indices : ndarray[int]
        Array of indices that lexicographically sort the keys along the specified axis.

    See Also
    --------
    numpy.lexsort

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    Notes
    -----
    For unsupported input types (e.g., string arrays, object arrays), the function
    will raise a TypeError during conversion in @add_boilerplate.
    """
    # After @add_boilerplate("keys"), keys is converted to a single stacked ndarray
    keys_array: ndarray = keys  # type: ignore[assignment]

    # Handle None input
    if keys_array is None:
        raise TypeError("'NoneType' object is not iterable")

    # NumPy's lexsort does not support axis=None
    if axis is None:
        raise TypeError(
            "'NoneType' object cannot be interpreted as an integer"
        )

    # Check if we have empty keys array (no keys or empty data)
    if keys_array.size == 0:
        if keys_array.shape[0] == 0:
            # No keys provided
            raise TypeError("Need at least one key to sort")
        else:
            # Valid keys but empty data - return empty index array
            # Determine the output shape based on the input data shape
            if keys_array[0].ndim == 0:
                output_shape: tuple[int, ...] = ()
            else:
                output_shape = keys_array.shape[
                    1:
                ]  # Remove the keys dimension
            return empty(output_shape, dtype=np.int64)

    # Check that all arrays have the same shape
    first_shape = keys_array[0].shape
    for i, key in enumerate(keys_array[1:], 1):
        if key.shape != first_shape:
            raise ValueError(
                f"All keys must have the same shape. Key {i} has shape {key.shape}, expected {first_shape}"
            )

    # If the keys are scalar, return a scalar 0 since there's only one element
    if keys_array[0].ndim == 0:
        from .creation_shape import full

        return full((), 0, dtype=np.int64)

    # Validate axis before processing
    if axis is not None:
        ndim = keys_array[0].ndim
        if axis >= ndim or axis < -ndim:
            raise np.exceptions.AxisError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )

    def _preprocess_key_for_sorting(key: ndarray) -> ndarray:
        """
        Preprocess a key to handle special floating-point values consistently.

        Maps special values to finite values that sort in the correct order
        to ensure consistent behavior across all backends.
        """
        from . import isnan, isinf, signbit
        from .._module.ssc_searching import where

        # Only process floating-point arrays that might contain special values
        if not np.issubdtype(key.dtype, np.floating):
            return key

        # Check if we have any special values
        has_nan = isnan(key).any()
        has_inf = isinf(key).any()

        if not (has_nan or has_inf):
            return key  # No special values, return as-is

        # Create a copy to avoid modifying the original
        processed_key = key.copy()

        # Get masks for special values
        nan_mask = isnan(key)
        inf_mask = isinf(key)
        pos_inf_mask = inf_mask & ~signbit(key)  # positive infinity
        neg_inf_mask = inf_mask & signbit(key)  # negative infinity
        finite_mask = ~(nan_mask | inf_mask)

        # Get the range of finite values to create appropriate replacements
        finite_min: float
        finite_max: float
        finite_range: float

        if finite_mask.any():
            finite_values = processed_key[finite_mask]
            finite_min = float(finite_values.min())
            finite_max = float(finite_values.max())
            finite_range = finite_max - finite_min
            if finite_range == 0:
                finite_range = 1.0
        else:
            # No finite values, use default range
            finite_min = 0.0
            finite_max = 0.0
            finite_range = 1.0

        # Replace special values to ensure correct ordering: -inf < finite < +inf < NaN
        if neg_inf_mask.any():
            neg_inf_replacement = finite_min - finite_range - 1000.0
            processed_key = where(
                neg_inf_mask, neg_inf_replacement, processed_key
            )

        if pos_inf_mask.any():
            pos_inf_replacement = finite_max + finite_range + 1000.0
            processed_key = where(
                pos_inf_mask, pos_inf_replacement, processed_key
            )

        if nan_mask.any():
            nan_replacement = finite_max + finite_range + 2000.0
            processed_key = where(nan_mask, nan_replacement, processed_key)

        return processed_key

    # Always preprocess keys to handle special floating-point values consistently
    processed_keys = [_preprocess_key_for_sorting(key) for key in keys_array]

    # Start with the first key (least significant in NumPy's convention)
    indices = argsort(processed_keys[0], axis=axis, kind="stable")

    # Process remaining keys in order (most significant last)
    for key in processed_keys[1:]:
        if key.ndim > 1:
            from .indexing import take_along_axis

            # Reorder the key according to current indices
            key_reordered = take_along_axis(key, indices, axis=axis)
            # Sort the reordered key (stable sort preserves order for equal elements)
            order = argsort(key_reordered, axis=axis, kind="stable")
            # Update indices by taking along the new order
            indices = take_along_axis(indices, order, axis=axis)
        else:
            # For 1D case, directly index and sort
            order = argsort(key[indices], axis=axis, kind="stable")
            indices = indices[order]

    return indices
