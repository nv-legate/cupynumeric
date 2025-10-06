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

from typing import TYPE_CHECKING, overload

from .._array.array import ndarray
from .._array.thunk import perform_where
from .._array.util import add_boilerplate
from .array_shape import ravel, reshape

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import SortSide


@add_boilerplate("a")
def searchsorted(
    a: ndarray,
    v: int | float | ndarray,
    side: SortSide = "left",
    sorter: ndarray | None = None,
) -> int | ndarray:
    """

    Find the indices into a sorted array a such that, if the corresponding
    elements in v were inserted before the indices, the order of a would be
    preserved.

    Parameters
    ----------
    a : 1-D array_like
        Input array. If sorter is None, then it must be sorted in ascending
        order, otherwise sorter must be an array of indices that sort it.
    v : scalar or array_like
        Values to insert into a.
    side : ``{'left', 'right'}``, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index. If there is no suitable index,
        return either 0 or N (where N is the length of a).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    indices : int or array_like[int]
        Array of insertion points with the same shape as v, or an integer
        if v is a scalar.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.searchsorted(v, side, sorter)


@add_boilerplate("a")
def argmax(
    a: ndarray,
    axis: int | None = None,
    out: ndarray | None = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """

    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmax

    Notes
    -----
    cuPyNumeric's parallel implementation may yield different results from
    NumPy when the array contains NaN(s).

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.argmax(axis=axis, out=out, keepdims=keepdims)


@add_boilerplate("a")
def argmin(
    a: ndarray,
    axis: int | None = None,
    out: ndarray | None = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """

    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmin

    Notes
    -----
    cuPyNumeric's parallel implementation may yield different results from
    NumPy when the array contains NaN(s).

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.argmin(axis=axis, out=out, keepdims=keepdims)


@add_boilerplate("a")
def flatnonzero(a: ndarray) -> ndarray:
    """

    Return indices that are non-zero in the flattened version of a.

    This is equivalent to `np.nonzero(np.ravel(a))[0]`.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of
        `a.ravel()` that are non-zero.

    See Also
    --------
    numpy.flatnonzero

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return nonzero(ravel(a))[0]


@overload
def where(
    a: npt.ArrayLike | ndarray, x: None, y: None
) -> tuple[ndarray, ...]: ...


@overload
def where(
    a: npt.ArrayLike | ndarray,
    x: npt.ArrayLike | ndarray,
    y: npt.ArrayLike | ndarray,
) -> ndarray: ...


@add_boilerplate("a", "x", "y")  # type: ignore [misc]
def where(
    a: ndarray, x: ndarray | None = None, y: ndarray | None = None
) -> ndarray | tuple[ndarray, ...]:
    """
    where(condition, [x, y])

    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : ndarray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    See Also
    --------
    numpy.where

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if x is None or y is None:
        if x is not None or y is not None:
            raise ValueError(
                "both 'x' and 'y' parameters must be specified together for"
                " 'where'"
            )
        return nonzero(a)
    return perform_where(a, x, y)


@add_boilerplate("a")
def argwhere(a: ndarray) -> ndarray:
    """
    argwhere(a)

    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : ndarray
        Indices of elements that are non-zero. Indices are grouped by element.
        This array will have shape (N, a.ndim) where N is the number of
        non-zero items.

    See Also
    --------
    numpy.argwhere

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return ndarray._from_thunk(a._thunk.argwhere())


@add_boilerplate("condition", "arr")
def extract(condition: ndarray, arr: ndarray) -> ndarray:
    """

    Return the elements of an array that satisfy some condition.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements
        of `arr` to extract.
    arr : array_like
        Input array of the same size as `condition`.

    Returns
    -------
    result : ndarray
        Rank 1 array of values from arr where `condition` is True.

    See Also
    --------
    numpy.extract

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if condition.size != arr.size:
        raise ValueError("arr array and condition array must be of same size")

    if condition.shape != arr.shape:
        condition_reshape = reshape(condition, arr.shape)
    else:
        condition_reshape = condition

    if condition_reshape.dtype == bool:
        thunk = arr._thunk.get_item(condition_reshape._thunk)
    else:
        bool_condition = condition_reshape.astype(bool)
        thunk = arr._thunk.get_item(bool_condition._thunk)

    return ndarray._from_thunk(thunk)


@add_boilerplate("a")
def nonzero(a: ndarray) -> tuple[ndarray, ...]:
    """

    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    numpy.nonzero

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.nonzero()
