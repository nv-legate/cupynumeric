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

from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from ..lib.array_utils import normalize_axis_tuple
from .creation_data import array

if TYPE_CHECKING:
    from ..types import NdShape, NdShapeLike


def _reshape_recur(ndim: int, arr: ndarray) -> tuple[int, ...]:
    if arr.ndim < ndim:
        cur_shape: tuple[int, ...] = _reshape_recur(ndim - 1, arr)
        if ndim == 2:
            cur_shape = (1,) + cur_shape
        else:
            cur_shape = cur_shape + (1,)
    else:
        cur_shape = arr.shape
    return cur_shape


def _atleast_nd(ndim: int, arys: Sequence[ndarray]) -> list[ndarray] | ndarray:
    inputs = list(convert_to_cupynumeric_ndarray(arr) for arr in arys)
    # 'reshape' change the shape of arrays
    # only when arr.shape != _reshape_recur(ndim,arr)
    result = list(arr.reshape(_reshape_recur(ndim, arr)) for arr in inputs)
    # if the number of arrays in `arys` is 1,
    # the return value is a single array
    if len(result) == 1:
        return result[0]
    return result


def atleast_1d(*arys: ndarray) -> list[ndarray] | ndarray:
    """

    Convert inputs to arrays with at least one dimension.
    Scalar inputs are converted to 1-dimensional arrays,
    whilst higher-dimensional inputs are preserved.

    Parameters
    ----------
    *arys : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or list of arrays, each with a.ndim >= 1.
        Copies are made only if necessary.

    See Also
    --------
    numpy.atleast_1d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return _atleast_nd(1, arys)


def atleast_2d(*arys: ndarray) -> list[ndarray] | ndarray:
    """

    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    *arys : array_like
        One or more array-like sequences.
        Non-array inputs are converted to arrays.
        Arrays that already have two or more dimensions are preserved.

    Returns
    -------
    res, res2, … : ndarray
        An array, or list of arrays, each with a.ndim >= 2.
        Copies are avoided where possible, and
        views with two or more dimensions are returned.

    See Also
    --------
    numpy.atleast_2d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return _atleast_nd(2, arys)


def atleast_3d(*arys: ndarray) -> list[ndarray] | ndarray:
    """

    View inputs as arrays with at least three dimensions.

    Parameters
    ----------
    *arys : array_like
        One or more array-like sequences.
        Non-array inputs are converted to arrays.
        Arrays that already have three or more dimensions are preserved.

    Returns
    -------
    res, res2, … : ndarray
        An array, or list of arrays, each with a.ndim >= 3.
        Copies are avoided where possible, and
        views with three or more dimensions are returned.
        For example, a 1-D array of shape (N,) becomes
        a view of shape (1, N, 1),  and a 2-D array of shape (M, N)
        becomes a view of shape (M, N, 1).

    See Also
    --------
    numpy.atleast_3d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return _atleast_nd(3, arys)


@add_boilerplate("a")
def squeeze(
    a: ndarray | None, axis: NdShapeLike | None = None
) -> ndarray | None:
    """

    Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple[int], optional
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.

    Raises
    ------
    ValueError
        If `axis` is not None, and an axis being squeezed is not of length 1

    See Also
    --------
    numpy.squeeze

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a is None:
        return None
    return a.squeeze(axis=axis)


def broadcast_shapes(*args: NdShapeLike | Sequence[NdShapeLike]) -> NdShape:
    """

    Broadcast the input shapes into a single shape.

    Parameters
    ----------
    `*args` : tuples of ints, or ints
        The shapes to be broadcast against each other.

    Returns
    -------
    tuple : Broadcasted shape.

    See Also
    --------
    numpy.broadcast_shapes

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # TODO: expected "Union[SupportsIndex, Sequence[SupportsIndex]]"
    return np.broadcast_shapes(*args)  # type: ignore [arg-type]


def _broadcast_to(
    arr: ndarray,
    shape: NdShapeLike,
    subok: bool = False,
    broadcasted: bool = False,
) -> ndarray:
    # create an array object w/ options passed from 'broadcast' routines
    arr = array(arr, copy=False, subok=subok)
    # 'broadcast_to' returns a read-only view of the original array
    out_shape = broadcast_shapes(arr.shape, shape)
    if out_shape != shape:
        raise ValueError(
            f"cannot broadcast an array of shape {arr.shape} to {shape}"
        )
    result = ndarray._from_thunk(
        arr._thunk.broadcast_to(out_shape), writeable=False
    )
    return result


@add_boilerplate("arr")
def broadcast_to(
    arr: ndarray, shape: NdShapeLike, subok: bool = False
) -> ndarray:
    """

    Broadcast an array to a new shape.

    Parameters
    ----------
    arr : array_like
        The array to broadcast.
    shape : tuple or int
        The shape of the desired array.
        A single integer i is interpreted as (i,).
    subok : bool, optional
        This option is ignored by cuPyNumeric.

    Returns
    -------
    broadcast : array
        A readonly view on the original array with the given shape.
        It is typically not contiguous.
        Furthermore, more than one element of a broadcasted array
        may refer to a single memory location.

    See Also
    --------
    numpy.broadcast_to

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    return _broadcast_to(arr, shape, subok)


def _broadcast_arrays(
    arrs: list[ndarray], subok: bool = False
) -> list[ndarray]:
    # create an arry object w/ options passed from 'broadcast' routines
    arrays = [array(arr, copy=False, subok=subok) for arr in arrs]
    # check if the broadcast can happen in the input list of arrays
    shapes = [arr.shape for arr in arrays]
    out_shape = broadcast_shapes(*shapes)
    # broadcast to the final shape
    arrays = [_broadcast_to(arr, out_shape, subok) for arr in arrays]
    return arrays


def broadcast_arrays(*args: Any, subok: bool = False) -> list[ndarray]:
    """

    Broadcast any number of arrays against each other.

    Parameters
    ----------
    `*args` : array_likes
        The arrays to broadcast.

    subok : bool, optional
        This option is ignored by cuPyNumeric

    Returns
    -------
    broadcasted : list of arrays
        These arrays are views on the original arrays.
        They are typically not contiguous.
        Furthermore, more than one element of a broadcasted array
        may refer to a single memory location.
        If you need to write to the arrays, make copies first.

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    arrs = [convert_to_cupynumeric_ndarray(arr) for arr in args]
    return _broadcast_arrays(arrs, subok=subok)


class broadcast:
    """Produce an object that broadcasts input parameters against one another.
    It has shape and nd properties and may be used as an iterator.

    Parameters
    ----------
    `*arrays` : array_likes
        The arrays to broadcast.

    Returns
    -------
    b: broadcast
        Broadcast the input parameters against one another, and return an
        object that encapsulates the result. Amongst others, it has shape
        and nd properties, and may be used as an iterator.

    """

    def __init__(self, *arrays: Any) -> None:
        arrs = [convert_to_cupynumeric_ndarray(arr) for arr in arrays]
        broadcasted = _broadcast_arrays(arrs)
        self._iters = tuple(arr.flat for arr in broadcasted)
        self._index = 0
        self._shape = broadcasted[0].shape
        self._size = np.prod(self.shape, dtype=int)

    def __iter__(self) -> broadcast:
        self._index = 0
        return self

    def __next__(self) -> Any:
        if self._index < self.size:
            result = tuple(each[self._index] for each in self._iters)
            self._index += 1
            return result
        raise StopIteration

    def reset(self) -> None:
        """Reset the broadcasted result's iterator(s)."""
        self._index = 0

    @property
    def index(self) -> int:
        """current index in broadcasted result"""
        return self._index

    @property
    def iters(self) -> tuple[Iterable[Any], ...]:
        """tuple of iterators along self's "components." """
        return self._iters

    @property
    def numiter(self) -> int:
        """Number of iterators possessed by the broadcasted result."""
        return len(self._iters)

    @property
    def nd(self) -> int:
        """Number of dimensions of broadcasted result."""
        return self.ndim

    @property
    def ndim(self) -> int:
        """Number of dimensions of broadcasted result."""
        return len(self.shape)

    @property
    def shape(self) -> NdShape:
        """Shape of broadcasted result."""
        return self._shape

    @property
    def size(self) -> int:
        """Total size of broadcasted result."""
        return int(self._size)


@add_boilerplate("a")
def expand_dims(
    a: ndarray, axis: int | tuple[int, ...] | list[int]
) -> ndarray:
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or tuple of ints
        Position in the expanded axes where the new axis (or axes) is placed.

    Returns
    -------
    result : ndarray
        View of `a` with the number of dimensions increased.

    See Also
    --------
    squeeze : The inverse operation, removing singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    atleast_1d, atleast_2d, atleast_3d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if isinstance(axis, int):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim
    normalized_axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(a.shape)
    shape = [
        1 if ax in normalized_axis else next(shape_it)
        for ax in range(out_ndim)
    ]

    return a.reshape(shape)
