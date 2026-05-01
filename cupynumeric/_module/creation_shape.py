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

import math
import operator
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate
from ..types import NdShapeLike

if TYPE_CHECKING:
    import numpy.typing as npt


def _uninitialized(
    shape: NdShapeLike, dtype: npt.DTypeLike = np.float64
) -> ndarray:
    return ndarray(shape=shape, dtype=dtype)


add_boilerplate("a")


class _MGridClass:
    """
    Slicing an instance of this class returns a multi-dimensional
    grid of coordinates, a.k.a. a "meshgrid".

    Complex value step are not supported, unlike NumPy.
    """

    def __init__(self) -> None:
        return

    def __getitem__(self, key: slice | Sequence[slice]) -> ndarray:
        """
        Parameters
        ----------
        key : Sequence[slice]
            The slices over which the meshgrid is constructed.

        Returns
        -------
        out : ndarray
            The meshgrid of coordinates. out has len(key) + 1
            dimensions, where the first dimension is of size len(key).
            out[i] represents the meshgrid for the i-th dimension.

        See Also
        --------
        numpy.mgrid

        Availability
        ------------
        Multiple GPUs, Multiple CPUs
        """

        if isinstance(key, slice) or len(key) == 1:
            from .creation_ranges import arange

            if not isinstance(key, slice):
                key = key[0]

            return arange(key.start if key.start else 0, key.stop, key.step)

        # Process slices to determine output shape and scalars for task
        shape = [len(key)]
        slice_dtypes = []
        cleaned_slices = []
        for s in key:
            # if no end, treat start as end
            start = s.start
            stop = s.stop
            step = s.step

            if stop is None:
                raise ValueError("slice stop cannot be None for mgrid")
            if start is None:
                start = 0
            if step is None:
                step = 1

            cleaned_slices.append(slice(start, stop, step))
            shape.append(int(math.ceil((stop - start) / step)))
            slice_dtypes.append(np.result_type(start, stop, step))

        # determine overarching dtype for meshgrid
        dtype = np.result_type(*slice_dtypes)

        # return empty array if empty shape
        for dim in shape:
            if dim == 0:
                return ndarray(tuple(shape), dtype=dtype)

        # otherwise, create a meshgrid and fill it
        result = ndarray(tuple(shape), dtype=dtype)
        result._thunk.mgrid(cleaned_slices)

        return result


mgrid = _MGridClass()


def _uninitialized_like(
    a: ndarray,
    dtype: npt.DTypeLike | None = None,
    shape: NdShapeLike | None = None,
) -> ndarray:
    shape = a.shape if shape is None else shape
    dtype = a.dtype if dtype is None else np.dtype(dtype)
    return ndarray._from_inputs(shape, dtype=dtype, inputs=(a,))


def empty(
    shape: NdShapeLike,
    dtype: npt.DTypeLike = np.float64,
    *,
    device: Any | None = None,
) -> ndarray:
    """
    empty(shape, dtype=float, *, device=None)

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the empty array.
    dtype : data-type, optional
        Desired output data-type for the array. Default is
        ``cupynumeric.float64``.
    device : None, optional
        Array API device selector. cuPyNumeric currently supports only
        ``None``.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape and dtype.

    See Also
    --------
    numpy.empty

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if device is not None:
        raise ValueError(
            "cuPyNumeric's Array API namespace currently only supports "
            f"device=None, got device={device!r}"
        )
    arr = _uninitialized(shape=shape, dtype=dtype)
    # FIXME: we need to initialize this to 0 temporarily until
    # we can check if LogicalStore is initialized
    arr.fill(0)
    return arr


@add_boilerplate("a")
def empty_like(
    a: ndarray,
    dtype: npt.DTypeLike | None = None,
    shape: NdShapeLike | None = None,
) -> ndarray:
    """

    empty_like(prototype, dtype=None)

    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same shape and type as
        `prototype`.

    See Also
    --------
    numpy.empty_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    arr = _uninitialized_like(a, dtype, shape)
    # FIXME: we need to initialize this to 0 temporarily until
    # we can check if LogicalStore is initialized.
    # See issue: https://github.com/nv-legate/cupynumeric.internal/issues/751
    arr.fill(0)
    return arr


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: npt.DTypeLike | None = np.float64,
) -> ndarray:
    """

    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.

    Returns
    -------
    I : ndarray
      An array  of shape (N, M) where all elements are equal to zero, except
      for the `k`-th diagonal, whose values are equal to one.

    See Also
    --------
    numpy.eye

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Validate N
    N = operator.index(N)
    if N < 0:
        raise ValueError("negative dimensions are not allowed")

    # Validate M
    if M is not None:
        M = operator.index(M)
        if M < 0:
            raise ValueError("negative dimensions are not allowed")
    else:
        M = N

    resolved_dtype = np.float64 if dtype is None else np.dtype(dtype)
    k = operator.index(k)
    result = ndarray((N, M), resolved_dtype)
    result._thunk.eye(k)
    return result


def identity(n: int, dtype: npt.DTypeLike = float) -> ndarray:
    """

    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one, and all other
        elements 0.

    See Also
    --------
    numpy.identity

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return eye(N=n, M=n, dtype=dtype)


def ones(shape: NdShapeLike, dtype: npt.DTypeLike = np.float64) -> ndarray:
    """

    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the new array.
    dtype : data-type, optional
        The desired data-type for the array. Default is `cupynumeric.float64`.

    Returns
    -------
    out : ndarray
        Array of ones with the given shape and dtype.

    See Also
    --------
    numpy.ones

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return full(shape, 1, dtype=dtype)


def ones_like(
    a: ndarray,
    dtype: npt.DTypeLike | None = None,
    shape: NdShapeLike | None = None,
) -> ndarray:
    """

    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    numpy.ones_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 1, dtype=usedtype, shape=shape)


def zeros(shape: NdShapeLike, dtype: npt.DTypeLike = np.float64) -> ndarray:
    """
    zeros(shape, dtype=float)

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the new array.
    dtype : data-type, optional
        The desired data-type for the array.  Default is `cupynumeric.float64`.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape and dtype.

    See Also
    --------
    numpy.zeros

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
    return full(shape, 0, dtype=dtype)


def zeros_like(
    a: ndarray,
    dtype: npt.DTypeLike | None = None,
    shape: NdShapeLike | None = None,
) -> ndarray:
    """

    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    numpy.zeros_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 0, dtype=usedtype, shape=shape)


def full(
    shape: NdShapeLike, value: Any, dtype: npt.DTypeLike | None = None
) -> ndarray:
    """

    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the new array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
         `cupynumeric.array(fill_value).dtype`.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape and dtype.

    See Also
    --------
    numpy.full

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is None:
        val = np.array(value)
    else:
        dtype = np.dtype(dtype)
        val = np.array(value, dtype=dtype)
    if np.dtype(dtype).itemsize == 1 and value > 255:
        raise OverflowError(f"Value {value} out of bounds for {dtype}")
    result = _uninitialized(shape, dtype=val.dtype)
    result._thunk.fill(val)
    return result


def full_like(
    a: ndarray,
    value: int | float,
    dtype: npt.DTypeLike | None = None,
    shape: NdShapeLike | None = None,
) -> ndarray:
    """

    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    numpy.full_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    if np.dtype(dtype).itemsize == 1 and value > 255:
        raise OverflowError(f"Value {value} out of bounds for {dtype}")
    result = _uninitialized_like(a, dtype=dtype, shape=shape)
    val = np.array(value).astype(dtype)
    result._thunk.fill(val)
    return result
