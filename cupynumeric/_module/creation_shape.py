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

import operator
from typing import TYPE_CHECKING, Any

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


def _uninitialized_like(
    a: ndarray,
    dtype: npt.DTypeLike | None = None,
    shape: NdShapeLike | None = None,
) -> ndarray:
    shape = a.shape if shape is None else shape
    dtype = a.dtype if dtype is None else np.dtype(dtype)
    return ndarray._from_inputs(shape, dtype=dtype, inputs=(a,))


def empty(shape: NdShapeLike, dtype: npt.DTypeLike = np.float64) -> ndarray:
    """
    empty(shape, dtype=float)

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the empty array.
    dtype : data-type, optional
        Desired output data-type for the array. Default is
        ``cupynumeric.float64``.

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
    arr = _uninitialized(shape=shape, dtype=dtype)
    # FIXME: we need to initialize this to 0 temporary until
    # we can check if LogicalStore is initialized
    # otherwise we get error when empty cupynumeric code is
    # taking eager mode
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
    # FIXME: we need to initialize this to 0 temporary until
    # we can check if LogicalStore is initialized
    # otherwise we get error when empty cupynumeric code is
    # taking eager mode. Please see issue
    # https://github.com/nv-legate/cupynumeric.internal/issues/751
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
    if dtype is not None:
        dtype = np.dtype(dtype)
    if M is None:
        M = N
    k = operator.index(k)
    result = ndarray((N, M), dtype)
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
