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

from typing import TYPE_CHECKING

from .._array.array import ndarray
from .._array.util import add_boilerplate
from .creation_shape import empty, ones

if TYPE_CHECKING:
    import numpy.typing as npt

import math


@add_boilerplate("v")
def diag(v: ndarray, k: int = 0) -> ndarray:
    """

    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``cupynumeric.diagonal`` if you use
    this function to extract a diagonal and wish to write to the resulting
    array; whether it returns a copy or a view depends on what version of numpy
    you are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    numpy.diag

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if v is None or v.ndim == 0:
        raise ValueError("Input must be 1- or 2-d")
    elif v.ndim == 1:
        return v.diagonal(offset=k, axis1=0, axis2=1, extract=False)
    elif v.ndim == 2:
        return v.diagonal(offset=k, axis1=0, axis2=1, extract=True)
    else:
        raise ValueError("diag requires 1- or 2-D array, use diagonal instead")


def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: npt.DTypeLike = float,
    *,
    like: ndarray | None = None,
) -> ndarray:
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.
    like : array_like
        Reference object to allow the creation of arrays which are not NumPy
        arrays. If an array-like passed in as `like` supports the
        `__array_function__` protocol, the result will be defined by it. In
        this case it ensures the creation of an array object compatible with
        that passed in via this argument.

    Returns
    -------
    tri : ndarray of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.

    See Also
    --------
    numpy.tri

    Notes
    -----
    `like` argument is currently not supported

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # TODO: add support for `like` (see issue #418)
    if like is not None:
        raise ValueError("like parameter is currently not supported")

    try:
        k = int(k)
    except (TypeError, ValueError) as e:
        raise TypeError("k parameter must be an integer.") from e

    def shape_to_int(val: int | float) -> int:
        return int(math.ceil(val)) if isinstance(val, float) else int(val)

    try:
        N = shape_to_int(N)
    except (TypeError, ValueError) as e:
        raise TypeError("N parameter must be an integer.") from e
    if M is None:
        M = N
    else:
        try:
            M = shape_to_int(M)
        except (TypeError, ValueError) as e:
            raise TypeError("M parameter must be integer or None.") from e

    if dtype is None:
        dtype = float

    if N < 0 or M < 0:
        out = empty((max(N, 0), max(M, 0)), dtype=dtype)
        return tril(out, k)

    out = ones((N, M), dtype=dtype)
    return tril(out, k)


@add_boilerplate("m")
def trilu(m: ndarray, k: int, lower: bool) -> ndarray:
    if m.ndim < 1:
        raise TypeError("Array must be at least 1-D")
    shape = m.shape if m.ndim >= 2 else m.shape * 2
    result = ndarray._from_inputs(shape, dtype=m.dtype, inputs=(m,))
    result._thunk.trilu(m._thunk, k, lower)
    return result


def tril(m: ndarray, k: int = 0) -> ndarray:
    """

    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like
        Input array of shape (M, N).
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    numpy.tril

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return trilu(m, k, True)


def triu(m: ndarray, k: int = 0) -> ndarray:
    """

    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    numpy.triu

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return trilu(m, k, False)


@add_boilerplate("v")
def diagflat(v: ndarray, k: int = 0) -> ndarray:
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the k-th diagonal of the
        output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) k giving the number of the diagonal above (below)
        the main.

    Returns
    -------
    out : ndarray
        The 2-D output array.

    See Also
    --------
    numpy.diagflat

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Flatten the input array and use diag to create the output
    return diag(v.ravel(), k=k)
