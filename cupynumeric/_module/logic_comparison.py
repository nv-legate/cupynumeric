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

import numpy as np
from legate.core import Scalar, types as ty

from .._array.thunk import perform_binary_reduction
from .._array.util import add_boilerplate, find_common_type
from ..config import BinaryOpCode
from .creation_shape import empty

if TYPE_CHECKING:
    from .._array.array import ndarray


@add_boilerplate("a", "b")
def allclose(
    a: ndarray,
    b: ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> ndarray:
    """

    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    allclose : ndarray scalar
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    See Also
    --------
    numpy.allclose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuPyNumeric does not support `equal_nan` yet for allclose"
        )
    args = (Scalar(rtol, ty.float64), Scalar(atol, ty.float64))
    return perform_binary_reduction(
        BinaryOpCode.ISCLOSE, a, b, dtype=np.dtype(bool), extra_args=args
    )


@add_boilerplate("a", "b")
def isclose(
    a: ndarray,
    b: ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> ndarray:
    """

    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    Notes
    -----
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    See Also
    --------
    numpy.isclose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuPyNumeric does not support `equal_nan` yet for isclose"
        )

    out_shape = np.broadcast_shapes(a.shape, b.shape)
    out = empty(out_shape, dtype=bool)

    common_type = find_common_type(a, b)
    a = a.astype(common_type)
    b = b.astype(common_type)

    out._thunk.isclose(a._thunk, b._thunk, rtol, atol, equal_nan)
    return out


@add_boilerplate("a1", "a2")
def array_equal(
    a1: ndarray, a2: ndarray, equal_nan: bool = False
) -> bool | ndarray:
    """

    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.
    equal_nan : bool
        Whether to compare NaN's as equal. If the dtype of a1 and a2 is
        complex, values will be considered equal if either the real or the
        imaginary component of a given value is ``nan``.

    Returns
    -------
    b : ndarray scalar
        Returns True if the arrays are equal.

    See Also
    --------
    numpy.array_equal

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuPyNumeric does not support `equal_nan` yet for `array_equal`"
        )

    if a1.shape != a2.shape:
        return False
    return perform_binary_reduction(
        BinaryOpCode.EQUAL, a1, a2, dtype=np.dtype(bool)
    )
