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

from .._array.util import convert_to_cupynumeric_ndarray
from .._ufunc.comparison import logical_and
from .._ufunc.floating import isinf, signbit

if TYPE_CHECKING:
    from .._array.array import ndarray


def isneginf(x: ndarray, out: ndarray | None = None) -> ndarray:
    """

    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True. The
        return value `out` is then a reference to that array.

    See Also
    --------
    numpy.isneginf

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cupynumeric_ndarray(x)
    if out is not None:
        out = convert_to_cupynumeric_ndarray(out, share=True)
    rhs1 = isinf(x)
    rhs2 = signbit(x)
    return logical_and(rhs1, rhs2, out=out)


def isposinf(x: ndarray, out: ndarray | None = None) -> ndarray:
    """

    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a boolean array is returned
        with values True where the corresponding element of the input is
        positive infinity and values False where the element of the input is
        not positive infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as zeros
        and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    numpy.isposinf

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cupynumeric_ndarray(x)
    if out is not None:
        out = convert_to_cupynumeric_ndarray(out, share=True)
    rhs1 = isinf(x)
    rhs2 = ~signbit(x)
    return logical_and(rhs1, rhs2, out=out)
