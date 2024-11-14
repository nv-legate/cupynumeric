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

from .._array.array import ndarray
from .._array.util import convert_to_cupynumeric_ndarray
from .creation_shape import full

if TYPE_CHECKING:
    import numpy.typing as npt


def iscomplex(x: ndarray | npt.NDArray[Any]) -> ndarray:
    """

    Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray[bool]
        Output array.

    See Also
    --------
    numpy.iscomplex

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cupynumeric_ndarray(x)
    if x.dtype.kind != "c":
        return full(x.shape, False, dtype=bool)
    else:
        return x.imag != 0


def iscomplexobj(x: ndarray | npt.NDArray[Any]) -> bool:
    """

    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    numpy.iscomplexobj

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if isinstance(x, ndarray):
        return x.dtype.kind == "c"
    else:
        return np.iscomplexobj(x)


def isreal(x: ndarray | npt.NDArray[Any]) -> ndarray:
    """

    Returns a bool array, where True if input element is real.

    If element has complex type with zero complex part, the return value
    for that element is True.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray, bool
        Boolean array of same shape as `x`.


    See Also
    --------
    numpy.isreal

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cupynumeric_ndarray(x)
    if x.dtype.kind != "c":
        return full(x.shape, True, dtype=bool)
    else:
        return x.imag == 0


def isrealobj(x: ndarray) -> bool:
    """

    Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, False if `x` is of a complex type.

    See Also
    --------
    numpy.isrealobj

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return not iscomplexobj(x)


def isscalar(x: ndarray | npt.NDArray[Any]) -> bool:
    """

    Returns True if the type of `element` is a scalar type.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `element` is a scalar type, False if it is not.

    See Also
    --------
    numpy.isscalar

    Notes
    -----
    This function falls back to NumPy for all object types but cuPyNumeric's
    ndarray, which always returns `False`.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # Since the input can be any value, we can't just convert it to cupynumeric
    # ndarray. Instead we check if the input is cupynumeric ndarray and, if
    # not, fall back to Numpy
    if isinstance(x, ndarray):
        return False
    else:
        return np.isscalar(x)
