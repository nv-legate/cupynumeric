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
from legate.core import Scalar

from .._array.thunk import perform_unary_op
from .._array.util import add_boilerplate
from ..config import UnaryOpCode
from .logic_truth import all

if TYPE_CHECKING:
    from .._array.array import ndarray


@add_boilerplate("val")
def real(val: ndarray) -> ndarray:
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The real component of the complex argument. If `val` is real, the type
        of `val` is used for the output.  If `val` has complex elements, the
        returned type is float.

    See Also
    --------
    numpy.real

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return val.real


@add_boilerplate("val")
def imag(val: ndarray) -> ndarray:
    """

    Return the imaginary part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The imaginary component of the complex argument. If `val` is real,
        the type of `val` is used for the output.  If `val` has complex
        elements, the returned type is float.

    See Also
    --------
    numpy.imag

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return val.imag


@add_boilerplate("z")
def angle(z: ndarray, deg: bool = False) -> ndarray:
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.

    See Also
    --------
    numpy.angle

    Notes
    -----
    This function passes the imaginary and real parts of the argument to
    `arctan2` to compute the result; consequently, it follows the convention
    of `arctan2` when the magnitude of the argument is zero.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if z is None:
        raise TypeError("can't compute 'angle' for None")
    extra_args = (Scalar(deg),)
    return perform_unary_op(UnaryOpCode.ANGLE, z, extra_args=extra_args)


@add_boilerplate("a")
def real_if_close(a: ndarray, tol: float = 100) -> ndarray:
    """
    If input is complex with all imaginary parts close to zero, return real
    parts.

    "Close to zero" is defined as tol * (machine epsilon of the type for a).

    Parameters
    ----------
    a : array_like
        Input array.
    tol : float, optional
        Tolerance in machine epsilons for the complex part of the elements
        in the array. If the tolerance is <=1, then the absolute tolerance
        is used. Default is 100.

    Returns
    -------
    out : ndarray
        If a is real, the type of a is used for the output. If a has complex
        elements, the returned type is float.

    See Also
    --------
    real, imag, angle
    numpy.real_if_close

    Notes
    -----
    Machine epsilon varies from machine to machine and between data types but
    Python floats on most platforms have a machine epsilon equal to
    2.2204460492503131e-16. You can use 'np.finfo(float).eps' to print out
    the machine epsilon for floats.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # If the array is already real, return it as-is
    if not issubclass(a.dtype.type, np.complexfloating):
        return a

    # Calculate tolerance threshold
    if tol > 1:
        tol = float(np.finfo(a.real.dtype).eps * tol)

    # Check if all imaginary parts are within tolerance
    if all(abs(a.imag) < tol):
        a = a.real

    return a
