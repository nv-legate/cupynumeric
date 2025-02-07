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

from typing import Any

import numpy as np

from .._array.array import ndarray
from ..config import WindowOpCode
from .creation_shape import empty, ones


def _create_window(M: int, op_code: WindowOpCode, *args: Any) -> ndarray:
    if M <= 0:
        return empty((0,))
    elif M == 1:
        return ones((1,))

    # TODO: the eager implementation could avoid a copy if we didn't have to
    # create the output ndarray upfront.
    out = ndarray((M,), dtype=np.float64)
    out._thunk.create_window(op_code, M, *args)
    return out


def bartlett(M: int) -> ndarray:
    """

    Return the Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    See Also
    --------
    numpy.bartlett

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    return _create_window(M, WindowOpCode.BARLETT)


def blackman(M: int) -> ndarray:
    """

    Return the Blackman window.

    The Blackman window is a taper formed by using the first three
    terms of a summation of cosines. It was designed to have close to the
    minimal leakage possible.  It is close to optimal, only slightly worse
    than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    numpy.blackman

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    return _create_window(M, WindowOpCode.BLACKMAN)


def hamming(M: int) -> ndarray:
    """

    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    numpy.hamming

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """

    return _create_window(M, WindowOpCode.HAMMING)


def hanning(M: int) -> ndarray:
    """

    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    numpy.hanning

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    return _create_window(M, WindowOpCode.HANNING)


def kaiser(M: int, beta: float) -> ndarray:
    """

    Return the Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    beta : float
        Shape parameter for window.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    numpy.kaiser

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    return _create_window(M, WindowOpCode.KAISER, beta)
