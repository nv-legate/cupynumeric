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

from typing import Sequence

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from .._module.array_rearrange import roll
from ..config import FFT_C2C, FFT_Z2Z, FFTCode, FFTDirection, FFTNormalization


def _sanitize_user_axes(
    a: ndarray,
    s: Sequence[int] | None,
    axes: Sequence[int] | None,
    is_c2r: bool = False,
) -> tuple[list[int], Sequence[int]]:
    if s is None:
        user_shape = False
        if axes is None:
            s = list(a.shape)
        else:
            s = [a.shape[ax] for ax in axes]
    else:
        user_shape = True
        s = list(s)
    if axes is None:
        axes = list(range(len(s)))
    if is_c2r and not user_shape:
        s[-1] = 2 * (a.shape[axes[-1]] - 1)
    return s, axes


def _operate_by_axes(a: ndarray, axes: Sequence[int]) -> bool:
    return (
        len(axes) != len(set(axes))
        or len(axes) != a.ndim
        or tuple(axes) != tuple(sorted(axes))
    )


@add_boilerplate("a")
def fft(
    a: ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
) -> ndarray:
    """
    Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [CT].

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Notes
    -----
    This is really `fftn` with different defaults.
    For more details see `fftn`.
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.fft

    Availability
    ------------
    Multiple GPUs (partial)
    """
    s = (n,) if n is not None else None
    axes = (axis,) if axis is not None else None
    return fftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def fft2(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> ndarray:
    """
    Compute the 2-dimensional discrete Fourier Transform.

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    a : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.fft2

    Availability
    --------
    Multiple GPUs (partial)
    """
    return fftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def fftn(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> ndarray:
    """
    Compute the N-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any number of axes in an *M*-dimensional array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.fftn

    Availability
    --------
    Multiple GPUs (partial)
    """
    if a.dtype == np.float32:
        a = a.astype(np.complex64)
    elif a.dtype == np.float64:
        a = a.astype(np.complex128)

    fft_type = None
    if a.dtype == np.complex128:
        fft_type = FFT_Z2Z
    elif a.dtype == np.complex64:
        fft_type = FFT_C2C
    else:
        raise TypeError(("FFT input not supported (missing a conversion?)"))
    return a.fft(
        s=s,
        axes=axes,
        kind=fft_type,
        direction=FFTDirection.FORWARD,
        norm=norm,
    )


@add_boilerplate("a")
def ifft(
    a: ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
) -> ndarray:
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(a)) == a`` to within numerical accuracy.
    For a general description of the algorithm and definitions,
    see `numpy.fft`.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``a[0]`` should contain the zero frequency term,
    * ``a[1:n//2]`` should contain the positive-frequency terms,
    * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``A[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together. See `numpy.fft` for details.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
        See notes about padding issues.
    axis : int, optional
        Axis over which to compute the inverse DFT.  If not given, the last
        axis is used.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Notes
    -----
    This is really `ifftn` with different defaults.
    For more details see `ifftn`.
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.ifft

    Availability
    --------
    Multiple GPUs (partial)
    """
    s = (n,) if n is not None else None
    computed_axis = (axis,) if axis is not None else None
    return ifftn(a=a, s=s, axes=computed_axis, norm=norm)


@add_boilerplate("a")
def ifft2(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> ndarray:
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the 2-dimensional discrete Fourier
    Transform over any number of axes in an M-dimensional array by means of
    the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``
    to within numerical accuracy.  By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e. it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.ifft2

    Availability
    --------
    Multiple GPUs (partial)
    """
    return ifftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def ifftn(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> ndarray:
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform over any number of axes in an M-dimensional array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(a)) == a`` to within numerical accuracy.
    For a description of the definitions and conventions used, see `numpy.fft`.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e. it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `a`,
        as explained in the parameters section above.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.ifftn

    Availability
    --------
    Multiple GPUs (partial)
    """
    # Convert to complex if real
    if a.dtype == np.float32:
        a = a.astype(np.complex64)
    elif a.dtype == np.float64:
        a = a.astype(np.complex128)
    # Check for types
    fft_type = None
    if a.dtype == np.complex128:
        fft_type = FFT_Z2Z
    elif a.dtype == np.complex64:
        fft_type = FFT_C2C
    else:
        raise TypeError("FFT input not supported (missing a conversion?)")
    return a.fft(
        s=s,
        axes=axes,
        kind=fft_type,
        direction=FFTDirection.INVERSE,
        norm=norm,
    )


@add_boilerplate("a")
def rfft(
    a: ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
) -> ndarray:
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        Number of points along transformation axis in the input to use.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        If `n` is even, the length of the transformed axis is ``(n/2)+1``.
        If `n` is odd, the length is ``(n+1)/2``.

    Notes
    ------
    This is really `rfftn` with different defaults.
    For more details see `rfftn`.
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.rfft

    Availability
    --------
    Multiple GPUs (partial)
    """
    s = (n,) if n is not None else None
    computed_axis = (axis,) if axis is not None else None
    return rfftn(a=a, s=s, axes=computed_axis, norm=norm)


@add_boilerplate("a")
def rfft2(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> ndarray:
    """
    Compute the 2-dimensional FFT of a real array.

    Parameters
    ----------
    a : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    Notes
    ------
    This is really `rfftn` with different defaults.
    For more details see `rfftn`.
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.rfft2

    Availability
    --------
    Multiple GPUs (partial)
    """
    return rfftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def rfftn(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> ndarray:
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    This function computes the N-dimensional discrete Fourier Transform over
    any number of axes in an M-dimensional real array by means of the Fast
    Fourier Transform (FFT).  By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    a : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.rfftn

    Availability
    --------
    Multiple GPUs (partial)
    """
    # Convert to real if complex
    if a.dtype != np.float32 and a.dtype != np.float64:
        a = a.real
    # Retrieve type
    fft_type = FFTCode.real_to_complex_code(a.dtype)

    s, axes = _sanitize_user_axes(a, s, axes)

    # Operate by axes
    if _operate_by_axes(a, axes):
        r2c = a.fft(
            s=(s[-1],),
            axes=(axes[-1],),
            kind=fft_type,
            direction=FFTDirection.FORWARD,
            norm=norm,
        )
        if len(axes) > 1:
            return r2c.fft(
                s=s[0:-1],
                axes=axes[0:-1],
                kind=fft_type.complex,
                direction=FFTDirection.FORWARD,
                norm=norm,
            )
        else:
            return r2c
    # Operate as a single FFT
    else:
        return a.fft(
            s=s,
            axes=axes,
            kind=fft_type,
            direction=FFTDirection.FORWARD,
            norm=norm,
        )


@add_boilerplate("a")
def irfft(
    a: ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
) -> ndarray:
    """
    Computes the inverse of `rfft`.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier Transform of real input computed by `rfft`.
    In other words, ``irfft(rfft(a), len(a)) == a`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by `rfft`, i.e. the
    real zero-frequency term followed by the complex positive frequency terms
    in order of increasing frequency.  Since the discrete Fourier Transform of
    real input is Hermitian-symmetric, the negative frequency terms are taken
    to be the complex conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary.  If the
        input is longer than this, it is cropped.  If it is shorter than this,
        it is padded with zeros.  If `n` is not given, it is taken to be
        ``2*(m-1)`` where ``m`` is the length of the input along the axis
        specified by `axis`.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified.

    Notes
    ------
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.irfft

    Availability
    --------
    Multiple GPUs (partial)
    """
    s = (n,) if n is not None else None
    computed_axis = (axis,) if axis is not None else None
    return irfftn(a=a, s=s, axes=computed_axis, norm=norm)


@add_boilerplate("a")
def irfft2(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> ndarray:
    """
    Computes the inverse of `rfft2`.

    Parameters
    ----------
    a : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.


    Notes
    ------
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.irfft2

    Availability
    --------
    Multiple GPUs (partial)
    """
    return irfftn(a=a, s=s, axes=axes, norm=norm)


@add_boilerplate("a")
def irfftn(
    a: ndarray,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> ndarray:
    """
    Computes the inverse of `rfftn`.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform for real input over any number of axes in an
    M-dimensional array by means of the Fast Fourier Transform (FFT).  In
    other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e. as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Parameters
    ----------
    a : array_like
        Input array.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped.  If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to
        be ``2*(m-1)`` where ``m`` is the length of the input along that axis.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `a`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given.  In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
        length of the final transformed axis of the input.  To get an odd
        number of output points in the final axis, `s` must be specified.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See Also
    --------
    numpy.fft.irfftn

    Availability
    --------
    Multiple GPUs (partial)
    """
    # Convert to complex if real
    if a.dtype == np.float32:
        a = a.astype(np.complex64)
    elif a.dtype == np.float64:
        a = a.astype(np.complex128)
    # Retrieve type
    fft_type = FFTCode.complex_to_real_code(a.dtype)

    s, axes = _sanitize_user_axes(a, s, axes, is_c2r=True)

    # Operate by axes
    if _operate_by_axes(a, axes):
        if len(axes) > 1:
            c2r = a.fft(
                s=s[0:-1],
                axes=axes[0:-1],
                kind=fft_type.complex,
                direction=FFTDirection.INVERSE,
                norm=norm,
            )
            if not isinstance(c2r, ndarray):
                c2r = convert_to_cupynumeric_ndarray(c2r)  # type: ignore [unreachable] # noqa: E501
        else:
            c2r = a
        return c2r.fft(
            s=(s[-1],),
            axes=(axes[-1],),
            kind=fft_type,
            direction=FFTDirection.INVERSE,
            norm=norm,
        )
    # Operate as a single FFT
    else:
        return a.fft(
            s=s,
            axes=axes,
            kind=fft_type,
            direction=FFTDirection.INVERSE,
            norm=norm,
        )


@add_boilerplate("a")
def hfft(
    a: ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
) -> ndarray:
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
    spectrum.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output. For `n` output
        points, ``n//2 + 1`` input points are necessary.  If the input is
        longer than this, it is cropped.  If it is shorter than this, it is
        padded with zeros.  If `n` is not given, it is taken to be ``2*(m-1)``
        where ``m`` is the length of the input along the axis specified by
        `axis`.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last
        axis is used.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*m - 2`` where ``m`` is the length of the transformed axis of
        the input. To get an odd number of output points, `n` must be
        specified, for instance as ``2*m - 1`` in the typical case,

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See also
    --------
    numpy.fft.hfft

    Availability
    --------
    Multiple GPUs (partial)
    """
    s = (n,) if n is not None else None
    computed_axis = (axis,) if axis is not None else None
    # Add checks to ensure input is hermitian?
    # Essentially a C2R FFT, with reverse sign
    # (forward transform, forward norm)
    return irfftn(
        a=a.conjugate(),
        s=s,
        axes=computed_axis,
        norm=FFTNormalization.reverse(norm),
    )


@add_boilerplate("a")
def ihfft(
    a: ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
) -> ndarray:
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    a : array_like
        Input array.
    n : int, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use.  If `n` is smaller than
        the length of the input, the input is cropped.  If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : ``{"backward", "ortho", "forward"}``, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    Notes
    ------
    Multi-GPU usage is limited to data parallel axis-wise batching.

    See also
    --------
    numpy.fft.ihfft

    Availability
    --------
    Multiple GPUs (partial)
    """
    s = (n,) if n is not None else None
    computed_axis = (axis,) if axis is not None else None
    # Add checks to ensure input is hermitian?
    # Essentially a R2C FFT, with reverse sign
    # (inverse transform, inverse norm)
    return rfftn(
        a=a, s=s, axes=computed_axis, norm=FFTNormalization.reverse(norm)
    ).conjugate()


integer_types = (int, np.integer)


@add_boilerplate("x")
def fftshift(x: ndarray, axes: int | tuple[int, ...] | None = None) -> ndarray:
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input array.

    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shift: int | tuple[int, ...]
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = tuple(dim // 2 for dim in x.shape)
    elif isinstance(axes, integer_types):
        shift = x.shape[axes] // 2
    else:
        shift = tuple(x.shape[ax] // 2 for ax in axes)

    return roll(x, shift, axes)


@add_boilerplate("x")
def ifftshift(
    x: ndarray, axes: int | tuple[int, ...] | None = None
) -> ndarray:
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : array_like
        Input array.

    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shift: int | tuple[int, ...]
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = tuple(-(dim // 2) for dim in x.shape)
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = tuple(-(x.shape[ax] // 2) for ax in axes)

    return roll(x, shift, axes)
