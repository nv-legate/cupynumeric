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
from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from .._ufunc.comparison import greater, less
from .._ufunc.floating import isinf, isnan
from .._utils.array import max_identity, min_identity
from ..config import ConvolveMethod
from .indexing import putmask

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import ConvolveMethod as ConvolveMethodType, ConvolveMode

import numpy as np


@add_boilerplate("a", "v")
def convolve(
    a: ndarray,
    v: ndarray,
    mode: ConvolveMode = "full",
    method: ConvolveMethodType = "auto",
) -> ndarray:
    """

    Returns the discrete, linear convolution of two ndarrays.

    If `a` and `v` are both 1-D and `v` is longer than `a`, the two are
    swapped before computation. For N-D cases, the arguments are never
    swapped.

    Parameters
    ----------
    a : (N,) array_like
        First input ndarray.
    v : (M,) array_like
        Second input ndarray.
    mode : ``{'full', 'valid', 'same'}``, optional
        'same':
          The output is the same size as `a`, centered with respect to
          the 'full' output. (default)

        'full':
          The output is the full discrete linear convolution of the inputs.

        'valid':
          The output consists only of those elements that do not
          rely on the zero-padding. In 'valid' mode, either `a` or `v`
          must be at least as large as the other in every dimension.
    method : ``{'auto', 'direct', 'fft'}``, optional
        A string indicating which method to use to calculate the convolution.

        'auto':
         Automatically chooses direct or Fourier method based on an estimate
         of which is faster (default)

        'direct':
         The convolution is determined directly from sums, the definition of
         convolution

        'fft':
          The Fourier Transform is used to perform the convolution

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    numpy.convolve

    Notes
    -----
    The current implementation only supports the 'same' mode.

    Unlike `numpy.convolve`, `cupynumeric.convolve` supports N-dimensional
    inputs, but it follows NumPy's behavior for 1-D inputs.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if mode != "same":
        raise NotImplementedError("Only support mode='same'")

    if a.ndim != v.ndim:
        raise RuntimeError("Arrays should have the same dimensions")
    elif a.ndim > 3:
        raise NotImplementedError(f"{a.ndim}-D arrays are not yet supported")

    if a.ndim == 1 and a.size < v.size:
        v, a = a, v

    if not hasattr(ConvolveMethod, method.upper()):
        raise ValueError(
            "Acceptable method flags are 'auto', 'direct', or 'fft'."
        )

    if a.dtype != v.dtype:
        v = v.astype(a.dtype)
    out = ndarray(
        shape=a.shape,
        dtype=a.dtype,
        inputs=(a, v),
    )
    out._thunk.convolve(a._thunk, v._thunk, mode, method)
    return out


@add_boilerplate("a")
def clip(
    a: ndarray,
    a_min: int | float | npt.ArrayLike | None,
    a_max: int | float | npt.ArrayLike | None,
    out: ndarray | None = None,
) -> ndarray:
    """

    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like or None
        Minimum value. If None, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        None.
    a_max : scalar or array_like or None
        Maximum value. If None, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        None. If `a_min` or `a_max` are array_like, then the three
        arrays will be broadcasted to match their shapes.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    See Also
    --------
    numpy.clip

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.clip(a_min, a_max, out=out)


@add_boilerplate("a")
def nan_to_num(
    a: ndarray,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> ndarray:
    """
    Replace NaN with zero and infinity with large finite numbers.

    If `x` is inexact, NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    Parameters
    ----------
    a : array_like
        Input data.
    copy : bool, optional
        Whether to create a copy of `x` (True) or to replace values
        in-place (False). The in-place operation only occurs if
        casting to an array does not require a copy.
        Default is True.
    nan : int, float, optional
        Value to be used to fill NaN values. If no value is passed
        then NaN values will be replaced with 0.0.
    posinf : int, float, optional
        Value to be used to fill positive infinity values. If no value is
        passed then positive infinity values will be replaced with a very
        large number.
    neginf : int, float, optional
        Value to be used to fill negative infinity values. If no value is
        passed then negative infinity values will be replaced with a very
        small (or negative) number.

    Returns
    -------
    out : ndarray
        A copy of `x`, with the non-finite values replaced. If `copy` is False,
        this may be `x` itself.

    See Also
    --------
    numpy.nan_to_num

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # please see https://github.com/nv-legate/cupynumeric.internal/issues/791
    if np.issubdtype(a.dtype, np.complexfloating) and not copy:
        print(
            "Warning: cupynumeric doesn't currently support copy=False for \
            `nan_to_num` when a is complex type, falling-back to numpy"
        )
        return convert_to_cupynumeric_ndarray(
            np.nan_to_num(
                np.array(a), copy=copy, nan=nan, posinf=posinf, neginf=neginf
            )
        )

    # If not inexact, return as is or copy
    if not np.issubdtype(a.dtype, np.inexact):
        return a.copy() if copy else a

    # Create output array and set default values
    out = a.copy() if copy else a
    dtype = out.real.dtype if out.dtype.kind == "c" else out.dtype

    # Convert values to float to match expected types
    posinf_val = (
        float(min_identity(dtype)) if posinf is None else float(posinf)
    )
    neginf_val = (
        float(max_identity(dtype)) if neginf is None else float(neginf)
    )
    nan_val = float(nan)

    def replace_special(x: ndarray) -> None:
        # Create masks for special values
        inf_mask = isinf(x)
        pos_mask = inf_mask & greater(x, 0)
        neg_mask = inf_mask & less(x, 0)
        nan_mask = isnan(x)

        # Replace values using where operations
        putmask(x, nan_mask, x.dtype.type(nan_val))
        putmask(x, pos_mask, x.dtype.type(posinf_val))
        putmask(x, neg_mask, x.dtype.type(neginf_val))

    if out.dtype.kind == "c":
        real_part = out.real
        imag_part = out.imag

        replace_special(real_part)
        replace_special(imag_part)
        out = real_part + 1j * imag_part
    else:
        # Handle real numbers directly
        replace_special(out)

    return out
