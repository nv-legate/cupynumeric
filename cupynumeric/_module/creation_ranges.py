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

import math
import warnings
from types import EllipsisType
from typing import TYPE_CHECKING

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate
from .._module.array_dimension import broadcast_arrays
from .._module.creation_data import asarray
from .._ufunc.floating import floor, isfinite, isinf
from .._ufunc.math import power
from .._utils import is_np2
from .array_dimension import expand_dims
from .logic_truth import any

if TYPE_CHECKING:
    import numpy.typing as npt

if is_np2:
    from numpy.exceptions import AxisError
else:
    from numpy import AxisError  # type: ignore[no-redef,attr-defined]

_builtin_max = max


def arange(
    start: int | float = 0,
    stop: int | float | None = None,
    step: int | float | None = 1,
    dtype: npt.DTypeLike | None = None,
) -> ndarray:
    """
    arange([start,] stop[, step,], dtype=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `cupynumeric.linspace` for these cases.

    Parameters
    ----------
    start : int or float, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : int or float
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : int or float, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : data-type
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    numpy.arange

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    if dtype is None:
        dtype = np.result_type(start, stop, step)
    else:
        dtype = np.dtype(dtype)

    N = math.ceil((stop - start) / step)
    result = ndarray((_builtin_max(0, N),), dtype)
    result._thunk.arange(start, stop, step)
    return result


@add_boilerplate("start", "stop")
def linspace(
    start: ndarray,
    stop: ndarray,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: npt.DTypeLike | None = None,
    axis: int = 0,
) -> ndarray | tuple[ndarray, float | ndarray]:
    """

    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : data-type, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float or ndarray, optional
        Only returned if `retstep` is True

        Size of spacing between samples.

    See Also
    --------
    numpy.linspace

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)
    div = (num - 1) if endpoint else num

    common_kind = np.result_type(start.dtype, stop.dtype).kind
    dt = np.complex128 if common_kind == "c" else np.float64
    if dtype is None:
        dtype = dt

    delta = stop - start
    y = arange(0, num, dtype=dt)

    out: tuple[int | EllipsisType | slice, ...]

    # Reshape these arrays into dimensions that allow them to broadcast
    if delta.ndim > 0:
        if axis is None or axis == 0:
            # First dimension
            y = y.reshape((-1,) + (1,) * delta.ndim)
            # Nothing else needs to be reshaped here because
            # they should all broadcast correctly with y
            if endpoint and num > 1:
                out = (-1,)
        elif axis == -1 or axis == delta.ndim:
            # Last dimension
            y = y.reshape((1,) * delta.ndim + (-1,))
            if endpoint and num > 1:
                out = (Ellipsis, -1)
            # Extend everything else with extra dimensions of 1 at the end
            # so that they can broadcast with y
            delta = delta.reshape(delta.shape + (1,))
            start = start.reshape(start.shape + (1,))
        elif axis < delta.ndim:
            # Somewhere in the middle
            y = y.reshape((1,) * axis + (-1,) + (1,) * (delta.ndim - axis))
            # Start array might be smaller than delta because of broadcast
            startax = start.ndim - len(delta.shape[axis:])
            start = start.reshape(
                start.shape[0:startax] + (1,) + start.shape[startax:]
            )
            if endpoint and num > 1:
                out = (Ellipsis, -1) + (slice(None, None, None),) * len(
                    delta.shape[axis:]
                )
            delta = delta.reshape(
                delta.shape[0:axis] + (1,) + delta.shape[axis:]
            )
        else:
            raise ValueError(
                "axis "
                + str(axis)
                + " is out of bounds for array of dimension "
                + str(delta.ndim + 1)
            )
    else:
        out = (-1,)
    # else delta is a scalar so start must be also
    # therefore it will trivially broadcast correctly

    step: float | ndarray
    if div > 0:
        step = delta / div
        if delta.ndim == 0:
            y *= step
        else:
            y = y * step
    else:
        # sequences with 0 items or 1 item with endpoint=True (i.e. div <= 0)
        # have an undefined step
        step = np.nan
        if delta.ndim == 0:
            y *= delta
        else:
            y = y * delta

    y += start.astype(y.dtype, copy=False)

    if endpoint and num > 1:
        y[out] = stop.astype(y.dtype, copy=False)

    if np.issubdtype(dtype, np.integer):
        floor(y, out=y)

    if retstep:
        return y.astype(dtype, copy=False), step
    else:
        return y.astype(dtype, copy=False)


@add_boilerplate()
def meshgrid(
    *xi: ndarray, copy: bool = True, sparse: bool = False, indexing: str = "xy"
) -> tuple[ndarray, ...]:
    """
    Return a tuple of coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
        If True the shape of the returned coordinate array for dimension *i*
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
        intended to be use with broadcasting.  When all
        coordinates are used in an expression, broadcasting still leads to a
        fully-dimensonal result array.

        Default is False.
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.

    Returns
    -------
    X1, X2,..., XN : tuple of ndarrays
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,
        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = np.meshgrid(x, y, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = np.meshgrid(x, y, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    ndim = len(xi)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [
        asarray(x).reshape(s0[:i] + (-1,) + s0[i + 1 :])
        for i, x in enumerate(xi)
    ]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].swapaxes(0, 1)
        output[1] = output[1].swapaxes(0, 1)

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = broadcast_arrays(*output, subok=True)

    if copy:
        output = [x.copy() for x in output]

    return tuple(output)


@add_boilerplate("start", "stop", "base")
def logspace(
    start: ndarray,
    stop: ndarray,
    num: int = 50,
    endpoint: bool = True,
    base: ndarray | None = None,
    dtype: npt.DTypeLike | None = None,
    axis: int = 0,
) -> ndarray:
    """
    Return numbers spaced evenly on a log scale.
    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).
    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : int, optional
        Number of samples to generate.  Default is 50.
    endpoint : bool, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : array_like, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : data-type, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.
    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.
    See Also
    --------
    numpy.logspace
    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if base is None:
        base = asarray(10.0)

    # Input validation
    if not isinstance(num, int):
        raise TypeError("'num' must be an integer")

    if any(base < 0):
        warnings.warn(
            "invalid value encountered in power", RuntimeWarning, stacklevel=2
        )

    # Validate axis parameter for scalar inputs
    max_ndim = max(np.ndim(start), np.ndim(stop))
    if max_ndim == 0:
        if axis != 0:
            raise AxisError(
                f"axis {axis} is out of bounds for array of dimension 1"
            )

    # Handle non-scalar base
    if base.ndim > 0:
        start, stop, base = broadcast_arrays(start, stop, base)
        base = expand_dims(base, axis=axis)

    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    result = power(base, y)

    if np.any(isfinite(base) & isfinite(y) & isinf(result)):
        warnings.warn(
            "overflow encountered in power", RuntimeWarning, stacklevel=2
        )

    # Handle dtype conversion
    if dtype is None:
        return result
    return result.astype(dtype, copy=False)
