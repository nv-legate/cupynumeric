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
from typing import TYPE_CHECKING, Any

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate
from ..lib.array_utils import normalize_axis_tuple
from .creation_shape import full
from .logic_truth import any
from .stats_order import nanquantile, quantile

if TYPE_CHECKING:
    import numpy.typing as npt


@add_boilerplate("a", "weights")
def average(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    weights: ndarray | None = None,
    returned: bool = False,
    *,
    keepdims: bool = False,
) -> ndarray | tuple[ndarray, ndarray]:
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        axis=None, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::

            avg = sum(a * weights) / sum(weights)

        The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        Return the average along the specified axis. When `returned` is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. `sum_of_weights` is of the
        same type as `retval`. The result dtype follows a general pattern.
        If `weights` is None, the result dtype will be that of `a` , or
        ``float64`` if `a` is integral. Otherwise, if `weights` is not None and
        `a` is non-integral, the result type will be the type of lowest
        precision capable of representing values of both `a` and `weights`. If
        `a` happens to be integral, the previous rules still applies but the
        result dtype will at least be ``float64``.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero.
    ValueError
        When the length of 1D `weights` is not the same as the shape of `a`
        along axis.

    See Also
    --------
    numpy.average

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    clean_axis: tuple[int, ...] | None = None
    if axis is not None:
        clean_axis = normalize_axis_tuple(axis, a.ndim, argname="axis")

    scl: npt.ArrayLike | ndarray = 1
    if weights is None:
        scl = (
            a.size
            if clean_axis is None
            else math.prod([a.shape[i] for i in clean_axis])
        )
        if a.dtype.kind == "i":
            scl = np.float64(scl)
        avg = a.sum(axis=clean_axis, keepdims=keepdims) / scl
    elif weights.shape == a.shape:
        scl = weights.sum(
            axis=clean_axis,
            keepdims=keepdims,
            dtype=(np.dtype(np.float64) if a.dtype.kind == "i" else None),
        )
        if any(scl == 0):
            raise ZeroDivisionError("Weights along axis sum to 0")
        avg = (a * weights).sum(axis=clean_axis, keepdims=keepdims) / scl
    else:
        if clean_axis is None:
            raise ValueError(
                "a and weights must share shape or axis must be specified"
            )
        if weights.ndim != 1 or len(clean_axis) != 1:
            raise ValueError(
                "Weights must be either 1 dimension along single "
                "axis or the same shape as a"
            )
        if weights.size != a.shape[clean_axis[0]]:
            raise ValueError("Weights length does not match axis")

        scl = weights.sum(
            dtype=(np.dtype(np.float64) if a.dtype.kind == "i" else None)
        )
        project_shape = [1] * a.ndim
        project_shape[clean_axis[0]] = -1
        weights = weights.reshape(project_shape)
        if any(scl == 0):
            raise ZeroDivisionError("Weights along axis sum to 0")
        avg = (a * weights).sum(axis=clean_axis[0], keepdims=keepdims) / scl

    if returned:
        if not isinstance(scl, ndarray) or scl.ndim == 0:
            scl = full(avg.shape, scl)
        return avg, scl
    else:
        return avg


@add_boilerplate("a")
def mean(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    keepdims: bool = False,
    where: ndarray | None = None,
) -> ndarray:
    """

    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in the mean.

    Returns
    -------
    m : ndarray
        If `out is None`, returns a new array of the same dtype a above
        containing the mean values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.mean

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.mean(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
    )


@add_boilerplate("a")
def nanmean(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    keepdims: bool = False,
    where: ndarray | None = None,
) -> ndarray:
    """

    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.


    where : array_like of bool, optional
        Elements to include in the mean.

    Returns
    -------
    m : ndarray
        If `out is None`, returns a new array of the same dtype as a above
        containing the mean values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.nanmean

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a._nanmean(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
    )


@add_boilerplate("a")
def var(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: ndarray | None = None,
) -> ndarray:
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of
    a distribution. The variance is computed for the flattened array
    by default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the variance is computed. The default is to
        compute the variance of the flattened array.

        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance. For arrays of integer type
        the default is float64; for arrays of float types
        it is the same as the array type.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have the
        same shape as the expected output, but the type is cast if necessary.
    ddof : int, optional
        “Delta Degrees of Freedom”: the divisor used in the calculation is
        N - ddof, where N represents the number of elements. By default
        ddof is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions of array,
        and selects elements to include in the reduction.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array of the same dtype as above
        containing the variance values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.var

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.var(
        axis=axis,
        dtype=dtype,
        out=out,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
    )


@add_boilerplate("a")
def median(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default,
        axis=None, will compute the median along a flattened version of
        the array.
        If a sequence of axes, the array is first flattened along the
        given axes, then the median is computed along the resulting
        flattened axis.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    numpy median

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    if a is None:
        raise TypeError("'None' is not suported input to 'median'")
    return quantile(
        a,
        0.5,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        keepdims=keepdims,
        method="midpoint",
    )


@add_boilerplate("a")
def nanmedian(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the median along the specified axis, while ignoring NaNs

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default,
        axis=None, will compute the median along a flattened version of
        the array.
        If a sequence of axes, the array is first flattened along the
        given axes, then the median is computed along the resulting
        flattened axis.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    numpy median

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    if a is None:
        raise TypeError("'None' is not suported input to 'nanmedian'")
    return nanquantile(
        a,
        0.5,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        keepdims=keepdims,
        method="midpoint",
    )
