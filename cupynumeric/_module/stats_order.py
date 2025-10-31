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
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from .._array.util import add_boilerplate
from .._ufunc.comparison import logical_not
from .._ufunc.floating import isnan
from ..lib.array_utils import normalize_axis_tuple
from .array_transpose import moveaxis
from .creation_data import asarray
from .creation_shape import zeros
from .ssc_counting import count_nonzero
from .ssc_searching import where
from .ssc_sorting import sort

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    from .._array.array import ndarray


# for the case when axis = tuple (non-singleton)
# reshuffling might have to be done (if tuple is non-consecutive)
# and the src array must be collapsed along that set of axes
#
# args:
#
# arr:    [in] source nd-array on which quantiles are calculated;
# axes_set: [in] tuple or list of axes (indices less than arr dimension);
#
# return: pair: (minimal_index, reshuffled_and_collapsed source array)
def _reshuffle_reshape(
    arr: ndarray, axes_set: Sequence[int]
) -> tuple[int, ndarray]:
    ndim = len(arr.shape)

    sorted_axes = tuple(sorted(axes_set))

    min_dim_index = sorted_axes[0]
    num_axes = len(sorted_axes)
    reshuffled_axes = tuple(range(min_dim_index, min_dim_index + num_axes))

    non_consecutive = sorted_axes != reshuffled_axes
    if non_consecutive:
        arr_shuffled = moveaxis(arr, sorted_axes, reshuffled_axes)
    else:
        arr_shuffled = arr

    # shape_reshuffled = arr_shuffled.shape # debug
    collapsed_shape = np.prod([arr_shuffled.shape[i] for i in reshuffled_axes])

    redimed = tuple(range(0, min_dim_index + 1)) + tuple(
        range(min_dim_index + num_axes, ndim)
    )
    reshaped = tuple(
        [
            collapsed_shape if k == min_dim_index else arr_shuffled.shape[k]
            for k in redimed
        ]
    )

    arr_reshaped = arr_shuffled.reshape(reshaped)
    return (min_dim_index, arr_reshaped)


# Define the gamma and index position for each of the distributions based
# on the paper/NumPy definition.
#
# `pos` is the "virtual index" at which we wish to sample, this is adjusted
# based on the alpha and beta parameters of the methods (which adjust for
# the fact that the distribution is sampled).
#
# `gamma` is weight for each the samples taken into account.  Some methods
# are non-interpolating. `gamma` may be calculated to pick a side, but
# we forward `None` to indicate the non-interpolating nature of the method
# (the result dtype is for example identical to the input one).
#
# `pos` is (to keep with the paper) 1-based index, thus we always subtract 1
# in the following step.

# Discontinuous methods:


# q = quantile input \in [0, 1]
# n = sizeof(array)
# pos = virtual index (often 1 based, to keep with paper)
def _inverted_cdf(q: float, n: int) -> tuple[float, int]:
    pos = q * n
    left = int(pos)

    g = pos - left
    gamma = 1.0 if g > 0 else 0.0

    return (gamma, left - 1)


def _averaged_inverted_cdf(q: float, n: int) -> tuple[float, int]:
    pos = q * n
    left = int(pos)

    g = pos - left
    gamma = 1.0 if g > 0 else 0.5

    return (gamma, left - 1)


_desired_mod_2: int = int(np.lib.NumpyVersion(np.__version__) < "2.0.1")


def _closest_observation(q: float, n: int) -> tuple[None, int]:
    pos = q * n - 0.5
    left = int(pos)

    # The calculation is done in a way that we should to take the next index
    # (gamme = 1) except if we hit it exactly.
    # If we do, we use round-to-even: The final index `left + gamma` should
    # be an even number.  But on older versions of numpy this was an odd
    # number (due to 0 based vs. 1 based indexing used in the rounding).
    if left != pos:
        gamma = 1
    elif left % 2 != _desired_mod_2:
        gamma = 1
    else:
        gamma = 0

    return (None, left - 1 + gamma)


# Continuous methods:


# Parzen method
def _interpolated_inverted_cdf(q: float, n: int) -> tuple[float, int]:
    pos = q * n
    left = int(pos)

    gamma = pos - left
    return (gamma, left - 1)


# Hazen method
def _hazen(q: float, n: int) -> tuple[float, int]:
    pos = q * n + 0.5
    left = int(pos)

    gamma = pos - left
    return (gamma, left - 1)


# Weibull method
def _weibull(q: float, n: int) -> tuple[float, int]:
    pos = q * (n + 1)
    left = int(pos)

    gamma = pos - left
    return (gamma, left - 1)


# Gumbel method
def _linear(q: float, n: int) -> tuple[float, int]:
    pos = q * (n - 1) + 1
    left = int(pos)

    gamma = pos - left
    return (gamma, left - 1)


# Johnson & Kotz method
def _median_unbiased(q: float, n: int) -> tuple[float, int]:
    fract = 1.0 / 3.0
    pos = q * (n + fract) + fract
    left = int(pos)

    gamma = pos - left
    return (gamma, left - 1)


# Blom method
def _normal_unbiased(q: float, n: int) -> tuple[float, int]:
    fract1 = 0.25
    fract2 = 3.0 / 8.0
    pos = q * (n + fract1) + fract2
    left = int(pos)

    gamma = pos - left
    return (gamma, left - 1)


def _lower(q: float, n: int) -> tuple[None, int]:
    pos = q * (n - 1)  # 0 based here
    left = int(pos)
    return (None, left)


def _higher(q: float, n: int) -> tuple[None, int]:
    pos = q * (n - 1)  # 0 based here
    left = int(math.ceil(pos))
    return (None, left)


def _midpoint(q: float, n: int) -> tuple[float, int]:
    pos = q * (n - 1)  # 0 based here
    left = int(pos)
    # Mid-point, unless pos is exact then we use that point.
    gamma = 0.5 if pos != left else 0.0

    return (gamma, left)


def _nearest(q: float, n: int) -> tuple[None, int]:
    pos = np.round(q * (n - 1))  # 0 based here
    left = int(pos)

    return (None, left)


# args:
#
# arr:      [in] source nd-array on which quantiles are calculated;
#                preccondition: assumed sorted!
# q_arr:    [in] quantile input values nd-array;
# axis:     [in] axis along which quantiles are calculated;
# method:   [in] func(q, n) returning (gamma, j),
#                where = array1D.size;
# keepdims: [in] boolean flag specifying whether collapsed axis
#                should be kept as dim=1;
# to_dtype: [in] dtype to convert the result to;
# qs_all:   [in/out] result pass through or created (returned)
#
def _quantile_impl(
    arr: ndarray,
    q_arr: npt.NDArray[Any],
    axis: int | None,
    axes_set: Sequence[int],
    original_shape: tuple[int, ...],
    method: Callable[[float, int], tuple[float | None, int]],
    keepdims: bool,
    to_dtype: np.dtype[Any],
    qs_all: ndarray | None,
) -> ndarray:
    ndims = len(arr.shape)

    if axis is None:
        n = arr.size

        if keepdims:
            remaining_shape = (1,) * len(original_shape)
        else:
            remaining_shape = ()  # only `q_arr` dictates shape;
        # quantile applied to `arr` seen as 1D;
    else:
        n = arr.shape[axis]

        # arr.shape -{axis}; if keepdims use 1 for arr.shape[axis]:
        # (can be empty [])
        #
        if keepdims:
            remaining_shape = tuple(
                1 if k in axes_set else original_shape[k]
                for k in range(0, len(original_shape))
            )
        else:
            remaining_shape = tuple(
                arr.shape[k] for k in range(0, ndims) if k != axis
            )

    # compose qarr.shape with arr.shape:
    #
    # result.shape = (q_arr.shape, arr.shape -{axis}):
    #
    qresult_shape = (*q_arr.shape, *remaining_shape)

    # construct result NdArray, non-flattening approach:
    #
    if qs_all is None:
        qs_all = zeros(qresult_shape, dtype=to_dtype)
    else:
        # implicit conversion from to_dtype to qs_all.dtype assumed
        #
        if qs_all.shape != qresult_shape:
            raise ValueError("wrong shape on output array")

    for index, q in np.ndenumerate(q_arr):
        gamma, left_pos = method(q, n)
        # Note that gamma may be None, in which case `right_pos` has no
        # meaning since use the exact index.
        right_pos = left_pos + 1

        # The virtual pos, which was used to calculate `left`, can be outside
        # the range, so fix all indices to be in range here.
        if left_pos >= n - 1:
            left_pos = right_pos = n - 1
        elif left_pos < 0:
            left_pos = right_pos = 0

        # If gamma is None, we only have to extract the correct values
        if gamma is None:
            qs_all[index] = arr.take(left_pos, axis).reshape(remaining_shape)
        else:
            # (N-1) dimensional ndarray of left, right
            # neighbor values:
            #
            # non-flattening approach:
            #
            # extract values at left and right position;
            arr_1D_lvals = arr.take(left_pos, axis).reshape(remaining_shape)
            arr_1D_rvals = arr.take(right_pos, axis).reshape(remaining_shape)

            # TODO: We may want to use a more precise interpolation formula
            # like NumPy here (or implement an `lerp` function to use).
            #
            # vectorized for axis != None;
            # (non-flattening approach)
            left = (1.0 - gamma) * arr_1D_lvals
            right = gamma * arr_1D_rvals
            qs_all[index] = left + right

    return qs_all


_ORDER_FUNCS = {
    "inverted_cdf": _inverted_cdf,
    "averaged_inverted_cdf": _averaged_inverted_cdf,
    "closest_observation": _closest_observation,
    "interpolated_inverted_cdf": _interpolated_inverted_cdf,
    "hazen": _hazen,
    "weibull": _weibull,
    "linear": _linear,
    "median_unbiased": _median_unbiased,
    "normal_unbiased": _normal_unbiased,
    "lower": _lower,
    "higher": _higher,
    "midpoint": _midpoint,
    "nearest": _nearest,
}


@add_boilerplate("a")
def quantile(
    a: ndarray,
    q: float | Iterable[float] | ndarray,
    axis: int | tuple[int, ...] | None = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is
        to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:
        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'
        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:
        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    Raises
    ------
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.quantile

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    """

    real_axis: int | None
    axes_set: Sequence[int] = ()
    original_shape = a.shape

    if axis is not None and isinstance(axis, Iterable):
        nrm_axis = normalize_axis_tuple(axis, a.ndim)
        if len(axis) == 1:
            real_axis = nrm_axis[0]
            a_rr = a
        else:
            # reshuffling requires non-negative axes:
            (real_axis, a_rr) = _reshuffle_reshape(a, nrm_axis)
            # What happens with multiple axes and overwrite_input = True ?
            # It seems overwrite_input is reset to False;
            overwrite_input = False
        axes_set = nrm_axis
    else:
        real_axis = axis
        a_rr = a
        if real_axis is not None:
            axes_set = normalize_axis_tuple(real_axis, a.ndim)
            real_axis = axes_set[0]

    # covers both array-like and scalar cases:
    #
    q_arr = np.asarray(q)

    # in the future k-sort (partition)
    # might be faster, for now it uses sort
    # arr = partition(arr, k = floor(nq), axis = real_axis)
    # but that would require a k-sort call for each `q`!
    # too expensive for many `q` values...
    # if no axis given then elements are sorted as a 1D array
    #
    if overwrite_input:
        a_rr.sort(axis=real_axis)
        arr = a_rr
    else:
        arr = sort(a_rr, axis=real_axis)

    if arr.dtype.kind == "c":
        raise TypeError("input array cannot be of complex type")

    # return type dependency on arr.dtype:
    #
    # it depends on interpolation method;
    # For discontinuous methods returning either end of the interval within
    # which the quantile falls, or the other; arr.dtype is returned;
    # else, logic below:
    #
    # if is_float(arr_dtype) && (arr.dtype >= dtype('float64')) then
    #    arr.dtype
    # else
    #    dtype('float64')
    #
    # see https://github.com/numpy/numpy/issues/22323
    #
    if method in [
        "inverted_cdf",
        "closest_observation",
        "lower",
        "higher",
        "nearest",
    ]:
        to_dtype = arr.dtype
    else:
        to_dtype = np.dtype("float64")

        # in case dtype("float128") becomes supported:
        #
        # to_dtype = (
        #     arr.dtype
        #     if (arr.dtype == np.dtype("float128"))
        #     else np.dtype("float64")
        # )

    res = _quantile_impl(
        arr,
        q_arr,
        real_axis,
        axes_set,
        original_shape,
        _ORDER_FUNCS[method],
        keepdims,
        to_dtype,
        out,
    )

    if out is not None:
        # out = res.astype(out.dtype) -- conversion done inside impl
        return out
    else:
        return res


@add_boilerplate("a")
def percentile(
    a: ndarray,
    q: float | Iterable[float] | ndarray,
    axis: int | tuple[int, ...] | None = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the q-th percentile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default is
        to compute the percentile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:
        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'
        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:
        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    Raises
    ------
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.percentile

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    """

    q_arr = np.asarray(q)
    q01 = q_arr / 100.0

    return quantile(
        a,
        q01,
        axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )


# args:
#
# arr:      [in] source nd-array on which quantiles are calculated;
#                NaNs ignored; precondition: assumed sorted!
# q_arr:    [in] quantile input values nd-array;
# axis:     [in] axis along which quantiles are calculated;
# method:   [in] func(q, n) returning (gamma, j),
#                where = array1D.size;
# keepdims: [in] boolean flag specifying whether collapsed axis
#                should be kept as dim=1;
# to_dtype: [in] dtype to convert the result to;
# qs_all:   [in/out] result pass through or created (returned)
#
def nanquantile_impl(
    arr: ndarray,
    q_arr: npt.NDArray[Any],
    non_nan_counts: ndarray,
    axis: int | None,
    axes_set: Sequence[int],
    original_shape: tuple[int, ...],
    method: Callable[[float, int], tuple[float | None, int]],
    keepdims: bool,
    to_dtype: np.dtype[Any],
    qs_all: ndarray | None,
) -> ndarray:
    ndims = len(arr.shape)

    if axis is None:
        if keepdims:
            remaining_shape = (1,) * len(original_shape)
        else:
            remaining_shape = ()  # only `q_arr` dictates shape;
        # quantile applied to `arr` seen as 1D;
    else:
        # arr.shape -{axis}; if keepdims use 1 for arr.shape[axis]:
        # (can be empty [])
        #
        if keepdims:
            remaining_shape = tuple(
                1 if k in axes_set else original_shape[k]
                for k in range(0, len(original_shape))
            )
        else:
            remaining_shape = tuple(
                arr.shape[k] for k in range(0, ndims) if k != axis
            )

    # compose qarr.shape with arr.shape:
    #
    # result.shape = (q_arr.shape, arr.shape -{axis}):
    #
    qresult_shape = (*q_arr.shape, *remaining_shape)

    # construct result Ndarray, non-flattening approach:
    #
    if qs_all is None:
        qs_all = zeros(qresult_shape, dtype=to_dtype)
    else:
        # implicit conversion from to_dtype to qs_all.dtype assumed
        #
        if qs_all.shape != qresult_shape:
            raise ValueError("wrong shape on output array")

    if not keepdims:
        assert non_nan_counts.shape == remaining_shape

    arr_gammas = zeros(remaining_shape, dtype=arr.dtype)
    arr_lvals = zeros(remaining_shape, dtype=arr.dtype)
    arr_rvals = zeros(remaining_shape, dtype=arr.dtype)

    # Similar to the non-nan implementation except that it needs to make
    # `n` depend on the number of non-nan-counts.
    for qindex, q in np.ndenumerate(q_arr):
        assert qs_all[qindex].shape == remaining_shape

        # TODO(aschaffer): Vectorize this operation, see
        # github.com/nv-legate/cupynumeric/pull/1121#discussion_r1484731763
        gamma = None
        for aindex, n in np.ndenumerate(non_nan_counts):
            # TODO (2024-08): `n` should be an integral type, but wasn't:
            n = int(n)
            if n == 0:
                # Cannot define a quantile over an empty range, return NaN
                # TODO(mpapadakis): mypy mysteriously complains that
                # expression has type "float", target has type "ndarray"
                arr_lvals[aindex] = np.nan  # type: ignore[assignment]
                arr_rvals[aindex] = np.nan  # type: ignore[assignment]
                continue

            gamma, left_pos = method(q, n)

            right_pos = left_pos + 1
            if left_pos >= n - 1:
                left_pos = right_pos = n - 1
            elif left_pos < 0:
                left_pos = right_pos = 0

            # assumption: since `non_nan_counts` has the same
            # shape as `remaining_shape` (checked above),
            # `aindex` are the same indices as those needed
            # to access `a`'s remaining shape slices;
            #
            full_l_index = (*aindex[:axis], left_pos, *aindex[axis:])
            arr_lvals[aindex] = arr[full_l_index]
            if gamma is not None:
                # TODO(mpapadakis): As above, mypy complains about assignment
                arr_gammas[aindex] = gamma  # type: ignore[assignment]

                full_r_index = (*aindex[:axis], right_pos, *aindex[axis:])
                arr_rvals[aindex] = arr[full_r_index]

        if gamma is None:
            # Note that gamma can only be always None or never
            qs_all[qindex] = arr_lvals
        else:
            left = (1 - arr_gammas) * arr_lvals
            right = arr_gammas * arr_rvals
            qs_all[qindex] = left + right

    return qs_all


@add_boilerplate("a")
def nanquantile(
    a: ndarray,
    q: float | Iterable[float] | ndarray,
    axis: int | tuple[int, ...] | None = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the q-th quantile of the data along the specified axis,
    while ignoring nan values.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array,
        containing nan values to be ignored.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is
        to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:
        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'
        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:
        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    Raises
    ------
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.nanquantile

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    """

    real_axis: int | None
    axes_set: Sequence[int] = ()
    original_shape = a.shape

    if axis is not None and isinstance(axis, Iterable):
        nrm_axis = normalize_axis_tuple(axis, a.ndim)
        if len(axis) == 1:
            real_axis = nrm_axis[0]
            a_rr = a
        else:
            (real_axis, a_rr) = _reshuffle_reshape(a, nrm_axis)
            # What happens with multiple axes and overwrite_input = True ?
            # It seems overwrite_input is reset to False;
            # But `overwrite_input` doesn't matter for the NaN version of this
            # function
            # overwrite_input = False
        axes_set = nrm_axis
    else:
        real_axis = axis
        a_rr = a
        if real_axis is not None:
            axes_set = normalize_axis_tuple(real_axis, a.ndim)
            real_axis = axes_set[0]

    # ndarray of non-NaNs:
    #
    non_nan_counts = asarray(
        count_nonzero(logical_not(isnan(a_rr)), axis=real_axis)
    )

    # covers both array-like and scalar cases:
    #
    q_arr = np.asarray(q)

    # in the future k-sort (partition)
    # might be faster, for now it uses sort
    # arr = partition(arr, k = floor(nq), axis = real_axis)
    # but that would require a k-sort call for each `q`!
    # too expensive for many `q` values...
    # if no axis given then elements are sorted as a 1D array
    #
    # replace NaN's by dtype.max:
    #
    arr = where(isnan(a_rr), np.finfo(a_rr.dtype).max, a_rr)
    arr.sort(axis=real_axis)

    if arr.dtype.kind == "c":
        raise TypeError("input array cannot be of complex type")

    # return type dependency on arr.dtype:
    #
    # it depends on interpolation method;
    # For discontinuous methods returning either end of the interval within
    # which the quantile falls, or the other; arr.dtype is returned;
    # else, logic below:
    #
    # if is_float(arr_dtype) && (arr.dtype >= dtype('float64')) then
    #    arr.dtype
    # else
    #    dtype('float64')
    #
    # see https://github.com/numpy/numpy/issues/22323
    #
    if method in [
        "inverted_cdf",
        "closest_observation",
        "lower",
        "higher",
        "nearest",
    ]:
        to_dtype = arr.dtype
    else:
        to_dtype = np.dtype("float64")

        # in case dtype("float128") becomes supported:
        #
        # to_dtype = (
        #     arr.dtype
        #     if (arr.dtype == np.dtype("float128"))
        #     else np.dtype("float64")
        # )

    res = nanquantile_impl(
        arr,
        q_arr,
        non_nan_counts,
        real_axis,
        axes_set,
        original_shape,
        _ORDER_FUNCS[method],
        keepdims,
        to_dtype,
        out,
    )

    if out is not None:
        # out = res.astype(out.dtype) -- conversion done inside impl
        return out
    else:
        return res


@add_boilerplate("a")
def nanpercentile(
    a: ndarray,
    q: float | Iterable[float] | ndarray,
    axis: int | tuple[int, ...] | None = None,
    out: ndarray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the q-th percentile of the data along the specified axis,
    while ignoring nan values.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array,
        containing nan values to be ignored.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default is
        to compute the percentile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:
        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'
        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:
        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    Raises
    ------
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.nanpercentile

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    """

    q_arr = np.asarray(q)
    q01 = q_arr / 100.0

    return nanquantile(
        a,
        q01,
        axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )
