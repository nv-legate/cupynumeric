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
from .._array.util import add_boilerplate
from ..types import SortSide
from .creation_data import asarray
from .creation_shape import ones, zeros
from .math_extrema import amax, amin
from .ssc_searching import searchsorted

if TYPE_CHECKING:
    import numpy.typing as npt

_builtin_max = max
_builtin_range = range


@add_boilerplate("x", "weights")
def bincount(
    x: ndarray, weights: ndarray | None = None, minlength: int = 0
) -> ndarray:
    """
    bincount(x, weights=None, minlength=0)

    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like
        1-D input array of non-negative ints.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : ndarray[int]
        The result of binning the input array.
        The length of `out` is equal to ``cupynumeric.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    numpy.bincount

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if x.ndim != 1:
        raise ValueError("the input array must be 1-dimensional")
    if weights is not None:
        if weights.shape != x.shape:
            raise ValueError("weights array must be same shape for bincount")
        if weights.dtype.kind == "c":
            raise ValueError("weights must be convertible to float64")
        # Make sure the weights are float64
        weights = weights.astype(np.float64)
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError("input array for bincount must be integer type")
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")
    # Note that the following are non-blocking operations,
    # though passing their results to `int` is blocking
    max_val, min_val = amax(x), amin(x)
    if int(min_val) < 0:
        raise ValueError("the input array must have no negative elements")
    minlength = _builtin_max(minlength, int(max_val) + 1)
    if x.size == 1:
        # Handle the special case of 0-D array
        if weights is None:
            out = zeros((minlength,), dtype=np.dtype(np.int64))
            # TODO: Remove this "type: ignore" once @add_boilerplate can
            # propagate "ndarray -> ndarray | npt.ArrayLike" in wrapped sigs
            out[x[0]] = 1  # type: ignore [assignment]
        else:
            out = zeros((minlength,), dtype=weights.dtype)
            index = x[0]
            out[index] = weights[0]
    else:
        # Normal case of bincount
        if weights is None:
            out = ndarray(
                (minlength,),
                dtype=np.dtype(np.int64),
                inputs=(x, weights),
            )
            out._thunk.bincount(x._thunk)
        else:
            out = ndarray(
                (minlength,),
                dtype=weights.dtype,
                inputs=(x, weights),
            )
            out._thunk.bincount(x._thunk, weights=weights._thunk)
    return out


@add_boilerplate("x", "weights")
def histogram(
    x: ndarray,
    bins: ndarray | npt.ArrayLike | int = 10,
    range: tuple[int, int] | tuple[float, float] | None = None,
    weights: ndarray | None = None,
    density: bool = False,
) -> tuple[ndarray, ndarray]:
    """
    Compute the histogram of a dataset.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If `bins` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max())``. Values outside the range are ignored. The first
        element of the range must be smaller than the second. This argument is
        ignored when bin edges are provided explicitly.
    weights : array_like, optional
        An array of weights, of the same shape as `a`. Each value in `a` only
        contributes its associated weight towards the bin count (instead of 1).
        If `density` is True, the weights are normalized, so that the integral
        of the density over the range remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in each
        bin. If ``True``, the result is the value of the probability *density*
        function at the bin, normalized such that the *integral* over the range
        is 1. Note that the sum of the histogram values will not be equal to 1
        unless bins of unity width are chosen; it is not a probability *mass*
        function.

    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : array
        Return the bin edges ``(length(hist)+1)``.

    See Also
    --------
    numpy.histogram

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result_type: np.dtype[Any] = np.dtype(np.int64)

    if np.ndim(bins) > 1:
        raise ValueError("`bins` must be 1d, when an array")

    # check isscalar(bins):
    #
    if np.ndim(bins) == 0:
        if not isinstance(bins, int):
            raise TypeError("`bins` must be array or integer type")

        num_intervals = bins

        if range is not None:
            assert isinstance(range, tuple) and len(range) == 2
            if range[0] >= range[1]:
                raise ValueError(
                    "`range` must be a pair of increasing values."
                )

            lower_b = range[0]
            higher_b = range[1]
        elif x.size == 0:
            lower_b = 0.0
            higher_b = 1.0
        else:
            lower_b = float(min(x))
            higher_b = float(max(x))

        step = (higher_b - lower_b) / num_intervals

        bins_array = asarray(
            [lower_b + k * step for k in _builtin_range(0, num_intervals)]
            + [higher_b],
            dtype=np.dtype(np.float64),
        )

        bins_orig_type = bins_array.dtype
    else:
        bins_as_arr = asarray(bins)
        bins_orig_type = bins_as_arr.dtype

        bins_array = bins_as_arr.astype(np.dtype(np.float64))
        num_intervals = bins_array.shape[0] - 1

        if not all((bins_array[1:] - bins_array[:-1]) >= 0):
            raise ValueError(
                "`bins` must increase monotonically, when an array"
            )

    if x.ndim != 1:
        x = x.flatten()

    if weights is not None:
        if weights.shape != x.shape:
            raise ValueError(
                "`weights` array must be same shape for histogram"
            )

        result_type = weights.dtype
        weights_array = weights.astype(np.dtype(np.float64))
    else:
        # case weights == None cannot be handled inside _thunk.histogram,
        # bc/ of hist ndarray inputs(), below;
        # needs to be handled here:
        #
        weights_array = ones(x.shape, dtype=np.dtype(np.float64))

    if x.size == 0:
        return (
            zeros((num_intervals,), dtype=result_type),
            bins_array.astype(bins_orig_type),
        )

    hist = ndarray(
        (num_intervals,),
        dtype=weights_array.dtype,
        inputs=(x, bins_array, weights_array),
    )
    hist._thunk.histogram(
        x._thunk, bins_array._thunk, weights=weights_array._thunk
    )

    # handle (density = True):
    #
    if density:
        result_type = np.dtype(np.float64)
        hist /= sum(hist)
        hist /= bins_array[1:] - bins_array[:-1]

    return hist.astype(result_type), bins_array.astype(bins_orig_type)


@add_boilerplate("x", "bins")
def digitize(
    x: ndarray,
    bins: ndarray,
    right: bool = False,
) -> ndarray | int:
    """
    Return the indices of the bins to which each value in input array belongs.

    =========  =============  ============================
    `right`    order of bins  returned index `i` satisfies
    =========  =============  ============================
    ``False``  increasing     ``bins[i-1] <= x < bins[i]``
    ``True``   increasing     ``bins[i-1] < x <= bins[i]``
    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
    =========  =============  ============================

    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
    returned as appropriate.

    Parameters
    ----------
    x : array_like
        Input array to be binned. Doesn't need to be 1-dimensional.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    indices : ndarray of ints
        Output array of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.digitize

    Notes
    -----
    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    will result in an IndexError.
    For monotonically *increasing* `bins`, the following are equivalent::

        np.digitize(x, bins, right=True)
        np.searchsorted(bins, x, side='left')

    Note that as the order of the arguments are reversed, the side must be too.
    The `searchsorted` call is marginally faster, as it does not do any
    monotonicity checks. Perhaps more importantly, it supports all dtypes.

    Examples
    --------
    >>> x = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = np.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    >>> for n in range(x.size):
    ...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
    ...
    0.0 <= 0.2 < 1.0
    4.0 <= 6.4 < 10.0
    2.5 <= 3.0 < 4.0
    1.0 <= 1.6 < 2.5
    >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
    >>> bins = np.array([0, 5, 10, 15, 20])
    >>> np.digitize(x,bins,right=True)
    array([1, 2, 3, 4, 4])
    >>> np.digitize(x,bins,right=False)
    array([1, 3, 3, 4, 5])

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # here for compatibility, searchsorted below is happy to take this
    if np.issubdtype(x.dtype, np.complexfloating):
        raise TypeError("x may not be complex")

    if bins.ndim > 1:
        raise ValueError("bins must be one-dimensional")

    increasing = (bins[1:] >= bins[:-1]).all()
    decreasing = (bins[1:] <= bins[:-1]).all()
    if not increasing and not decreasing:
        raise ValueError("bins must be monotonically increasing or decreasing")

    # this is backwards because the arguments below are swapped
    side: SortSide = "left" if right else "right"
    if decreasing:
        # reverse the bins, and invert the results
        return len(bins) - searchsorted(bins.flip(), x, side=side)
    else:
        return searchsorted(bins, x, side=side)
