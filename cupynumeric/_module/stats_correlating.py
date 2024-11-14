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
from .._array.util import add_boilerplate
from .array_joining import concatenate
from .creation_data import array
from .creation_shape import empty
from .linalg_mvp import dot
from .math_misc import clip
from .math_sum_prod_diff import sum
from .stats_avgs_vars import average


@add_boilerplate("m", "y", "fweights", "aweights")
def cov(
    m: ndarray,
    y: ndarray | None = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: ndarray | None = None,
    aweights: ndarray | None = None,
    *,
    dtype: np.dtype[Any] | None = None,
) -> ndarray:
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof``.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. The default value is ``None``.
    fweights : array_like, int, optional
        1-D array of integer frequency weights; the number of times each
        observation vector should be repeated.
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `float64` precision.

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    numpy.cov

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Check inputs
    if ddof is not None and not isinstance(ddof, int):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None and y.ndim > 2:
        raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m.dtype, np.float64)
        else:
            dtype = np.result_type(m.dtype, y.dtype, np.float64)

    X = array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return empty((0, 0))
    if y is not None:
        y = array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        # TODO(mpapadakis): Could have saved on an intermediate copy of X in
        # this case, if it was already of the right shape.
        X = concatenate((X, y), axis=0)

    if ddof is None:
        if not bias:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w: ndarray | None = None
    if fweights is not None:
        if fweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")
        w = fweights
    if aweights is not None:
        if aweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            # Cannot be done in-place with *= when aweights.dtype != w.dtype
            w = w * aweights

    avg, w_sum = average(X, axis=1, weights=w, returned=True)

    # Determine the normalization
    fact: ndarray | float = 0.0
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum

    # TODO(mpapadakis): @add_boilerplate should extend the types of array
    # arguments from `ndarray` to `npt.ArrayLike | ndarray`.
    fact = clip(fact, 0.0, None)  # type: ignore[arg-type]

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = dot(X, X_T.conj())
    # Cannot be done in-place with /= when the dtypes differ
    c = c / fact

    return c.squeeze()
