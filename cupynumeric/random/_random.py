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
from .._utils.coverage import clone_class
from ..runtime import runtime
from ._generator import default_rng, get_static_generator  # NOQA

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike


def seed(init: int | None = None) -> None:
    """
    Reseed the legacy random number generator.

    This function is effective only when cuRAND is NOT used in the build
    and is a no-op otherwise.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if init is None:
        runtime.set_next_random_epoch(0)
        return

    if not isinstance(init, int):
        raise TypeError("seed must be an integer or None")
    try:
        init = int(init)
    except TypeError as te:
        raise TypeError("seed must be an integer or None") from te
    if not (0 <= init <= 2**32 - 1):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    runtime.set_next_random_epoch(init)


def beta(
    a: float,
    b: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    beta(a, b, size=None)

    Draw samples from a Beta distribution.

    The Beta distribution is a special case of the Dirichlet distribution,
    and is related to the Gamma distribution.  It has the probability
    distribution function

    .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                     (1 - x)^{\\beta - 1},

    where the normalization, B, is the beta function,

    .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                 (1 - t)^{\\beta - 1} dt.

    It is often seen in Bayesian inference and order statistics.

    Parameters
    ----------
    a : float
        Alpha, positive (>0).
    b : float
        Beta, positive (>0).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized beta distribution.

    See Also
    --------
    numpy.random.beta

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().beta(a, b, size, dtype)


def binomial(
    ntrials: int,
    p: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    """
    binomial(n, p, size=None)

    Draw samples from a binomial distribution.

    Samples are drawn from a binomial distribution with specified
    parameters, n trials and p probability of success where
    n an integer >= 0 and p is in the interval ``[0,1]``. (n may be
    input as a float, but it is truncated to an integer in use)

    Parameters
    ----------
    n : int
        Parameter of the distribution, >= 0. Floats are also accepted,
        but they will be truncated to integers.
    p : float
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized binomial distribution, where
        each sample is equal to the number of successes over the n trials.

    See Also
    --------
    numpy.random.binomial

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().binomial(ntrials, p, size, dtype)


def bytes(length: int) -> ndarray:
    """
    bytes(length)

    Return random bytes.

    Parameters
    ----------
    length : int
        Number of random bytes.

    Returns
    -------
    out : bytes
        String of length `length`.

    See Also
    --------
    numpy.random.bytes

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().bytes(length)


def chisquare(
    df: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    chisquare(df, size=None)

    Draw samples from a chi-square distribution.

    When `df` independent random variables, each with standard normal
    distributions (mean 0, variance 1), are squared and summed, the
    resulting distribution is chi-square.  This distribution is often
    used in hypothesis testing.

    Parameters
    ----------
    df : float
         Number of degrees of freedom, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized chi-square distribution.

    Raises
    ------
    ValueError
        When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
        is given.

    See Also
    --------
    numpy.random.chisquare

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().chisquare(df, size, dtype)


def exponential(
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    exponential(scale=1.0, size=None)

    Draw samples from an exponential distribution.

    Its probability density function is

    .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta}
                                        \\exp(-\\frac{x}{\\beta}),

    for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
    which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
    The rate parameter is an alternative, widely used parameterization
    of the exponential distribution [3]_.

    The exponential distribution is a continuous analogue of the
    geometric distribution.  It describes many common situations, such as
    the size of raindrops measured over many rainstorms [1]_, or the time
    between page requests to Wikipedia [2]_.

    Parameters
    ----------
    scale : float, optional
        The scale parameter, :math:`\\beta = 1/\\lambda`. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized exponential distribution.

    See Also
    --------
    numpy.random.exponential

    References
    ----------
    .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
           Random Signal Principles", 4th ed, 2001, p. 57.
    .. [2] Wikipedia, "Poisson process",
           https://en.wikipedia.org/wiki/Poisson_process
    .. [3] Wikipedia, "Exponential distribution",
           https://en.wikipedia.org/wiki/Exponential_distribution

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().exponential(scale, size, dtype)


def f(
    dfnum: float,
    dfden: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    f(dfnum, dfden, size=None)

    Draw samples from an F distribution.

    Samples are drawn from an F distribution with specified parameters,
    `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
    freedom in denominator), where both parameters must be greater than
    zero.

    The random variate of the F distribution (also known as the
    Fisher distribution) is a continuous probability distribution
    that arises in ANOVA tests, and is the ratio of two chi-square
    variates.

    Parameters
    ----------
    dfnum : float
        Degrees of freedom in numerator, must be > 0.
    dfden : float
        Degrees of freedom in denominator, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Fisher distribution.

    See Also
    --------
    numpy.random.f

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().f(dfnum, dfden, size, dtype)


def gamma(
    shape: float,
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    gamma(shape, scale=1.0, size=None)

    Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    `shape` (sometimes designated "k") and `scale` (sometimes designated
    "theta"), where both parameters are > 0.

    Parameters
    ----------
    shape : float
        The shape of the gamma distribution. Must be non-negative.
    scale : float, optional
        The scale of the gamma distribution. Must be non-negative.
        Default is equal to 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized gamma distribution.

    See Also
    --------
    numpy.random.gamma

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().gamma(shape, scale, size, dtype)


def geometric(
    p: float, size: NdShapeLike | None = None, dtype: npt.DTypeLike = np.uint32
) -> ndarray:
    """
    geometric(p, size=None)

    Draw samples from the geometric distribution.

    Bernoulli trials are experiments with one of two outcomes:
    success or failure (an example of such an experiment is flipping
    a coin).  The geometric distribution models the number of trials
    that must be run in order to achieve success.  It is therefore
    supported on the positive integers, ``k = 1, 2, ...``.

    The probability mass function of the geometric distribution is

    .. math:: f(k) = (1 - p)^{k - 1} p

    where `p` is the probability of success of an individual trial.

    Parameters
    ----------
    p : float
        The probability of success of an individual trial.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized geometric distribution.

    See Also
    --------
    numpy.random.geometric

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().geometric(p, size, dtype)


def gumbel(
    loc: float = 0.0,
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    gumbel(loc=0.0, scale=1.0, size=None)

    Draw samples from a Gumbel distribution.

    Draw samples from a Gumbel distribution with specified location and
    scale.

    Parameters
    ----------
    loc : float, optional
        The location of the mode of the distribution. Default is 0.
    scale : float, optional
        The scale parameter of the distribution. Default is 1. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Gumbel distribution.

    See Also
    --------
    numpy.random.gumbel

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().gumbel(loc, scale, size, dtype)


def hypergeometric(
    ngood: int,
    nbad: int,
    nsample: int,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    """
    hypergeometric(ngood, nbad, nsample, size=None)

    Draw samples from a Hypergeometric distribution.

    Samples are drawn from a hypergeometric distribution with specified
    parameters, `ngood` (ways to make a good selection), `nbad` (ways to make
    a bad selection), and `nsample` (number of items sampled, which is less
    than or equal to the sum ``ngood + nbad``).

    Parameters
    ----------
    ngood : int
        Number of ways to make a good selection.  Must be nonnegative.
    nbad : int
        Number of ways to make a bad selection.  Must be nonnegative.
    nsample : int
        Number of items sampled.  Must be at least 1 and at most
        ``ngood + nbad``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized hypergeometric distribution. Each
        sample is the number of good items within a randomly selected subset of
        size `nsample` taken from a set of `ngood` good items and `nbad` bad
        items.

    See Also
    --------
    numpy.random.hypergeometric

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().hypergeometric(
        ngood, nbad, nsample, size, dtype
    )


def laplace(
    loc: float = 0.0,
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    laplace(loc=0.0, scale=1.0, size=None)

    Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).

    The Laplace distribution is similar to the Gaussian/normal distribution,
    but is sharper at the peak and has fatter tails. It represents the
    difference between two independent, identically distributed exponential
    random variables.

    Parameters
    ----------
    loc : float, optional
        The position, :math:`\\mu`, of the distribution peak. Default is 0.
    scale : float, optional
        :math:`\\lambda`, the exponential decay. Default is 1. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Laplace distribution.

    See Also
    --------
    numpy.random.laplace

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().laplace(loc, scale, size, dtype)


def logistic(
    loc: float = 0.0,
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    logistic(loc=0.0, scale=1.0, size=None)

    Draw samples from a logistic distribution.

    Samples are drawn from a logistic distribution with specified
    parameters, loc (location or mean, also median), and scale (>0).

    Parameters
    ----------
    loc : float, optional
        Parameter of the distribution. Default is 0.
    scale : float, optional
        Parameter of the distribution. Must be non-negative.
        Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized logistic distribution.

    See Also
    --------
    numpy.random.logistic

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().logistic(loc, scale, size, dtype)


def lognormal(
    mean: float = 0.0,
    sigma: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    lognormal(mean=0.0, sigma=1.0, size=None)

    Draw samples from a log-normal distribution.

    Draw samples from a log-normal distribution with specified mean,
    standard deviation, and array shape.  Note that the mean and standard
    deviation are not the values for the distribution itself, but of the
    underlying normal distribution it is derived from.

    Parameters
    ----------
    mean : float, optional
        Mean value of the underlying normal distribution. Default is 0.
    sigma : float, optional
        Standard deviation of the underlying normal distribution. Must be
        non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized log-normal distribution.

    See Also
    --------
    numpy.random.lognormal

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().lognormal(mean, sigma, size, dtype)


def logseries(
    p: float, size: NdShapeLike | None = None, dtype: npt.DTypeLike = np.uint32
) -> ndarray:
    """
    logseries(p, size=None)

    Draw samples from a logarithmic series distribution.

    Samples are drawn from a log series distribution with specified
    shape parameter, 0 < ``p`` < 1.

    Parameters
    ----------
    p : float
        Shape parameter for the distribution.  Must be in the range (0, 1).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized logarithmic series distribution.

    See Also
    --------
    numpy.random.logseries

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().logseries(p, size, dtype)


def negative_binomial(
    n: int,
    p: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    """
    negative_binomial(n, p, size=None)

    Draw samples from a negative binomial distribution.

    Samples are drawn from a negative binomial distribution with specified
    parameters, `n` successes and `p` probability of success where `n`
    is > 0 and `p` is in the interval (0, 1].

    Parameters
    ----------
    n : int
        Parameter of the distribution, > 0.
    p : float
        Parameter of the distribution, > 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized negative binomial distribution,
        where each sample is equal to N, the number of failures that
        occurred before a total of n successes was reached.

    See Also
    --------
    numpy.random.negative_binomial

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().negative_binomial(n, p, size, dtype)


def noncentral_chisquare(
    df: float,
    nonc: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    noncentral_chisquare(df, nonc, size=None)

    Draw samples from a noncentral chi-square distribution.

    The noncentral :math:`\\chi^2` distribution is a generalization of
    the :math:`\\chi^2` distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom, must be > 0.
    nonc : float
        Non-centrality, must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized noncentral chi-square
        distribution.

    See Also
    --------
    numpy.random.noncentral_chisquare

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().noncentral_chisquare(df, nonc, size, dtype)


def noncentral_f(
    dfnum: float,
    dfden: float,
    nonc: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    noncentral_f(dfnum, dfden, nonc, size=None)

    Draw samples from the noncentral F distribution.

    Samples are drawn from an F distribution with specified parameters,
    `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
    freedom in denominator), where both parameters > 1.
    `nonc` is the non-centrality parameter.

    Parameters
    ----------
    dfnum : float
        Numerator degrees of freedom, must be > 0.
    dfden : float
        Denominator degrees of freedom, must be > 0.
    nonc : float
        Non-centrality parameter, the sum of the squares of the numerator
        means, must be >= 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized noncentral Fisher distribution.

    See Also
    --------
    numpy.random.noncentral_f

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().noncentral_f(dfnum, dfden, nonc, size, dtype)


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    normal(loc=0.0, scale=1.0, size=None)

    Draw random samples from a normal (Gaussian) distribution.

    The probability density function of the normal distribution, first
    derived by De Moivre and 200 years later by both Gauss and Laplace
    independently [1]_, is often called the bell curve because of
    its characteristic shape.

    The normal distribution occurs often in nature.  For example, it
    describes the commonly occurring distribution of samples influenced
    by a large number of tiny, random disturbances, each with its own
    unique distribution [1]_.

    Parameters
    ----------
    loc : float
        Mean ("centre") of the distribution.
    scale : float
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized normal distribution.

    See Also
    --------
    numpy.random.normal

    References
    ----------
    .. [1] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001,
           pp. 51, 51, 125.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().normal(loc, scale, size, dtype)


def pareto(
    a: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    pareto(a, size=None)

    Draw samples from a Pareto II or Lomax distribution with
    specified shape.

    The Lomax or Pareto II distribution is a shifted Pareto
    distribution. The classical Pareto distribution can be
    obtained from the Lomax distribution by adding 1 and
    multiplying by the scale parameter ``m``.  The
    smallest value of the Lomax distribution is zero while for the
    classical Pareto distribution it is ``mu``, where the standard
    Pareto distribution has location ``mu = 1``.  Lomax can also
    be considered as a simplified version of the Generalized
    Pareto distribution (available in SciPy), with the scale set
    to one and the location set to zero.

    The Pareto distribution must be greater than zero, and is
    unbounded above.  It is also known as the "80-20 rule".  In
    this distribution, 80 percent of the weights are in the lowest
    20 percent of the range, while the other 20 percent fill the
    remaining 80 percent of the range.

    Parameters
    ----------
    a : float
        Shape of the distribution. Must be positive.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Pareto distribution.

    See Also
    --------
    numpy.random.pareto

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().pareto(a, size, dtype)


def poisson(lam: float = 1.0, size: NdShapeLike | None = None) -> ndarray:
    """
    poisson(lam=1.0, size=None)

    Draw samples from a Poisson distribution.

    The Poisson distribution is the limit of the binomial distribution
    for large N.

    Parameters
    ----------
    lam : float
        Expected number of events occurring in a fixed-time interval,
        must be >= 0. A sequence must be broadcastable over the requested
        size.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Poisson distribution.

    See Also
    --------
    numpy.random.poisson

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().poisson(lam, size)


def power(
    a: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    power(a, size=None)

    Draws samples in [0, 1] from a power distribution with positive
    exponent a - 1.

    Also known as the power function distribution.

    Parameters
    ----------
    a : float
        Parameter of the distribution. Must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized power distribution.

    Raises
    ------
    ValueError
        If a <= 0.

    See Also
    --------
    numpy.random.power

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().power(a, size, dtype)


def rand(*shapeargs: int) -> float | ndarray:
    """
    rand(d0, d1, ..., dn)

    Random values in a given shape.

    Create an array of the given shape and populate it with
    random samples from a uniform distribution
    over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single float is returned.

    Returns
    -------
    out : ndarray, shape ``(d0, d1, ..., dn)``
        Random values.

    See Also
    --------
    numpy.random.rand

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return uniform(0.0, 1.0, size=shapeargs, dtype=np.float64)


def randint(
    low: int,
    high: int | None = None,
    size: NdShapeLike | None = None,
    dtype: np.dtype[Any] | type = int,
) -> int | ndarray | npt.NDArray[Any]:
    """
    randint(low, high=None, size=None, dtype=int)

    Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int
        Lowest (signed) integers to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result.  The default value is int.

    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    numpy.random.randint

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if not isinstance(low, int):
        raise NotImplementedError("'low' must be an integer")
    if high is not None and not isinstance(high, int):
        raise NotImplementedError("'high' must be an integer or None")

    if high is None:
        low, high = 0, low
        if high <= 0:
            raise ValueError("high <= 0")
    elif low >= high:
        raise ValueError("low >= high")

    return get_static_generator().integers(low, high, size, dtype)


def randn(*shapeargs: int) -> float | ndarray:
    """
    randn(d0, d1, ..., dn)

    Return a sample (or samples) from the "standard normal" distribution.

    If positive int_like arguments are provided, `randn` generates an array
    of shape ``(d0, d1, ..., dn)``, filled
    with random floats sampled from a univariate "normal" (Gaussian)
    distribution of mean 0 and variance 1. A single float randomly sampled
    from the distribution is returned if no argument is provided.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single float is returned.

    Returns
    -------
    Z : ndarray or float
        A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
        the standard normal distribution, or a single such float if
        no parameters were supplied.

    See Also
    --------
    numpy.random.randn

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return normal(0.0, 1.0, shapeargs)


def random(size: NdShapeLike | None = None) -> float | ndarray:
    """
    random(size=None)

    Return random floats in the half-open interval [0.0, 1.0). Alias for
    `random_sample` to ease forward-porting to the new random API.

    See Also
    --------
    numpy.random.random

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().random(size)


# deprecated in numpy from version 1.11.0
def random_integers(
    low: int,
    high: int | None = None,
    size: NdShapeLike | None = None,
    dtype: np.dtype[Any] | type = int,
) -> int | ndarray | npt.NDArray[Any]:
    """
    random_integers(low, high=None, size=None)

    Random integers of type `np.int_` between `low` and `high`, inclusive.

    Return random integers of type `np.int_` from the "discrete uniform"
    distribution in the closed interval [`low`, `high`].  If `high` is
    None (the default), then results are from [1, `low`]. The `np.int_`
    type translates to the C long integer type and its precision
    is platform dependent.

    This function has been deprecated. Use randint instead.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int, optional
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    numpy.random.random_integers

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if high is None:
        high = low
        low = 0
    return randint(low, high + 1, size, dtype)


def random_sample(
    size: NdShapeLike | None = None, dtype: npt.DTypeLike = np.float64
) -> float | ndarray:
    """
    random_sample(size=None)

    Return random floats in the half-open interval ``[0.0, 1.0)``.

    Results are from the "continuous uniform" distribution over the
    stated interval.  To sample :math:`Unif[a, b), b > a` multiply
    the output of `random_sample` by `(b-a)` and add `a`::

      (b - a) * random_sample() + a

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : float or ndarray of floats
        Array of random floats of shape `size` (unless ``size=None``, in which
        case a single float is returned).

    See Also
    --------
    numpy.random.random_sample

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return uniform(0.0, 1.0, size, dtype)


ranf = random_sample


def rayleigh(
    scale: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    rayleigh(scale=1.0, size=None)

    Draw samples from a Rayleigh distribution.

    The :math:`\\chi` and Weibull distributions are generalizations of the
    Rayleigh.

    Parameters
    ----------
    scale : float, optional
        Scale, also equals the mode. Must be non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Rayleigh distribution.

    See Also
    --------
    numpy.random.rayleigh

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().rayleigh(scale, size, dtype)


sample = random_sample


def standard_cauchy(
    size: NdShapeLike | None = None, dtype: npt.DTypeLike = np.float64
) -> ndarray:
    """
    standard_cauchy(size=None)

    Draw samples from a standard Cauchy distribution with mode = 0.

    Also known as the Lorentz distribution.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    samples : ndarray or scalar
        The drawn samples.

    See Also
    --------
    numpy.random.standard_cauchy

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().standard_cauchy(size, dtype)


def standard_exponential(
    size: NdShapeLike | None = None, dtype: npt.DTypeLike = np.float64
) -> ndarray:
    """
    standard_exponential(size=None)

    Draw samples from the standard exponential distribution.

    `standard_exponential` is identical to the exponential distribution
    with a scale parameter of 1.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : float or ndarray
        Drawn samples.

    See Also
    --------
    numpy.random.standard_exponential

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().standard_exponential(size, dtype)


def standard_gamma(
    shape: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    standard_gamma(shape, size=None)

    Draw samples from a standard Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    shape (sometimes designated "k") and scale=1.

    Parameters
    ----------
    shape : float
        Parameter, must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized standard gamma distribution.

    See Also
    --------
    numpy.random.standard_gamma

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().standard_gamma(shape, size, dtype)


def standard_t(
    df: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    standard_t(df, size=None)

    Draw samples from a standard Student's t distribution with `df` degrees
    of freedom.

    A special case of the hyperbolic distribution.  As `df` gets
    large, the result resembles that of the standard normal
    distribution (`standard_normal`).

    Parameters
    ----------
    df : float
        Degrees of freedom, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized standard Student's t distribution.

    See Also
    --------
    numpy.random.standard_t

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().standard_t(df, size, dtype)


def triangular(
    left: float,
    mode: float,
    right: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    triangular(left, mode, right, size=None)

    Draw samples from the triangular distribution over the
    interval ``[left, right]``.

    The triangular distribution is a continuous probability
    distribution with lower limit left, peak at mode, and upper
    limit right. Unlike the other distributions, these parameters
    directly define the shape of the pdf.

    Parameters
    ----------
    left : float
        Lower limit.
    mode : float
        The value where the peak of the distribution occurs.
        The value must fulfill the condition ``left <= mode <= right``.
    right : float
        Upper limit, must be larger than `left`.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized triangular distribution.

    See Also
    --------
    numpy.random.triangular

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().triangular(left, mode, right, size, dtype)


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    uniform(low=0.0, high=1.0, size=None)

    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float, optional
        Upper boundary of the output interval.  All values generated will be
        less than or equal to high.  The high limit may be included in the
        returned array of floats due to floating-point rounding in the
        equation ``low + (high-low) * random_sample()``.  The default value
        is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    numpy.random.uniform

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().uniform(low, high, size, dtype)


def vonmises(
    mu: float,
    kappa: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    vonmises(mu, kappa, size=None)

    Draw samples from a von Mises distribution.

    Samples are drawn from a von Mises distribution with specified mode
    (mu) and dispersion (kappa), on the interval [-pi, pi].

    The von Mises distribution (also known as the circular normal
    distribution) is a continuous probability distribution on the unit
    circle.  It may be thought of as the circular analogue of the normal
    distribution.

    Parameters
    ----------
    mu : float
        Mode ("center") of the distribution.
    kappa : float
        Dispersion of the distribution, has to be >=0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized von Mises distribution.

    See Also
    --------
    numpy.random.vonmises

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().vonmises(mu, kappa, size, dtype)


def wald(
    mean: float,
    scale: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    wald(mean, scale, size=None)

    Draw samples from a Wald, or inverse Gaussian, distribution.

    As the scale approaches infinity, the distribution becomes more like a
    Gaussian. Some references claim that the Wald is an inverse Gaussian
    with mean equal to 1, but this is by no means universal.

    The inverse Gaussian distribution was first studied in relationship to
    Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
    because there is an inverse relationship between the time to cover a
    unit distance and distance covered in unit time.

    Parameters
    ----------
    mean : float
        Distribution mean, must be > 0.
    scale : float
        Scale parameter, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Wald distribution.

    See Also
    --------
    numpy.random.wald

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().wald(mean, scale, size, dtype)


def weibull(
    a: float,
    size: NdShapeLike | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    """
    weibull(a, size=None)

    Draw samples from a Weibull distribution.

    Draw samples from a 1-parameter Weibull distribution with the given
    shape parameter `a`.

    .. math:: X = (-ln(U))^{1/a}

    Here, U is drawn from the uniform distribution over (0,1].

    The more common 2-parameter Weibull, including a scale parameter
    :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.

    Parameters
    ----------
    a : float
        Shape parameter of the distribution.  Must be nonnegative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Weibull distribution.

    See Also
    --------
    numpy.random.weibull

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().weibull(a, size, dtype)


def zipf(
    a: float, size: NdShapeLike | None = None, dtype: npt.DTypeLike = np.uint32
) -> ndarray:
    """
    zipf(a, size=None)

    Draw samples from a Zipf distribution.

    Samples are drawn from a Zipf distribution with specified parameter
    `a` > 1.

    The Zipf distribution (also known as the zeta distribution) is a
    discrete probability distribution that satisfies Zipf's law: the
    frequency of an item is inversely proportional to its rank in a
    frequency table.

    Parameters
    ----------
    a : float
        Distribution parameter. Must be greater than 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Zipf distribution.

    See Also
    --------
    numpy.random.zipf

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return get_static_generator().zipf(a, size, dtype)


def _random_state_fallback(obj: Any) -> Any:
    # meant for the `self` argument; forward any unimplemented methods to the
    # wrapped vanilla NumPy RandomState
    if isinstance(obj, RandomState):
        return obj._np_random_state
    # eagerly convert any cuPyNumeric ndarrays to NumPy
    if isinstance(obj, ndarray):
        return obj.__array__()
    return obj


@clone_class(np.random.RandomState, fallback=_random_state_fallback)
class RandomState:
    """
    Container for a pseudo-random number generator.

    Exposes a number of methods for generating random numbers drawn from a
    variety of probability distributions.

    Parameters
    ----------
    seed : int, optional
        Random seed used to initialize the pseudo-random number generator.
    """

    def __init__(self, seed: int | None = None):
        self._np_random_state = np.random.RandomState(seed or 0)
