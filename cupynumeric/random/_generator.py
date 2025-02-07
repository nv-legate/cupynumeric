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

from ._bitgenerator import XORWOW, BitGenerator

if TYPE_CHECKING:
    import numpy.typing as npt

    from .._array.array import ndarray
    from ..types import NdShapeLike


class Generator:
    def __init__(self, bit_generator: BitGenerator) -> None:
        """
        Generator(bit_generator)

        Container for the BitGenerators.

        ``Generator`` exposes a number of methods for generating random numbers
        drawn from a variety of probability distributions. In addition to the
        distribution-specific arguments, each method takes a keyword argument
        `size` that defaults to ``None``. If `size` is ``None``, then a single
        value is generated and returned. If `size` is an integer, then a 1-D
        array filled with generated values is returned. If `size` is a tuple,
        then an array with that shape is filled and returned.


        The function :func:`cupynumeric.random.default_rng` will instantiate
        a `Generator` with cuPyNumeric's default `BitGenerator`.

        Parameters
        ----------
        bit_generator : BitGenerator
            BitGenerator to use as the core generator.

        See Also
        --------
        numpy.random.Generator
        default_rng : Recommended constructor for `Generator`.

        Availability
        ------------
        Multiple GPUs, Multiple CPUs

        """
        self.bit_generator = bit_generator

    def beta(
        self,
        a: float,
        b: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.beta(a=a, b=b, shape=size, dtype=dtype)

    def binomial(
        self,
        ntrials: int,
        p: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        return self.bit_generator.binomial(
            ntrials=ntrials, p=p, shape=size, dtype=dtype
        )

    def bytes(self, length: int | tuple[int, ...]) -> ndarray:
        return self.bit_generator.bytes(length=length)

    def cauchy(
        self,
        x0: float,
        gamma: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.cauchy(
            x0=x0, gamma=gamma, shape=size, dtype=dtype
        )

    def chisquare(
        self,
        df: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.chisquare(
            df=df, nonc=0.0, shape=size, dtype=dtype
        )

    def exponential(
        self,
        scale: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.exponential(
            scale=scale, shape=size, dtype=dtype
        )

    def f(
        self,
        dfnum: float,
        dfden: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.f(
            dfnum=dfnum, dfden=dfden, shape=size, dtype=dtype
        )

    def gamma(
        self,
        shape: float,
        scale: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.gamma(
            k=shape, theta=scale, shape=size, dtype=dtype
        )

    def geometric(
        self,
        p: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        return self.bit_generator.geometric(p=p, shape=size, dtype=dtype)

    def gumbel(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.gumbel(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def hypergeometric(
        self,
        ngood: int,
        nbad: int,
        nsample: int,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        return self.bit_generator.hypergeometric(
            ngood=ngood, nbad=nbad, nsample=nsample, shape=size, dtype=dtype
        )

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.int64,
        endpoint: bool = False,
    ) -> ndarray:
        return self.bit_generator.integers(low, high, size, dtype, endpoint)

    def laplace(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.laplace(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def logistic(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.logistic(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def lognormal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.lognormal(mean, sigma, size, dtype)

    def logseries(
        self,
        p: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        return self.bit_generator.logseries(p=p, shape=size, dtype=dtype)

    def negative_binomial(
        self,
        ntrials: int,
        p: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        return self.bit_generator.negative_binomial(
            ntrials, p, shape=size, dtype=dtype
        )

    def noncentral_chisquare(
        self,
        df: float,
        nonc: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.chisquare(
            df=df, nonc=nonc, shape=size, dtype=dtype
        )

    def noncentral_f(
        self,
        dfnum: float,
        dfden: float,
        nonc: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.noncentral_f(
            dfnum=dfnum, dfden=dfden, nonc=nonc, shape=size, dtype=dtype
        )

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.normal(
            mean=loc, sigma=scale, shape=size, dtype=dtype
        )

    def pareto(
        self,
        a: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.pareto(alpha=a, shape=size, dtype=dtype)

    def poisson(
        self, lam: float = 1.0, size: NdShapeLike | None = None
    ) -> ndarray:
        return self.bit_generator.poisson(lam, size)

    def power(
        self,
        a: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.power(alpha=a, shape=size, dtype=dtype)

    def random(
        self,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
        out: ndarray | None = None,
    ) -> ndarray:
        if out is not None:
            if size is not None and out.shape != size:
                raise ValueError(
                    "size must match out.shape when used together"
                )
            if out.dtype != dtype:
                raise TypeError(
                    "Supplied output array has the wrong type. "
                    "Expected {}, got {}".format(dtype, out.dtype)
                )
        return self.bit_generator.random(size, dtype, out)

    def rayleigh(
        self,
        scale: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.rayleigh(
            sigma=scale, shape=size, dtype=dtype
        )

    def standard_cauchy(
        self,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.cauchy(0.0, 1.0, size, dtype)

    def standard_exponential(
        self,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.exponential(1.0, size, dtype)

    def standard_gamma(
        self,
        shape: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.gamma(shape=shape, scale=1.0, size=size, dtype=dtype)

    def standard_t(
        self,
        df: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.standard_t(df=df, shape=size, dtype=dtype)

    def triangular(
        self,
        left: float,
        mode: float,
        right: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.triangular(
            a=left, b=right, c=mode, shape=size, dtype=dtype
        )

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.uniform(low, high, size, dtype)

    def vonmises(
        self,
        mu: float,
        kappa: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.vonmises(
            mu=mu, kappa=kappa, shape=size, dtype=dtype
        )

    def wald(
        self,
        mean: float,
        scale: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.wald(mean, scale, shape=size, dtype=dtype)

    def weibull(
        self,
        a: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.weibull(lam=1, k=a, shape=size, dtype=dtype)

    def zipf(
        self,
        a: float,
        size: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        return self.bit_generator.zipf(alpha=a, shape=size, dtype=dtype)


def default_rng(
    seed: int | BitGenerator | Generator | None = None,
) -> Generator:
    """
    Construct a new Generator with the default BitGenerator (XORWOW).

    Parameters
    ----------
    seed : {None, int,  BitGenerator, Generator}, optional
        A seed to initialize the ``BitGenerator``. If ``None``, then fresh,
        unpredictable entropy will be pulled from the OS.
        Additionally, when passed a ``BitGenerator``, it will be wrapped by
        ``Generator``. If passed a ``Generator``, it will be returned
        unaltered.

    Returns
    -------
    Generator
        The initialized generator object.

    Notes
    -----
    If ``seed`` is not a ``BitGenerator`` or a ``Generator``, a new
    ``BitGenerator`` is instantiated. This function does not manage
    a default global instance.

    See Also
    --------
    numpy.random.default_rng

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if seed is None:
        return Generator(XORWOW())
    elif isinstance(seed, BitGenerator):
        return Generator(seed)
    elif isinstance(seed, Generator):
        return seed
    else:
        return Generator(XORWOW(seed))


_static_generator = None


def get_static_generator() -> Generator:
    global _static_generator
    if _static_generator is None:
        _static_generator = default_rng()
    return _static_generator
