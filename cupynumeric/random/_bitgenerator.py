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

import time
from abc import abstractproperty
from typing import TYPE_CHECKING

import numpy as np

from .._array.array import ndarray
from ..config import BitGeneratorType
from ..runtime import runtime

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike


class BitGenerator:
    def __init__(
        self, seed: int | None = None, forceBuild: bool = False
    ) -> None:
        """
        BitGenerator(seed=None)

        Base Class for generic BitGenerators, which provide a stream
        of random bits based on different algorithms. Must be overridden.

        Parameters
        ----------
        seed : {int, None}, optional
            A seed to initialize the `BitGenerator`. If None, then fresh,
            unpredictable entropy will be pulled from the OS.

        See Also
        --------
        numpy.random.BitGenerator

        Availability
        ------------
        Multiple GPUs, Multiple CPUs
        """
        # make sure handle is always initialized
        self.handle = 0

        if type(self) is BitGenerator:
            raise NotImplementedError(
                "BitGenerator is a base class and cannot be instantized"
            )

        self.seed = seed or time.perf_counter_ns()
        self.flags = 0
        self.handle = runtime.bitgenerator_create(
            self.generatorType, seed, self.flags, forceBuild
        )

    @abstractproperty
    def generatorType(self) -> BitGeneratorType: ...

    def __del__(self) -> None:
        if self.handle != 0:
            runtime.bitgenerator_destroy(self.handle, disposing=True)

    # when output is false => skip ahead
    def random_raw(self, shape: NdShapeLike | None = None) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(np.uint32))
        res._thunk.bitgenerator_random_raw(
            self.handle, self.generatorType, self.seed, self.flags
        )
        return res

    def integers(
        self,
        low: int,
        high: int | None = None,
        shape: NdShapeLike | None = None,
        type: npt.DTypeLike = np.int64,
        endpoint: bool = False,
    ) -> ndarray:
        if shape is None:
            shape = ()
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(type))
        if high is None:
            high = low
            low = 0
        if endpoint:
            high = high + 1
        res._thunk.bitgenerator_integers(
            self.handle, self.generatorType, self.seed, self.flags, low, high
        )
        return res

    def random(
        self,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
        res: ndarray | None = None,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        if res is None:
            res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_uniform(
            self.handle, self.generatorType, self.seed, self.flags, 0, 1
        )
        return res

    def lognormal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_lognormal(
            self.handle, self.generatorType, self.seed, self.flags, mean, sigma
        )
        return res

    def normal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_normal(
            self.handle, self.generatorType, self.seed, self.flags, mean, sigma
        )
        return res

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_uniform(
            self.handle, self.generatorType, self.seed, self.flags, low, high
        )
        return res

    def poisson(self, lam: float, shape: NdShapeLike | None = None) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(np.uint32))
        res._thunk.bitgenerator_poisson(
            self.handle, self.generatorType, self.seed, self.flags, lam
        )
        return res

    def exponential(
        self,
        scale: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_exponential(
            self.handle, self.generatorType, self.seed, self.flags, scale
        )
        return res

    def gumbel(
        self,
        mu: float = 0.0,
        beta: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_gumbel(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def laplace(
        self,
        mu: float = 0.0,
        beta: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_laplace(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def logistic(
        self,
        mu: float = 0.0,
        beta: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_logistic(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def pareto(
        self,
        alpha: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_pareto(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def power(
        self,
        alpha: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_power(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def rayleigh(
        self,
        sigma: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_rayleigh(
            self.handle, self.generatorType, self.seed, self.flags, sigma
        )
        return res

    def cauchy(
        self,
        x0: float,
        gamma: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_cauchy(
            self.handle, self.generatorType, self.seed, self.flags, x0, gamma
        )
        return res

    def triangular(
        self,
        a: float,
        b: float,
        c: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_triangular(
            self.handle, self.generatorType, self.seed, self.flags, a, b, c
        )
        return res

    def weibull(
        self,
        lam: float,
        k: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_weibull(
            self.handle, self.generatorType, self.seed, self.flags, lam, k
        )
        return res

    def bytes(self, length: int | tuple[int, ...]) -> ndarray:
        if not isinstance(length, tuple):
            length = (length,)
        res = ndarray(length, dtype=np.dtype(np.uint8))
        res._thunk.bitgenerator_bytes(
            self.handle, self.generatorType, self.seed, self.flags
        )
        return res

    def beta(
        self,
        a: float,
        b: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_beta(
            self.handle, self.generatorType, self.seed, self.flags, a, b
        )
        return res

    def f(
        self,
        dfnum: float,
        dfden: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_f(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
            dfnum,
            dfden,
        )
        return res

    def logseries(
        self,
        p: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_logseries(
            self.handle, self.generatorType, self.seed, self.flags, p
        )
        return res

    def noncentral_f(
        self,
        dfnum: float,
        dfden: float,
        nonc: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_noncentral_f(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
            dfnum,
            dfden,
            nonc,
        )
        return res

    def chisquare(
        self,
        df: float,
        nonc: float = 0.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_chisquare(
            self.handle, self.generatorType, self.seed, self.flags, df, nonc
        )
        return res

    def gamma(
        self,
        k: float,
        theta: float = 1.0,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_gamma(
            self.handle, self.generatorType, self.seed, self.flags, k, theta
        )
        return res

    def standard_t(
        self,
        df: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_standard_t(
            self.handle, self.generatorType, self.seed, self.flags, df
        )
        return res

    def hypergeometric(
        self,
        ngood: int,
        nbad: int,
        nsample: int,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_hypergeometric(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
            ngood,
            nbad,
            nsample,
        )
        return res

    def vonmises(
        self,
        mu: float,
        kappa: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_vonmises(
            self.handle, self.generatorType, self.seed, self.flags, mu, kappa
        )
        return res

    def zipf(
        self,
        alpha: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_zipf(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def geometric(
        self,
        p: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_geometric(
            self.handle, self.generatorType, self.seed, self.flags, p
        )
        return res

    def wald(
        self,
        mean: float,
        scale: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_wald(
            self.handle, self.generatorType, self.seed, self.flags, mean, scale
        )
        return res

    def binomial(
        self,
        ntrials: int,
        p: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_binomial(
            self.handle, self.generatorType, self.seed, self.flags, ntrials, p
        )
        return res

    def negative_binomial(
        self,
        ntrials: int,
        p: float,
        shape: NdShapeLike | None = None,
        dtype: npt.DTypeLike = np.uint32,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_negative_binomial(
            self.handle, self.generatorType, self.seed, self.flags, ntrials, p
        )
        return res


class XORWOW(BitGenerator):
    @property
    def generatorType(self) -> BitGeneratorType:
        return BitGeneratorType.XORWOW


class MRG32k3a(BitGenerator):
    @property
    def generatorType(self) -> BitGeneratorType:
        return BitGeneratorType.MRG32K3A


class PHILOX4_32_10(BitGenerator):
    @property
    def generatorType(self) -> BitGeneratorType:
        return BitGeneratorType.PHILOX4_32_10
