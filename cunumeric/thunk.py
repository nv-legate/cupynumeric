# Copyright 2021-2022 NVIDIA Corporation
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

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

from .config import ConvertCode

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from legate.core import FieldID, Future, Region

    from .config import (
        BinaryOpCode,
        BitGeneratorType,
        FFTDirection,
        FFTType,
        UnaryOpCode,
        UnaryRedCode,
        WindowOpCode,
    )
    from .runtime import Runtime
    from .types import (
        BitOrder,
        ConvolveMode,
        NdShape,
        OrderType,
        SelectKind,
        SortSide,
        SortType,
    )


class NumPyThunk(ABC):
    """This is the base class for NumPy computations. It has methods
    for all the kinds of computations and operations that can be done
    on cuNumeric ndarrays.

    :meta private:
    """

    def __init__(self, runtime: Runtime, dtype: np.dtype[Any]) -> None:
        self.runtime = runtime
        self.context = runtime.legate_context
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        s = 1
        if self.ndim == 0:
            return s
        for p in self.shape:
            s *= p
        return s

    # Abstract methods

    @abstractproperty
    def storage(self) -> Union[Future, tuple[Region, Union[int, FieldID]]]:
        """Return the Legion storage primitive for this NumPy thunk"""
        ...

    @abstractproperty
    def shape(self) -> NdShape:
        ...

    @abstractmethod
    def __numpy_array__(self) -> npt.NDArray[Any]:
        ...

    @abstractproperty
    def can_write_through_numpy_array(self) -> bool:
        ...

    @abstractmethod
    def imag(self) -> NumPyThunk:
        ...

    @abstractmethod
    def real(self) -> NumPyThunk:
        ...

    @abstractmethod
    def conj(self) -> NumPyThunk:
        ...

    @abstractmethod
    def convolve(self, v: Any, out: Any, mode: ConvolveMode) -> None:
        ...

    @abstractmethod
    def fft(
        self,
        out: Any,
        axes: Sequence[int],
        kind: FFTType,
        direction: FFTDirection,
    ) -> None:
        ...

    @abstractmethod
    def copy(self, rhs: Any, deep: bool) -> None:
        ...

    @abstractmethod
    def repeat(
        self, repeats: Any, axis: int, scalar_repeats: bool
    ) -> NumPyThunk:
        ...

    @property
    @abstractmethod
    def scalar(self) -> bool:
        ...

    @abstractmethod
    def get_scalar_array(self) -> npt.NDArray[Any]:
        ...

    @abstractmethod
    def get_item(self, key: Any) -> NumPyThunk:
        ...

    @abstractmethod
    def set_item(self, key: Any, value: Any) -> None:
        ...

    @abstractmethod
    def reshape(self, newshape: NdShape, order: OrderType) -> NumPyThunk:
        ...

    @abstractmethod
    def squeeze(self, axis: Optional[int]) -> NumPyThunk:
        ...

    @abstractmethod
    def swapaxes(self, axis1: int, axis2: int) -> NumPyThunk:
        ...

    @abstractmethod
    def convert(
        self,
        rhs: Any,
        warn: bool = True,
        nan_op: ConvertCode = ConvertCode.NOOP,
        temporary: bool = False,
    ) -> None:
        ...

    @abstractmethod
    def fill(self, value: Any) -> None:
        ...

    @abstractmethod
    def transpose(
        self, axes: Union[None, tuple[int, ...], list[int]]
    ) -> NumPyThunk:
        ...

    @abstractmethod
    def flip(self, rhs: Any, axes: Union[None, int, tuple[int, ...]]) -> None:
        ...

    @abstractmethod
    def contract(
        self,
        lhs_modes: list[str],
        rhs1_thunk: Any,
        rhs1_modes: list[str],
        rhs2_thunk: Any,
        rhs2_modes: list[str],
        mode2extent: dict[str, int],
    ) -> None:
        ...

    @abstractmethod
    def choose(self, rhs: Any, *args: Any) -> None:
        ...

    @abstractmethod
    def select(
        self,
        condlist: Iterable[Any],
        choicelist: Iterable[Any],
        default: npt.NDArray[Any],
    ) -> None:
        ...

    @abstractmethod
    def _diag_helper(
        self, rhs: Any, offset: int, naxes: int, extract: bool, trace: bool
    ) -> None:
        ...

    @abstractmethod
    def put(self, indices: Any, values: Any, check_bounds: bool) -> None:
        ...

    @abstractmethod
    def putmask(self, mask: Any, values: Any) -> None:
        ...

    @abstractmethod
    def eye(self, k: int) -> None:
        ...

    @abstractmethod
    def arange(self, start: float, stop: float, step: float) -> None:
        ...

    @abstractmethod
    def tile(self, rhs: Any, reps: Union[Any, Sequence[int]]) -> None:
        ...

    @abstractmethod
    def trilu(self, rhs: Any, k: int, lower: bool) -> None:
        ...

    @abstractmethod
    def bincount(self, rhs: Any, weights: Optional[NumPyThunk] = None) -> None:
        ...

    @abstractmethod
    def nonzero(self) -> tuple[NumPyThunk, ...]:
        ...

    @abstractmethod
    def bitgenerator_random_raw(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_integers(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        low: int,
        high: int,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_uniform(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        low: float,
        high: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_lognormal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mean: float,
        sigma: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_normal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mean: float,
        sigma: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_poisson(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        lam: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_exponential(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        scale: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_gumbel(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_laplace(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_logistic(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_pareto(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        alpha: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_power(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        alpha: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_rayleigh(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        sigma: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_cauchy(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        x0: float,
        gamma: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_triangular(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        a: float,
        b: float,
        c: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_weibull(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        lam: float,
        k: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_bytes(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_beta(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        a: float,
        b: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        dfnum: float,
        dfden: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_logseries(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        p: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_noncentral_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        dfnum: float,
        dfden: float,
        nonc: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_chisquare(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        df: float,
        nonc: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_gamma(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        k: float,
        theta: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_standard_t(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        df: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_hypergeometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        ngood: int,
        nbad: int,
        nsample: int,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_vonmises(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        kappa: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_zipf(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        alpha: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_geometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        p: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_wald(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mean: float,
        scale: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        ntrials: int,
        p: float,
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_negative_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        ntrials: int,
        p: float,
    ) -> None:
        ...

    @abstractmethod
    def random_uniform(self) -> None:
        ...

    @abstractmethod
    def partition(
        self,
        rhs: Any,
        kth: Union[int, Sequence[int]],
        argpartition: bool = False,
        axis: Union[int, None] = -1,
        kind: SelectKind = "introselect",
        order: Union[None, str, list[str]] = None,
    ) -> None:
        ...

    @abstractmethod
    def random_normal(self) -> None:
        ...

    @abstractmethod
    def random_integer(
        self,
        low: Union[int, npt.NDArray[Any]],
        high: Union[int, npt.NDArray[Any]],
    ) -> None:
        ...

    @abstractmethod
    def searchsorted(self, rhs: Any, v: Any, side: SortSide = "left") -> None:
        ...

    @abstractmethod
    def sort(
        self,
        rhs: Any,
        argsort: bool = False,
        axis: Union[int, None] = -1,
        kind: SortType = "quicksort",
        order: Union[None, str, list[str]] = None,
    ) -> None:
        ...

    @abstractmethod
    def unary_op(
        self,
        op: UnaryOpCode,
        rhs: Any,
        where: Any,
        args: Any,
        multiout: Optional[Any] = None,
    ) -> None:
        ...

    @abstractmethod
    def unary_reduction(
        self,
        op: UnaryRedCode,
        rhs: Any,
        where: Any,
        orig_axis: Union[int, None],
        axes: tuple[int, ...],
        keepdims: bool,
        args: Any,
        initial: Any,
    ) -> None:
        ...

    @abstractmethod
    def isclose(
        self, rhs1: Any, rhs2: Any, rtol: float, atol: float, equal_nan: bool
    ) -> None:
        ...

    @abstractmethod
    def binary_op(
        self, op: BinaryOpCode, rhs1: Any, rhs2: Any, where: Any, args: Any
    ) -> None:
        ...

    @abstractmethod
    def binary_reduction(
        self,
        op: BinaryOpCode,
        rhs1: Any,
        rhs2: Any,
        broadcast: Union[NdShape, None],
        args: Any,
    ) -> None:
        ...

    @abstractmethod
    def broadcast_to(self, shape: NdShape) -> NumPyThunk:
        ...

    @abstractmethod
    def argwhere(self) -> NumPyThunk:
        ...

    @abstractmethod
    def where(self, rhs1: Any, rhs2: Any, rhs3: Any) -> None:
        ...

    @abstractmethod
    def cholesky(self, src: Any, no_tril: bool) -> None:
        ...

    @abstractmethod
    def solve(self, a: Any, b: Any) -> None:
        ...

    @abstractmethod
    def scan(
        self,
        op: int,
        rhs: Any,
        axis: int,
        dtype: Optional[npt.DTypeLike],
        nan_to_identity: bool,
    ) -> None:
        ...

    @abstractmethod
    def unique(self) -> NumPyThunk:
        ...

    @abstractmethod
    def create_window(self, op_code: WindowOpCode, M: Any, *args: Any) -> None:
        ...

    @abstractmethod
    def packbits(
        self, src: Any, axis: Union[int, None], bitorder: BitOrder
    ) -> None:
        ...

    @abstractmethod
    def unpackbits(
        self, src: Any, axis: Union[int, None], bitorder: BitOrder
    ) -> None:
        ...

    @abstractmethod
    def _wrap(self, src: Any, new_len: int) -> None:
        ...

    @abstractmethod
    def histogram(self, src: Any, bins: Any, weights: Any) -> None:
        ...
