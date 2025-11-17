# Copyright 2025 NVIDIA Corporation
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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Literal

from ..config import ConvertCode
from ..runtime import runtime

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from legate.core import Scalar
    from typing_extensions import CapsuleType

    from ..config import (
        BinaryOpCode,
        BitGeneratorType,
        FFTDirection,
        FFTType,
        UnaryOpCode,
        UnaryRedCode,
        WindowOpCode,
    )
    from ..types import (
        BitOrder,
        BoundsMode,
        CastingKind,
        ConvolveMethod,
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
    on cuPyNumeric ndarrays.

    :meta private:
    """

    def __init__(self, dtype: np.dtype[Any]) -> None:
        self.library = runtime.library
        self.dtype = dtype

    @abstractmethod
    def __dlpack__(
        self,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType: ...

    @abstractmethod
    def __dlpack_device__(self) -> tuple[int, int]: ...

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

    @property
    @abstractmethod
    def shape(self) -> NdShape: ...

    @abstractmethod
    def __numpy_array__(self) -> npt.NDArray[Any]: ...

    @abstractmethod
    def imag(self) -> NumPyThunk: ...

    @abstractmethod
    def real(self) -> NumPyThunk: ...

    @abstractmethod
    def conj(self) -> NumPyThunk: ...

    @abstractmethod
    def convolve(
        self,
        input: Any,
        filter: Any,
        mode: ConvolveMode,
        method: ConvolveMethod,
    ) -> None: ...

    @abstractmethod
    def fft(
        self,
        rhs: Any,
        axes: Sequence[int],
        kind: FFTType,
        direction: FFTDirection,
    ) -> None: ...

    @abstractmethod
    def copy(self, rhs: Any, deep: bool) -> None: ...

    @abstractmethod
    def repeat(
        self, repeats: Any, axis: int, scalar_repeats: bool
    ) -> NumPyThunk: ...

    @abstractmethod
    def pad(
        self,
        pad_width: tuple[tuple[int, int], ...],
        mode: str,
        constant_value_thunk: Any = None,
        constant_rows: int = 0,
        constant_cols: int = 0,
    ) -> None: ...

    @property
    @abstractmethod
    def scalar(self) -> bool: ...

    @abstractmethod
    def get_item(self, key: Any) -> NumPyThunk: ...

    @abstractmethod
    def set_item(self, key: Any, value: Any) -> None: ...

    @abstractmethod
    def reshape(self, newshape: NdShape, order: OrderType) -> NumPyThunk: ...

    @abstractmethod
    def squeeze(self, axis: int | tuple[int, ...] | None) -> NumPyThunk: ...

    @abstractmethod
    def swapaxes(self, axis1: int, axis2: int) -> NumPyThunk: ...

    @abstractmethod
    def convert(
        self,
        rhs: Any,
        warn: bool = True,
        nan_op: ConvertCode = ConvertCode.NOOP,
        temporary: bool = False,
    ) -> None: ...

    @abstractmethod
    def fill(self, value: Any) -> None: ...

    @abstractmethod
    def transpose(self, axes: tuple[int, ...] | list[int]) -> NumPyThunk: ...

    @abstractmethod
    def flip(self, rhs: Any, axes: int | tuple[int, ...] | None) -> None: ...

    @abstractmethod
    def take(
        self,
        indices: Any,
        axis: int | None = None,
        out: Any | None = None,
        mode: BoundsMode = "raise",
    ) -> Any: ...

    @abstractmethod
    def take_along_axis(
        self,
        indices: Any,
        axis: int,
        out: Any | None = None,
        mode: BoundsMode = "raise",
    ) -> Any: ...

    @abstractmethod
    def contract(
        self,
        lhs_modes: list[str],
        rhs1_thunk: Any,
        rhs1_modes: list[str],
        rhs2_thunk: Any,
        rhs2_modes: list[str],
        mode2extent: dict[str, int],
    ) -> None: ...

    @abstractmethod
    def choose(self, rhs: Any, *args: Any) -> None: ...

    @abstractmethod
    def select(
        self,
        condlist: Iterable[Any],
        choicelist: Iterable[Any],
        default: npt.NDArray[Any],
    ) -> None: ...

    @abstractmethod
    def _diag_helper(
        self, rhs: Any, offset: int, naxes: int, extract: bool, trace: bool
    ) -> None: ...

    @abstractmethod
    def put(self, indices: Any, values: Any, check_bounds: bool) -> None: ...

    @abstractmethod
    def putmask(self, mask: Any, values: Any) -> None: ...

    @abstractmethod
    def eye(self, k: int) -> None: ...

    @abstractmethod
    def arange(self, start: float, stop: float, step: float) -> None: ...

    @abstractmethod
    def tile(self, rhs: Any, reps: Any | Sequence[int]) -> None: ...

    @abstractmethod
    def trilu(self, rhs: Any, k: int, lower: bool) -> None: ...

    @abstractmethod
    def bincount(
        self, rhs: Any, weights: NumPyThunk | None = None
    ) -> None: ...

    @abstractmethod
    def nonzero(self) -> tuple[NumPyThunk, ...]: ...

    @abstractmethod
    def bitgenerator_random_raw(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_integers(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        low: int,
        high: int,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_uniform(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        low: float,
        high: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_lognormal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mean: float,
        sigma: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_normal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mean: float,
        sigma: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_poisson(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        lam: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_exponential(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        scale: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_gumbel(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        beta: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_laplace(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        beta: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_logistic(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        beta: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_pareto(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        alpha: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_power(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        alpha: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_rayleigh(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        sigma: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_cauchy(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        x0: float,
        gamma: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_triangular(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        a: float,
        b: float,
        c: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_weibull(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        lam: float,
        k: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_bytes(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_beta(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        a: float,
        b: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        dfnum: float,
        dfden: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_logseries(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        p: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_noncentral_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        dfnum: float,
        dfden: float,
        nonc: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_chisquare(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        df: float,
        nonc: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_gamma(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        k: float,
        theta: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_standard_t(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        df: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_hypergeometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        ngood: int,
        nbad: int,
        nsample: int,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_vonmises(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        kappa: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_zipf(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        alpha: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_geometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        p: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_wald(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mean: float,
        scale: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        ntrials: int,
        p: float,
    ) -> None: ...

    @abstractmethod
    def bitgenerator_negative_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        ntrials: int,
        p: float,
    ) -> None: ...

    @abstractmethod
    def random_uniform(self) -> None: ...

    @abstractmethod
    def partition(
        self,
        rhs: Any,
        kth: int | Sequence[int],
        argpartition: bool = False,
        axis: int | None = -1,
        kind: SelectKind = "introselect",
        order: str | list[str] | None = None,
    ) -> None: ...

    @abstractmethod
    def random_normal(self) -> None: ...

    @abstractmethod
    def random_integer(
        self, low: int | npt.NDArray[Any], high: int | npt.NDArray[Any]
    ) -> None: ...

    @abstractmethod
    def searchsorted(
        self, rhs: Any, v: Any, side: SortSide = "left"
    ) -> None: ...

    @abstractmethod
    def sort(
        self,
        rhs: Any,
        argsort: bool = False,
        axis: int | None = -1,
        kind: SortType = "quicksort",
        order: str | list[str] | None = None,
    ) -> None: ...

    @abstractmethod
    def _matmul(
        self,
        rhs: Any,
        out: Any | None = None,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    @abstractmethod
    def _add(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _multiply(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _subtract(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _true_divide(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _floor_divide(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _logaddexp(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _logaddexp2(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _negative(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _positive(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _power(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _float_power(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _remainder(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _absolute(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _rint(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _sign(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _conjugate(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _exp(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _exp2(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _log(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _log2(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _log10(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _expm1(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _log1p(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _square(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _sqrt(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _cbrt(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _reciprocal(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _gcd(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _lcm(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _greater_equal(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _equal(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _greater(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _less(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _less_equal(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _not_equal(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _logical_and(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _logical_or(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _logical_xor(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _logical_not(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _maximum(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _minimum(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _bitwise_and(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _bitwise_or(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _bitwise_xor(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _invert(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _left_shift(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _right_shift(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _isfinite(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _isinf(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _isnan(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _fabs(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _signbit(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _copysign(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _nextafter(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _ldexp(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _fmod(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _floor(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _ceil(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _trunc(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _sin(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _cos(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _tan(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arcsin(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arccos(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arctan(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arctan2(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _hypot(
        self,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _sinh(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _cosh(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _tanh(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arcsinh(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arccosh(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _arctanh(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _deg2rad(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def _rad2deg(
        self,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def unary_op(
        self,
        op: UnaryOpCode,
        rhs: Any,
        where: Any,
        args: tuple[Scalar, ...] = (),
        multiout: Any | None = None,
    ) -> None: ...

    @abstractmethod
    def unary_reduction(
        self,
        op: UnaryRedCode,
        rhs: Any,
        where: Any,
        orig_axis: int | None,
        axes: tuple[int, ...],
        keepdims: bool,
        args: tuple[Scalar, ...],
        initial: Any,
    ) -> None: ...

    @abstractmethod
    def isclose(
        self, rhs1: Any, rhs2: Any, rtol: float, atol: float, equal_nan: bool
    ) -> None: ...

    @abstractmethod
    def binary_op(
        self,
        op: BinaryOpCode,
        rhs1: Any,
        rhs2: Any,
        where: Any,
        args: tuple[Scalar, ...],
    ) -> None: ...

    @abstractmethod
    def binary_reduction(
        self,
        op: BinaryOpCode,
        rhs1: Any,
        rhs2: Any,
        broadcast: NdShape | None,
        args: tuple[Scalar, ...],
    ) -> None: ...

    @abstractmethod
    def broadcast_to(self, shape: NdShape) -> NumPyThunk: ...

    @abstractmethod
    def argwhere(self) -> NumPyThunk: ...

    @abstractmethod
    def where(self, rhs1: Any, rhs2: Any, rhs3: Any) -> None: ...

    @abstractmethod
    def cholesky(self, src: Any) -> None: ...

    @abstractmethod
    def eig(self, ew: Any, ev: Any) -> None: ...

    @abstractmethod
    def eigvals(self, ew: Any) -> None: ...

    @abstractmethod
    def eigh(self, ew: Any, ev: Any, uplo_l: bool) -> None: ...

    @abstractmethod
    def eigvalsh(self, ew: Any, uplo_l: bool) -> None: ...

    @abstractmethod
    def qr(self, q: Any, r: Any) -> None: ...

    @abstractmethod
    def solve(self, a: Any, b: Any) -> None: ...

    @abstractmethod
    def svd(self, u: Any, s: Any, vh: Any) -> None: ...

    @abstractmethod
    def scan(
        self,
        op: int,
        rhs: Any,
        axis: int,
        dtype: npt.DTypeLike | None,
        nan_to_identity: bool,
    ) -> None: ...

    @abstractmethod
    def unique(self) -> NumPyThunk: ...

    @abstractmethod
    def create_window(
        self, op_code: WindowOpCode, M: Any, *args: Any
    ) -> None: ...

    @abstractmethod
    def packbits(
        self, src: Any, axis: int | None, bitorder: BitOrder
    ) -> None: ...

    @abstractmethod
    def unpackbits(
        self, src: Any, axis: int | None, bitorder: BitOrder
    ) -> None: ...

    @abstractmethod
    def _wrap(self, src: Any, new_len: int) -> None: ...

    @abstractmethod
    def histogram(self, src: Any, bins: Any, weights: Any) -> None: ...

    @abstractmethod
    def histogramdd(
        self, coords: Any, weights: Any, bins_set: Any
    ) -> None: ...

    def stencil_hint(
        self, low_offsets: tuple[int, ...], high_offsets: tuple[int, ...]
    ) -> None: ...

    @abstractmethod
    def in1d(
        self,
        ar2: NumPyThunk,
        assume_unique: bool = False,
        invert: bool = False,
        kind: Literal["sort", "table"] | None = None,
        ar2_min: int = 0,
        ar2_max: int = 0,
    ) -> NumPyThunk: ...
