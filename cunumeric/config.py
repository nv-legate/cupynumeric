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

import os
from abc import abstractmethod
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Union, cast

import numpy as np
from legate.core import Library, get_legate_runtime

if TYPE_CHECKING:
    import numpy.typing as npt

    from .runtime import Runtime


class _CunumericSharedLib:
    CUNUMERIC_ADVANCED_INDEXING: int
    CUNUMERIC_ARANGE: int
    CUNUMERIC_ARGWHERE: int
    CUNUMERIC_BATCHED_CHOLESKY: int
    CUNUMERIC_BINARY_OP: int
    CUNUMERIC_BINARY_RED: int
    CUNUMERIC_BINCOUNT: int
    CUNUMERIC_BINOP_ADD: int
    CUNUMERIC_BINOP_ARCTAN2: int
    CUNUMERIC_BINOP_BITWISE_AND: int
    CUNUMERIC_BINOP_BITWISE_OR: int
    CUNUMERIC_BINOP_BITWISE_XOR: int
    CUNUMERIC_BINOP_COPYSIGN: int
    CUNUMERIC_BINOP_DIVIDE: int
    CUNUMERIC_BINOP_EQUAL: int
    CUNUMERIC_BINOP_FLOAT_POWER: int
    CUNUMERIC_BINOP_FLOOR_DIVIDE: int
    CUNUMERIC_BINOP_FMOD: int
    CUNUMERIC_BINOP_GCD: int
    CUNUMERIC_BINOP_GREATER: int
    CUNUMERIC_BINOP_GREATER_EQUAL: int
    CUNUMERIC_BINOP_HYPOT: int
    CUNUMERIC_BINOP_ISCLOSE: int
    CUNUMERIC_BINOP_LCM: int
    CUNUMERIC_BINOP_LDEXP: int
    CUNUMERIC_BINOP_LEFT_SHIFT: int
    CUNUMERIC_BINOP_LESS: int
    CUNUMERIC_BINOP_LESS_EQUAL: int
    CUNUMERIC_BINOP_LOGADDEXP2: int
    CUNUMERIC_BINOP_LOGADDEXP: int
    CUNUMERIC_BINOP_LOGICAL_AND: int
    CUNUMERIC_BINOP_LOGICAL_OR: int
    CUNUMERIC_BINOP_LOGICAL_XOR: int
    CUNUMERIC_BINOP_MAXIMUM: int
    CUNUMERIC_BINOP_MINIMUM: int
    CUNUMERIC_BINOP_MOD: int
    CUNUMERIC_BINOP_MULTIPLY: int
    CUNUMERIC_BINOP_NEXTAFTER: int
    CUNUMERIC_BINOP_NOT_EQUAL: int
    CUNUMERIC_BINOP_POWER: int
    CUNUMERIC_BINOP_RIGHT_SHIFT: int
    CUNUMERIC_BINOP_SUBTRACT: int
    CUNUMERIC_BITGENERATOR: int
    CUNUMERIC_BITGENOP_DISTRIBUTION: int
    CUNUMERIC_BITGENTYPE_DEFAULT: int
    CUNUMERIC_BITGENTYPE_XORWOW: int
    CUNUMERIC_BITGENTYPE_MRG32K3A: int
    CUNUMERIC_BITGENTYPE_MTGP32: int
    CUNUMERIC_BITGENTYPE_MT19937: int
    CUNUMERIC_BITGENTYPE_PHILOX4_32_10: int
    CUNUMERIC_BITGENDIST_INTEGERS_16: int
    CUNUMERIC_BITGENDIST_INTEGERS_32: int
    CUNUMERIC_BITGENDIST_INTEGERS_64: int
    CUNUMERIC_BITGENDIST_UNIFORM_32: int
    CUNUMERIC_BITGENDIST_UNIFORM_64: int
    CUNUMERIC_BITGENDIST_LOGNORMAL_32: int
    CUNUMERIC_BITGENDIST_LOGNORMAL_64: int
    CUNUMERIC_BITGENDIST_NORMAL_32: int
    CUNUMERIC_BITGENDIST_NORMAL_64: int
    CUNUMERIC_BITGENDIST_POISSON: int
    CUNUMERIC_BITGENDIST_EXPONENTIAL_32: int
    CUNUMERIC_BITGENDIST_EXPONENTIAL_64: int
    CUNUMERIC_BITGENDIST_GUMBEL_32: int
    CUNUMERIC_BITGENDIST_GUMBEL_64: int
    CUNUMERIC_BITGENDIST_LAPLACE_32: int
    CUNUMERIC_BITGENDIST_LAPLACE_64: int
    CUNUMERIC_BITGENDIST_LOGISTIC_32: int
    CUNUMERIC_BITGENDIST_LOGISTIC_64: int
    CUNUMERIC_BITGENDIST_PARETO_32: int
    CUNUMERIC_BITGENDIST_PARETO_64: int
    CUNUMERIC_BITGENDIST_POWER_32: int
    CUNUMERIC_BITGENDIST_POWER_64: int
    CUNUMERIC_BITGENDIST_RAYLEIGH_32: int
    CUNUMERIC_BITGENDIST_RAYLEIGH_64: int
    CUNUMERIC_BITGENDIST_CAUCHY_32: int
    CUNUMERIC_BITGENDIST_CAUCHY_64: int
    CUNUMERIC_BITGENDIST_TRIANGULAR_32: int
    CUNUMERIC_BITGENDIST_TRIANGULAR_64: int
    CUNUMERIC_BITGENDIST_WEIBULL_32: int
    CUNUMERIC_BITGENDIST_WEIBULL_64: int
    CUNUMERIC_BITGENDIST_BYTES: int
    CUNUMERIC_BITGENDIST_BETA_32: int
    CUNUMERIC_BITGENDIST_BETA_64: int
    CUNUMERIC_BITGENDIST_F_32: int
    CUNUMERIC_BITGENDIST_F_64: int
    CUNUMERIC_BITGENDIST_LOGSERIES: int
    CUNUMERIC_BITGENDIST_NONCENTRAL_F_32: int
    CUNUMERIC_BITGENDIST_NONCENTRAL_F_64: int
    CUNUMERIC_BITGENDIST_CHISQUARE_32: int
    CUNUMERIC_BITGENDIST_CHISQUARE_64: int
    CUNUMERIC_BITGENDIST_GAMMA_32: int
    CUNUMERIC_BITGENDIST_GAMMA_64: int
    CUNUMERIC_BITGENDIST_STANDARD_T_32: int
    CUNUMERIC_BITGENDIST_STANDARD_T_64: int
    CUNUMERIC_BITGENDIST_HYPERGEOMETRIC: int
    CUNUMERIC_BITGENDIST_VONMISES_32: int
    CUNUMERIC_BITGENDIST_VONMISES_64: int
    CUNUMERIC_BITGENDIST_ZIPF: int
    CUNUMERIC_BITGENDIST_GEOMETRIC: int
    CUNUMERIC_BITGENDIST_WALD_32: int
    CUNUMERIC_BITGENDIST_WALD_64: int
    CUNUMERIC_BITGENDIST_BINOMIAL: int
    CUNUMERIC_BITGENDIST_NEGATIVE_BINOMIAL: int
    CUNUMERIC_BITGENOP_CREATE: int
    CUNUMERIC_BITGENOP_DESTROY: int
    CUNUMERIC_BITGENOP_RAND_RAW: int
    CUNUMERIC_BITORDER_BIG: int
    CUNUMERIC_BITORDER_LITTLE: int
    CUNUMERIC_CHOOSE: int
    CUNUMERIC_CONTRACT: int
    CUNUMERIC_CONVERT: int
    CUNUMERIC_CONVERT_NAN_NOOP: int
    CUNUMERIC_CONVERT_NAN_PROD: int
    CUNUMERIC_CONVERT_NAN_SUM: int
    CUNUMERIC_CONVOLVE: int
    CUNUMERIC_DIAG: int
    CUNUMERIC_DOT: int
    CUNUMERIC_EYE: int
    CUNUMERIC_FFT: int
    CUNUMERIC_FFT_C2C: int
    CUNUMERIC_FFT_C2R: int
    CUNUMERIC_FFT_D2Z: int
    CUNUMERIC_FFT_FORWARD: int
    CUNUMERIC_FFT_INVERSE: int
    CUNUMERIC_FFT_R2C: int
    CUNUMERIC_FFT_Z2D: int
    CUNUMERIC_FFT_Z2Z: int
    CUNUMERIC_FILL: int
    CUNUMERIC_FLIP: int
    CUNUMERIC_GEMM: int
    CUNUMERIC_HISTOGRAM: int
    CUNUMERIC_LOAD_CUDALIBS: int
    CUNUMERIC_MATMUL: int
    CUNUMERIC_MATVECMUL: int
    CUNUMERIC_MAX_MAPPERS: int
    CUNUMERIC_MAX_REDOPS: int
    CUNUMERIC_MAX_TASKS: int
    CUNUMERIC_NONZERO: int
    CUNUMERIC_PACKBITS: int
    CUNUMERIC_POTRF: int
    CUNUMERIC_PUTMASK: int
    CUNUMERIC_RAND: int
    CUNUMERIC_READ: int
    CUNUMERIC_RED_ALL: int
    CUNUMERIC_RED_ANY: int
    CUNUMERIC_RED_ARGMAX: int
    CUNUMERIC_RED_ARGMIN: int
    CUNUMERIC_RED_CONTAINS: int
    CUNUMERIC_RED_COUNT_NONZERO: int
    CUNUMERIC_RED_MAX: int
    CUNUMERIC_RED_MIN: int
    CUNUMERIC_RED_NANARGMAX: int
    CUNUMERIC_RED_NANARGMIN: int
    CUNUMERIC_RED_NANMAX: int
    CUNUMERIC_RED_NANMIN: int
    CUNUMERIC_RED_NANPROD: int
    CUNUMERIC_RED_NANSUM: int
    CUNUMERIC_RED_PROD: int
    CUNUMERIC_RED_SUM: int
    CUNUMERIC_RED_SUM_SQUARES: int
    CUNUMERIC_RED_VARIANCE: int
    CUNUMERIC_REPEAT: int
    CUNUMERIC_SCALAR_UNARY_RED: int
    CUNUMERIC_SCAN_GLOBAL: int
    CUNUMERIC_SCAN_LOCAL: int
    CUNUMERIC_SCAN_PROD: int
    CUNUMERIC_SCAN_SUM: int
    CUNUMERIC_SEARCHSORTED: int
    CUNUMERIC_SELECT: int
    CUNUMERIC_SOLVE: int
    CUNUMERIC_SORT: int
    CUNUMERIC_SYRK: int
    CUNUMERIC_TILE: int
    CUNUMERIC_TRANSPOSE_COPY_2D: int
    CUNUMERIC_TRILU: int
    CUNUMERIC_TRSM: int
    CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME: int
    CUNUMERIC_TUNABLE_NUM_GPUS: int
    CUNUMERIC_TUNABLE_NUM_PROCS: int
    CUNUMERIC_UNARY_OP: int
    CUNUMERIC_UNARY_RED: int
    CUNUMERIC_UNIQUE: int
    CUNUMERIC_UNIQUE_REDUCE: int
    CUNUMERIC_UNZIP_INDICES: int
    CUNUMERIC_UNLOAD_CUDALIBS: int
    CUNUMERIC_UNPACKBITS: int
    CUNUMERIC_UOP_ABSOLUTE: int
    CUNUMERIC_UOP_ARCCOS: int
    CUNUMERIC_UOP_ARCCOSH: int
    CUNUMERIC_UOP_ARCSIN: int
    CUNUMERIC_UOP_ARCSINH: int
    CUNUMERIC_UOP_ARCTAN: int
    CUNUMERIC_UOP_ARCTANH: int
    CUNUMERIC_UOP_CBRT: int
    CUNUMERIC_UOP_CEIL: int
    CUNUMERIC_UOP_CLIP: int
    CUNUMERIC_UOP_CONJ: int
    CUNUMERIC_UOP_COPY: int
    CUNUMERIC_UOP_COS: int
    CUNUMERIC_UOP_COSH: int
    CUNUMERIC_UOP_DEG2RAD: int
    CUNUMERIC_UOP_EXP2: int
    CUNUMERIC_UOP_EXP: int
    CUNUMERIC_UOP_EXPM1: int
    CUNUMERIC_UOP_FLOOR: int
    CUNUMERIC_UOP_FREXP: int
    CUNUMERIC_UOP_GETARG: int
    CUNUMERIC_UOP_IMAG: int
    CUNUMERIC_UOP_INVERT: int
    CUNUMERIC_UOP_ISFINITE: int
    CUNUMERIC_UOP_ISINF: int
    CUNUMERIC_UOP_ISNAN: int
    CUNUMERIC_UOP_LOG10: int
    CUNUMERIC_UOP_LOG1P: int
    CUNUMERIC_UOP_LOG2: int
    CUNUMERIC_UOP_LOG: int
    CUNUMERIC_UOP_LOGICAL_NOT: int
    CUNUMERIC_UOP_MODF: int
    CUNUMERIC_UOP_NEGATIVE: int
    CUNUMERIC_UOP_POSITIVE: int
    CUNUMERIC_UOP_RAD2DEG: int
    CUNUMERIC_UOP_REAL: int
    CUNUMERIC_UOP_RECIPROCAL: int
    CUNUMERIC_UOP_RINT: int
    CUNUMERIC_UOP_SIGN: int
    CUNUMERIC_UOP_SIGNBIT: int
    CUNUMERIC_UOP_SIN: int
    CUNUMERIC_UOP_SINH: int
    CUNUMERIC_UOP_SQRT: int
    CUNUMERIC_UOP_SQUARE: int
    CUNUMERIC_UOP_TAN: int
    CUNUMERIC_UOP_TANH: int
    CUNUMERIC_UOP_TRUNC: int
    CUNUMERIC_WHERE: int
    CUNUMERIC_WINDOW: int
    CUNUMERIC_WINDOW_BARLETT: int
    CUNUMERIC_WINDOW_BLACKMAN: int
    CUNUMERIC_WINDOW_HAMMING: int
    CUNUMERIC_WINDOW_HANNING: int
    CUNUMERIC_WINDOW_KAISER: int
    CUNUMERIC_WRAP: int
    CUNUMERIC_WRITE: int
    CUNUMERIC_ZIP: int

    @abstractmethod
    def cunumeric_has_curand(self) -> int:
        ...

    @abstractmethod
    def cunumeric_register_reduction_op(
        self, type_uid: int, elem_type_code: int
    ) -> None:
        ...


# Load the cuNumeric library first so we have a shard object that
# we can use to initialize all these configuration enumerations
class CuNumericLib(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.runtime: Union[Runtime, None] = None
        self.shared_object: Union[_CunumericSharedLib, None] = None

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from cunumeric.install_info import libpath

        return os.path.join(
            libpath, "libcunumeric" + self.get_library_extension()
        )

    def get_c_header(self) -> str:
        from cunumeric.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "cunumeric_perform_registration"

    def initialize(self, shared_object: _CunumericSharedLib) -> None:
        assert self.runtime is None
        self.shared_object = shared_object

    def set_runtime(self, runtime: Runtime) -> None:
        assert self.runtime is None
        assert self.shared_object is not None
        self.runtime = runtime

    def destroy(self) -> None:
        if self.runtime is not None:
            self.runtime.destroy()


CUNUMERIC_LIB_NAME = "cunumeric"
cunumeric_lib = CuNumericLib(CUNUMERIC_LIB_NAME)
cunumeric_context = get_legate_runtime().register_library(cunumeric_lib)
_cunumeric = cast(_CunumericSharedLib, cunumeric_lib.shared_object)


# Match these to CuNumericOpCode in cunumeric_c.h
@unique
class CuNumericOpCode(IntEnum):
    ADVANCED_INDEXING = _cunumeric.CUNUMERIC_ADVANCED_INDEXING
    ARANGE = _cunumeric.CUNUMERIC_ARANGE
    ARGWHERE = _cunumeric.CUNUMERIC_ARGWHERE
    BATCHED_CHOLESKY = _cunumeric.CUNUMERIC_BATCHED_CHOLESKY
    BINARY_OP = _cunumeric.CUNUMERIC_BINARY_OP
    BINARY_RED = _cunumeric.CUNUMERIC_BINARY_RED
    BINCOUNT = _cunumeric.CUNUMERIC_BINCOUNT
    BITGENERATOR = _cunumeric.CUNUMERIC_BITGENERATOR
    CHOOSE = _cunumeric.CUNUMERIC_CHOOSE
    CONTRACT = _cunumeric.CUNUMERIC_CONTRACT
    CONVERT = _cunumeric.CUNUMERIC_CONVERT
    CONVOLVE = _cunumeric.CUNUMERIC_CONVOLVE
    DIAG = _cunumeric.CUNUMERIC_DIAG
    DOT = _cunumeric.CUNUMERIC_DOT
    EYE = _cunumeric.CUNUMERIC_EYE
    FFT = _cunumeric.CUNUMERIC_FFT
    FILL = _cunumeric.CUNUMERIC_FILL
    FLIP = _cunumeric.CUNUMERIC_FLIP
    GEMM = _cunumeric.CUNUMERIC_GEMM
    HISTOGRAM = _cunumeric.CUNUMERIC_HISTOGRAM
    LOAD_CUDALIBS = _cunumeric.CUNUMERIC_LOAD_CUDALIBS
    MATMUL = _cunumeric.CUNUMERIC_MATMUL
    MATVECMUL = _cunumeric.CUNUMERIC_MATVECMUL
    NONZERO = _cunumeric.CUNUMERIC_NONZERO
    PACKBITS = _cunumeric.CUNUMERIC_PACKBITS
    POTRF = _cunumeric.CUNUMERIC_POTRF
    PUTMASK = _cunumeric.CUNUMERIC_PUTMASK
    RAND = _cunumeric.CUNUMERIC_RAND
    READ = _cunumeric.CUNUMERIC_READ
    REPEAT = _cunumeric.CUNUMERIC_REPEAT
    SCALAR_UNARY_RED = _cunumeric.CUNUMERIC_SCALAR_UNARY_RED
    SCAN_GLOBAL = _cunumeric.CUNUMERIC_SCAN_GLOBAL
    SCAN_LOCAL = _cunumeric.CUNUMERIC_SCAN_LOCAL
    SEARCHSORTED = _cunumeric.CUNUMERIC_SEARCHSORTED
    SELECT = _cunumeric.CUNUMERIC_SELECT
    SOLVE = _cunumeric.CUNUMERIC_SOLVE
    SORT = _cunumeric.CUNUMERIC_SORT
    SYRK = _cunumeric.CUNUMERIC_SYRK
    TILE = _cunumeric.CUNUMERIC_TILE
    TRANSPOSE_COPY_2D = _cunumeric.CUNUMERIC_TRANSPOSE_COPY_2D
    TRILU = _cunumeric.CUNUMERIC_TRILU
    TRSM = _cunumeric.CUNUMERIC_TRSM
    UNARY_OP = _cunumeric.CUNUMERIC_UNARY_OP
    UNARY_RED = _cunumeric.CUNUMERIC_UNARY_RED
    UNIQUE = _cunumeric.CUNUMERIC_UNIQUE
    UNIQUE_REDUCE = _cunumeric.CUNUMERIC_UNIQUE_REDUCE
    UNZIP = _cunumeric.CUNUMERIC_UNZIP_INDICES
    UNLOAD_CUDALIBS = _cunumeric.CUNUMERIC_UNLOAD_CUDALIBS
    UNPACKBITS = _cunumeric.CUNUMERIC_UNPACKBITS
    WHERE = _cunumeric.CUNUMERIC_WHERE
    WINDOW = _cunumeric.CUNUMERIC_WINDOW
    WRAP = _cunumeric.CUNUMERIC_WRAP
    WRITE = _cunumeric.CUNUMERIC_WRITE
    ZIP = _cunumeric.CUNUMERIC_ZIP


# Match these to CuNumericUnaryOpCode in cunumeric_c.h
@unique
class UnaryOpCode(IntEnum):
    ABSOLUTE = _cunumeric.CUNUMERIC_UOP_ABSOLUTE
    ARCCOS = _cunumeric.CUNUMERIC_UOP_ARCCOS
    ARCCOSH = _cunumeric.CUNUMERIC_UOP_ARCCOSH
    ARCSIN = _cunumeric.CUNUMERIC_UOP_ARCSIN
    ARCSINH = _cunumeric.CUNUMERIC_UOP_ARCSINH
    ARCTAN = _cunumeric.CUNUMERIC_UOP_ARCTAN
    ARCTANH = _cunumeric.CUNUMERIC_UOP_ARCTANH
    CBRT = _cunumeric.CUNUMERIC_UOP_CBRT
    CEIL = _cunumeric.CUNUMERIC_UOP_CEIL
    CLIP = _cunumeric.CUNUMERIC_UOP_CLIP
    CONJ = _cunumeric.CUNUMERIC_UOP_CONJ
    COPY = _cunumeric.CUNUMERIC_UOP_COPY
    COS = _cunumeric.CUNUMERIC_UOP_COS
    COSH = _cunumeric.CUNUMERIC_UOP_COSH
    DEG2RAD = _cunumeric.CUNUMERIC_UOP_DEG2RAD
    EXP = _cunumeric.CUNUMERIC_UOP_EXP
    EXP2 = _cunumeric.CUNUMERIC_UOP_EXP2
    EXPM1 = _cunumeric.CUNUMERIC_UOP_EXPM1
    FLOOR = _cunumeric.CUNUMERIC_UOP_FLOOR
    FREXP = _cunumeric.CUNUMERIC_UOP_FREXP
    GETARG = _cunumeric.CUNUMERIC_UOP_GETARG
    IMAG = _cunumeric.CUNUMERIC_UOP_IMAG
    INVERT = _cunumeric.CUNUMERIC_UOP_INVERT
    ISFINITE = _cunumeric.CUNUMERIC_UOP_ISFINITE
    ISINF = _cunumeric.CUNUMERIC_UOP_ISINF
    ISNAN = _cunumeric.CUNUMERIC_UOP_ISNAN
    LOG = _cunumeric.CUNUMERIC_UOP_LOG
    LOG10 = _cunumeric.CUNUMERIC_UOP_LOG10
    LOG1P = _cunumeric.CUNUMERIC_UOP_LOG1P
    LOG2 = _cunumeric.CUNUMERIC_UOP_LOG2
    LOGICAL_NOT = _cunumeric.CUNUMERIC_UOP_LOGICAL_NOT
    MODF = _cunumeric.CUNUMERIC_UOP_MODF
    NEGATIVE = _cunumeric.CUNUMERIC_UOP_NEGATIVE
    POSITIVE = _cunumeric.CUNUMERIC_UOP_POSITIVE
    RAD2DEG = _cunumeric.CUNUMERIC_UOP_RAD2DEG
    REAL = _cunumeric.CUNUMERIC_UOP_REAL
    RECIPROCAL = _cunumeric.CUNUMERIC_UOP_RECIPROCAL
    RINT = _cunumeric.CUNUMERIC_UOP_RINT
    SIGN = _cunumeric.CUNUMERIC_UOP_SIGN
    SIGNBIT = _cunumeric.CUNUMERIC_UOP_SIGNBIT
    SIN = _cunumeric.CUNUMERIC_UOP_SIN
    SINH = _cunumeric.CUNUMERIC_UOP_SINH
    SQRT = _cunumeric.CUNUMERIC_UOP_SQRT
    SQUARE = _cunumeric.CUNUMERIC_UOP_SQUARE
    TAN = _cunumeric.CUNUMERIC_UOP_TAN
    TANH = _cunumeric.CUNUMERIC_UOP_TANH
    TRUNC = _cunumeric.CUNUMERIC_UOP_TRUNC


# Match these to CuNumericUnaryRedCode in cunumeric_c.h
@unique
class UnaryRedCode(IntEnum):
    ALL = _cunumeric.CUNUMERIC_RED_ALL
    ANY = _cunumeric.CUNUMERIC_RED_ANY
    ARGMAX = _cunumeric.CUNUMERIC_RED_ARGMAX
    ARGMIN = _cunumeric.CUNUMERIC_RED_ARGMIN
    CONTAINS = _cunumeric.CUNUMERIC_RED_CONTAINS
    COUNT_NONZERO = _cunumeric.CUNUMERIC_RED_COUNT_NONZERO
    MAX = _cunumeric.CUNUMERIC_RED_MAX
    MIN = _cunumeric.CUNUMERIC_RED_MIN
    NANARGMAX = _cunumeric.CUNUMERIC_RED_NANARGMAX
    NANARGMIN = _cunumeric.CUNUMERIC_RED_NANARGMIN
    NANMAX = _cunumeric.CUNUMERIC_RED_NANMAX
    NANMIN = _cunumeric.CUNUMERIC_RED_NANMIN
    NANPROD = _cunumeric.CUNUMERIC_RED_NANPROD
    NANSUM = _cunumeric.CUNUMERIC_RED_NANSUM
    PROD = _cunumeric.CUNUMERIC_RED_PROD
    SUM = _cunumeric.CUNUMERIC_RED_SUM
    SUM_SQUARES = _cunumeric.CUNUMERIC_RED_SUM_SQUARES
    VARIANCE = _cunumeric.CUNUMERIC_RED_VARIANCE


# Match these to CuNumericBinaryOpCode in cunumeric_c.h
@unique
class BinaryOpCode(IntEnum):
    ADD = _cunumeric.CUNUMERIC_BINOP_ADD
    ARCTAN2 = _cunumeric.CUNUMERIC_BINOP_ARCTAN2
    BITWISE_AND = _cunumeric.CUNUMERIC_BINOP_BITWISE_AND
    BITWISE_OR = _cunumeric.CUNUMERIC_BINOP_BITWISE_OR
    BITWISE_XOR = _cunumeric.CUNUMERIC_BINOP_BITWISE_XOR
    COPYSIGN = _cunumeric.CUNUMERIC_BINOP_COPYSIGN
    DIVIDE = _cunumeric.CUNUMERIC_BINOP_DIVIDE
    EQUAL = _cunumeric.CUNUMERIC_BINOP_EQUAL
    FLOAT_POWER = _cunumeric.CUNUMERIC_BINOP_FLOAT_POWER
    FLOOR_DIVIDE = _cunumeric.CUNUMERIC_BINOP_FLOOR_DIVIDE
    FMOD = _cunumeric.CUNUMERIC_BINOP_FMOD
    GCD = _cunumeric.CUNUMERIC_BINOP_GCD
    GREATER = _cunumeric.CUNUMERIC_BINOP_GREATER
    GREATER_EQUAL = _cunumeric.CUNUMERIC_BINOP_GREATER_EQUAL
    HYPOT = _cunumeric.CUNUMERIC_BINOP_HYPOT
    ISCLOSE = _cunumeric.CUNUMERIC_BINOP_ISCLOSE
    LCM = _cunumeric.CUNUMERIC_BINOP_LCM
    LDEXP = _cunumeric.CUNUMERIC_BINOP_LDEXP
    LEFT_SHIFT = _cunumeric.CUNUMERIC_BINOP_LEFT_SHIFT
    LESS = _cunumeric.CUNUMERIC_BINOP_LESS
    LESS_EQUAL = _cunumeric.CUNUMERIC_BINOP_LESS_EQUAL
    LOGADDEXP = _cunumeric.CUNUMERIC_BINOP_LOGADDEXP
    LOGADDEXP2 = _cunumeric.CUNUMERIC_BINOP_LOGADDEXP2
    LOGICAL_AND = _cunumeric.CUNUMERIC_BINOP_LOGICAL_AND
    LOGICAL_OR = _cunumeric.CUNUMERIC_BINOP_LOGICAL_OR
    LOGICAL_XOR = _cunumeric.CUNUMERIC_BINOP_LOGICAL_XOR
    MAXIMUM = _cunumeric.CUNUMERIC_BINOP_MAXIMUM
    MINIMUM = _cunumeric.CUNUMERIC_BINOP_MINIMUM
    MOD = _cunumeric.CUNUMERIC_BINOP_MOD
    MULTIPLY = _cunumeric.CUNUMERIC_BINOP_MULTIPLY
    NEXTAFTER = _cunumeric.CUNUMERIC_BINOP_NEXTAFTER
    NOT_EQUAL = _cunumeric.CUNUMERIC_BINOP_NOT_EQUAL
    POWER = _cunumeric.CUNUMERIC_BINOP_POWER
    RIGHT_SHIFT = _cunumeric.CUNUMERIC_BINOP_RIGHT_SHIFT
    SUBTRACT = _cunumeric.CUNUMERIC_BINOP_SUBTRACT


@unique
class WindowOpCode(IntEnum):
    BARLETT = _cunumeric.CUNUMERIC_WINDOW_BARLETT
    BLACKMAN = _cunumeric.CUNUMERIC_WINDOW_BLACKMAN
    HAMMING = _cunumeric.CUNUMERIC_WINDOW_HAMMING
    HANNING = _cunumeric.CUNUMERIC_WINDOW_HANNING
    KAISER = _cunumeric.CUNUMERIC_WINDOW_KAISER


# Match these to RandGenCode in rand_util.h
@unique
class RandGenCode(IntEnum):
    UNIFORM = 1
    NORMAL = 2
    INTEGER = 3


# Match these to CuNumericTunable in cunumeric_c.h
@unique
class CuNumericTunable(IntEnum):
    NUM_GPUS = _cunumeric.CUNUMERIC_TUNABLE_NUM_GPUS
    NUM_PROCS = _cunumeric.CUNUMERIC_TUNABLE_NUM_PROCS
    MAX_EAGER_VOLUME = _cunumeric.CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME


# Match these to CuNumericScanCode in cunumeric_c.h
@unique
class ScanCode(IntEnum):
    PROD = _cunumeric.CUNUMERIC_SCAN_PROD
    SUM = _cunumeric.CUNUMERIC_SCAN_SUM


# Match these to CuNumericConvertCode in cunumeric_c.h
@unique
class ConvertCode(IntEnum):
    NOOP = _cunumeric.CUNUMERIC_CONVERT_NAN_NOOP
    PROD = _cunumeric.CUNUMERIC_CONVERT_NAN_PROD
    SUM = _cunumeric.CUNUMERIC_CONVERT_NAN_SUM


# Match these to BitGeneratorOperation in cunumeric_c.h
@unique
class BitGeneratorOperation(IntEnum):
    CREATE = _cunumeric.CUNUMERIC_BITGENOP_CREATE
    DESTROY = _cunumeric.CUNUMERIC_BITGENOP_DESTROY
    RAND_RAW = _cunumeric.CUNUMERIC_BITGENOP_RAND_RAW
    DISTRIBUTION = _cunumeric.CUNUMERIC_BITGENOP_DISTRIBUTION


# Match these to BitGeneratorType in cunumeric_c.h
@unique
class BitGeneratorType(IntEnum):
    DEFAULT = _cunumeric.CUNUMERIC_BITGENTYPE_DEFAULT
    XORWOW = _cunumeric.CUNUMERIC_BITGENTYPE_XORWOW
    MRG32K3A = _cunumeric.CUNUMERIC_BITGENTYPE_MRG32K3A
    MTGP32 = _cunumeric.CUNUMERIC_BITGENTYPE_MTGP32
    MT19937 = _cunumeric.CUNUMERIC_BITGENTYPE_MT19937
    PHILOX4_32_10 = _cunumeric.CUNUMERIC_BITGENTYPE_PHILOX4_32_10


# Match these to BitGeneratorDistribution in cunumeric_c.h
@unique
class BitGeneratorDistribution(IntEnum):
    INTEGERS_16 = _cunumeric.CUNUMERIC_BITGENDIST_INTEGERS_16
    INTEGERS_32 = _cunumeric.CUNUMERIC_BITGENDIST_INTEGERS_32
    INTEGERS_64 = _cunumeric.CUNUMERIC_BITGENDIST_INTEGERS_64
    UNIFORM_32 = _cunumeric.CUNUMERIC_BITGENDIST_UNIFORM_32
    UNIFORM_64 = _cunumeric.CUNUMERIC_BITGENDIST_UNIFORM_64
    LOGNORMAL_32 = _cunumeric.CUNUMERIC_BITGENDIST_LOGNORMAL_32
    LOGNORMAL_64 = _cunumeric.CUNUMERIC_BITGENDIST_LOGNORMAL_64
    NORMAL_32 = _cunumeric.CUNUMERIC_BITGENDIST_NORMAL_32
    NORMAL_64 = _cunumeric.CUNUMERIC_BITGENDIST_NORMAL_64
    POISSON = _cunumeric.CUNUMERIC_BITGENDIST_POISSON
    EXPONENTIAL_32 = _cunumeric.CUNUMERIC_BITGENDIST_EXPONENTIAL_32
    EXPONENTIAL_64 = _cunumeric.CUNUMERIC_BITGENDIST_EXPONENTIAL_64
    GUMBEL_32 = _cunumeric.CUNUMERIC_BITGENDIST_GUMBEL_32
    GUMBEL_64 = _cunumeric.CUNUMERIC_BITGENDIST_GUMBEL_64
    LAPLACE_32 = _cunumeric.CUNUMERIC_BITGENDIST_LAPLACE_32
    LAPLACE_64 = _cunumeric.CUNUMERIC_BITGENDIST_LAPLACE_64
    LOGISTIC_32 = _cunumeric.CUNUMERIC_BITGENDIST_LOGISTIC_32
    LOGISTIC_64 = _cunumeric.CUNUMERIC_BITGENDIST_LOGISTIC_64
    PARETO_32 = _cunumeric.CUNUMERIC_BITGENDIST_PARETO_32
    PARETO_64 = _cunumeric.CUNUMERIC_BITGENDIST_PARETO_64
    POWER_32 = _cunumeric.CUNUMERIC_BITGENDIST_POWER_32
    POWER_64 = _cunumeric.CUNUMERIC_BITGENDIST_POWER_64
    RAYLEIGH_32 = _cunumeric.CUNUMERIC_BITGENDIST_RAYLEIGH_32
    RAYLEIGH_64 = _cunumeric.CUNUMERIC_BITGENDIST_RAYLEIGH_64
    CAUCHY_32 = _cunumeric.CUNUMERIC_BITGENDIST_CAUCHY_32
    CAUCHY_64 = _cunumeric.CUNUMERIC_BITGENDIST_CAUCHY_64
    TRIANGULAR_32 = _cunumeric.CUNUMERIC_BITGENDIST_TRIANGULAR_32
    TRIANGULAR_64 = _cunumeric.CUNUMERIC_BITGENDIST_TRIANGULAR_64
    WEIBULL_32 = _cunumeric.CUNUMERIC_BITGENDIST_WEIBULL_32
    WEIBULL_64 = _cunumeric.CUNUMERIC_BITGENDIST_WEIBULL_64
    BYTES = _cunumeric.CUNUMERIC_BITGENDIST_BYTES
    BETA_32 = _cunumeric.CUNUMERIC_BITGENDIST_BETA_32
    BETA_64 = _cunumeric.CUNUMERIC_BITGENDIST_BETA_64
    F_32 = _cunumeric.CUNUMERIC_BITGENDIST_F_32
    F_64 = _cunumeric.CUNUMERIC_BITGENDIST_F_64
    LOGSERIES = _cunumeric.CUNUMERIC_BITGENDIST_LOGSERIES
    NONCENTRAL_F_32 = _cunumeric.CUNUMERIC_BITGENDIST_NONCENTRAL_F_32
    NONCENTRAL_F_64 = _cunumeric.CUNUMERIC_BITGENDIST_NONCENTRAL_F_64
    CHISQUARE_32 = _cunumeric.CUNUMERIC_BITGENDIST_CHISQUARE_32
    CHISQUARE_64 = _cunumeric.CUNUMERIC_BITGENDIST_CHISQUARE_64
    GAMMA_32 = _cunumeric.CUNUMERIC_BITGENDIST_GAMMA_32
    GAMMA_64 = _cunumeric.CUNUMERIC_BITGENDIST_GAMMA_64
    STANDARD_T_32 = _cunumeric.CUNUMERIC_BITGENDIST_STANDARD_T_32
    STANDARD_T_64 = _cunumeric.CUNUMERIC_BITGENDIST_STANDARD_T_64
    HYPERGEOMETRIC = _cunumeric.CUNUMERIC_BITGENDIST_HYPERGEOMETRIC
    VONMISES_32 = _cunumeric.CUNUMERIC_BITGENDIST_VONMISES_32
    VONMISES_64 = _cunumeric.CUNUMERIC_BITGENDIST_VONMISES_64
    ZIPF = _cunumeric.CUNUMERIC_BITGENDIST_ZIPF
    GEOMETRIC = _cunumeric.CUNUMERIC_BITGENDIST_GEOMETRIC
    WALD_32 = _cunumeric.CUNUMERIC_BITGENDIST_WALD_32
    WALD_64 = _cunumeric.CUNUMERIC_BITGENDIST_WALD_64
    BINOMIAL = _cunumeric.CUNUMERIC_BITGENDIST_BINOMIAL
    NEGATIVE_BINOMIAL = _cunumeric.CUNUMERIC_BITGENDIST_NEGATIVE_BINOMIAL


# Match these to fftType in fft_util.h
class FFTType:
    def __init__(
        self,
        name: str,
        type_id: int,
        input_dtype: npt.DTypeLike,
        output_dtype: npt.DTypeLike,
        single_precision: bool,
        complex_type: Union[FFTType, None] = None,
    ) -> None:
        self._name = name
        self._type_id = type_id
        self._complex_type = self if complex_type is None else complex_type
        self._input_dtype = input_dtype
        self._output_dtype = output_dtype
        self._single_precision = single_precision

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return str(self)

    @property
    def type_id(self) -> int:
        return self._type_id

    @property
    def complex(self) -> FFTType:
        return self._complex_type

    @property
    def input_dtype(self) -> npt.DTypeLike:
        return self._input_dtype

    @property
    def output_dtype(self) -> npt.DTypeLike:
        return self._output_dtype

    @property
    def is_single_precision(self) -> bool:
        return self._single_precision


FFT_C2C = FFTType(
    "C2C",
    _cunumeric.CUNUMERIC_FFT_C2C,
    np.complex64,
    np.complex64,
    True,
)

FFT_Z2Z = FFTType(
    "Z2Z",
    _cunumeric.CUNUMERIC_FFT_Z2Z,
    np.complex128,
    np.complex128,
    False,
)

FFT_R2C = FFTType(
    "R2C",
    _cunumeric.CUNUMERIC_FFT_R2C,
    np.float32,
    np.complex64,
    True,
    FFT_C2C,
)

FFT_C2R = FFTType(
    "C2R",
    _cunumeric.CUNUMERIC_FFT_C2R,
    np.complex64,
    np.float32,
    True,
    FFT_C2C,
)

FFT_D2Z = FFTType(
    "D2Z",
    _cunumeric.CUNUMERIC_FFT_D2Z,
    np.float64,
    np.complex128,
    False,
    FFT_Z2Z,
)

FFT_Z2D = FFTType(
    "Z2D",
    _cunumeric.CUNUMERIC_FFT_Z2D,
    np.complex128,
    np.float64,
    False,
    FFT_Z2Z,
)


class FFTCode:
    @staticmethod
    def real_to_complex_code(dtype: npt.DTypeLike) -> FFTType:
        if dtype == np.float64:
            return FFT_D2Z
        elif dtype == np.float32:
            return FFT_R2C
        else:
            raise TypeError(
                (
                    "Data type for FFT not supported "
                    "(supported types are float32 and float64)"
                )
            )

    @staticmethod
    def complex_to_real_code(dtype: npt.DTypeLike) -> FFTType:
        if dtype == np.complex128:
            return FFT_Z2D
        elif dtype == np.complex64:
            return FFT_C2R
        else:
            raise TypeError(
                (
                    "Data type for FFT not supported "
                    "(supported types are complex64 and complex128)"
                )
            )


@unique
class FFTDirection(IntEnum):
    FORWARD = _cunumeric.CUNUMERIC_FFT_FORWARD
    INVERSE = _cunumeric.CUNUMERIC_FFT_INVERSE


# Match these to CuNumericBitorder in cunumeric_c.h
@unique
class Bitorder(IntEnum):
    BIG = _cunumeric.CUNUMERIC_BITORDER_BIG
    LITTLE = _cunumeric.CUNUMERIC_BITORDER_LITTLE


@unique
class FFTNormalization(IntEnum):
    FORWARD = 1
    INVERSE = 2
    ORTHOGONAL = 3

    @staticmethod
    def from_string(in_string: str) -> Union[FFTNormalization, None]:
        if in_string == "forward":
            return FFTNormalization.FORWARD
        elif in_string == "ortho":
            return FFTNormalization.ORTHOGONAL
        elif in_string == "backward" or in_string is None:
            return FFTNormalization.INVERSE
        else:
            return None

    @staticmethod
    def reverse(in_string: Union[str, None]) -> str:
        if in_string == "forward":
            return "backward"
        elif in_string == "backward" or in_string is None:
            return "forward"
        else:
            return in_string
