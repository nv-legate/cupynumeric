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

import os
import platform
from abc import abstractmethod
from ctypes import CDLL, RTLD_GLOBAL
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Any, cast

import cffi  # type: ignore
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class _ReductionOpIds:
    argmax_redop_id: int
    argmin_redop_id: int


class _CupynumericSharedLib:
    CUPYNUMERIC_ADVANCED_INDEXING: int
    CUPYNUMERIC_ARANGE: int
    CUPYNUMERIC_ARGWHERE: int
    CUPYNUMERIC_BATCHED_CHOLESKY: int
    CUPYNUMERIC_BINARY_OP: int
    CUPYNUMERIC_BINARY_RED: int
    CUPYNUMERIC_BINCOUNT: int
    CUPYNUMERIC_BINOP_ADD: int
    CUPYNUMERIC_BINOP_ARCTAN2: int
    CUPYNUMERIC_BINOP_BITWISE_AND: int
    CUPYNUMERIC_BINOP_BITWISE_OR: int
    CUPYNUMERIC_BINOP_BITWISE_XOR: int
    CUPYNUMERIC_BINOP_COPYSIGN: int
    CUPYNUMERIC_BINOP_DIVIDE: int
    CUPYNUMERIC_BINOP_EQUAL: int
    CUPYNUMERIC_BINOP_FLOAT_POWER: int
    CUPYNUMERIC_BINOP_FLOOR_DIVIDE: int
    CUPYNUMERIC_BINOP_FMOD: int
    CUPYNUMERIC_BINOP_GCD: int
    CUPYNUMERIC_BINOP_GREATER: int
    CUPYNUMERIC_BINOP_GREATER_EQUAL: int
    CUPYNUMERIC_BINOP_HYPOT: int
    CUPYNUMERIC_BINOP_ISCLOSE: int
    CUPYNUMERIC_BINOP_LCM: int
    CUPYNUMERIC_BINOP_LDEXP: int
    CUPYNUMERIC_BINOP_LEFT_SHIFT: int
    CUPYNUMERIC_BINOP_LESS: int
    CUPYNUMERIC_BINOP_LESS_EQUAL: int
    CUPYNUMERIC_BINOP_LOGADDEXP2: int
    CUPYNUMERIC_BINOP_LOGADDEXP: int
    CUPYNUMERIC_BINOP_LOGICAL_AND: int
    CUPYNUMERIC_BINOP_LOGICAL_OR: int
    CUPYNUMERIC_BINOP_LOGICAL_XOR: int
    CUPYNUMERIC_BINOP_MAXIMUM: int
    CUPYNUMERIC_BINOP_MINIMUM: int
    CUPYNUMERIC_BINOP_MOD: int
    CUPYNUMERIC_BINOP_MULTIPLY: int
    CUPYNUMERIC_BINOP_NEXTAFTER: int
    CUPYNUMERIC_BINOP_NOT_EQUAL: int
    CUPYNUMERIC_BINOP_POWER: int
    CUPYNUMERIC_BINOP_RIGHT_SHIFT: int
    CUPYNUMERIC_BINOP_SUBTRACT: int
    CUPYNUMERIC_BITGENERATOR: int
    CUPYNUMERIC_BITGENOP_DISTRIBUTION: int
    CUPYNUMERIC_BITGENTYPE_DEFAULT: int
    CUPYNUMERIC_BITGENTYPE_XORWOW: int
    CUPYNUMERIC_BITGENTYPE_MRG32K3A: int
    CUPYNUMERIC_BITGENTYPE_MTGP32: int
    CUPYNUMERIC_BITGENTYPE_MT19937: int
    CUPYNUMERIC_BITGENTYPE_PHILOX4_32_10: int
    CUPYNUMERIC_BITGENDIST_INTEGERS_16: int
    CUPYNUMERIC_BITGENDIST_INTEGERS_32: int
    CUPYNUMERIC_BITGENDIST_INTEGERS_64: int
    CUPYNUMERIC_BITGENDIST_UNIFORM_32: int
    CUPYNUMERIC_BITGENDIST_UNIFORM_64: int
    CUPYNUMERIC_BITGENDIST_LOGNORMAL_32: int
    CUPYNUMERIC_BITGENDIST_LOGNORMAL_64: int
    CUPYNUMERIC_BITGENDIST_NORMAL_32: int
    CUPYNUMERIC_BITGENDIST_NORMAL_64: int
    CUPYNUMERIC_BITGENDIST_POISSON: int
    CUPYNUMERIC_BITGENDIST_EXPONENTIAL_32: int
    CUPYNUMERIC_BITGENDIST_EXPONENTIAL_64: int
    CUPYNUMERIC_BITGENDIST_GUMBEL_32: int
    CUPYNUMERIC_BITGENDIST_GUMBEL_64: int
    CUPYNUMERIC_BITGENDIST_LAPLACE_32: int
    CUPYNUMERIC_BITGENDIST_LAPLACE_64: int
    CUPYNUMERIC_BITGENDIST_LOGISTIC_32: int
    CUPYNUMERIC_BITGENDIST_LOGISTIC_64: int
    CUPYNUMERIC_BITGENDIST_PARETO_32: int
    CUPYNUMERIC_BITGENDIST_PARETO_64: int
    CUPYNUMERIC_BITGENDIST_POWER_32: int
    CUPYNUMERIC_BITGENDIST_POWER_64: int
    CUPYNUMERIC_BITGENDIST_RAYLEIGH_32: int
    CUPYNUMERIC_BITGENDIST_RAYLEIGH_64: int
    CUPYNUMERIC_BITGENDIST_CAUCHY_32: int
    CUPYNUMERIC_BITGENDIST_CAUCHY_64: int
    CUPYNUMERIC_BITGENDIST_TRIANGULAR_32: int
    CUPYNUMERIC_BITGENDIST_TRIANGULAR_64: int
    CUPYNUMERIC_BITGENDIST_WEIBULL_32: int
    CUPYNUMERIC_BITGENDIST_WEIBULL_64: int
    CUPYNUMERIC_BITGENDIST_BYTES: int
    CUPYNUMERIC_BITGENDIST_BETA_32: int
    CUPYNUMERIC_BITGENDIST_BETA_64: int
    CUPYNUMERIC_BITGENDIST_F_32: int
    CUPYNUMERIC_BITGENDIST_F_64: int
    CUPYNUMERIC_BITGENDIST_LOGSERIES: int
    CUPYNUMERIC_BITGENDIST_NONCENTRAL_F_32: int
    CUPYNUMERIC_BITGENDIST_NONCENTRAL_F_64: int
    CUPYNUMERIC_BITGENDIST_CHISQUARE_32: int
    CUPYNUMERIC_BITGENDIST_CHISQUARE_64: int
    CUPYNUMERIC_BITGENDIST_GAMMA_32: int
    CUPYNUMERIC_BITGENDIST_GAMMA_64: int
    CUPYNUMERIC_BITGENDIST_STANDARD_T_32: int
    CUPYNUMERIC_BITGENDIST_STANDARD_T_64: int
    CUPYNUMERIC_BITGENDIST_HYPERGEOMETRIC: int
    CUPYNUMERIC_BITGENDIST_VONMISES_32: int
    CUPYNUMERIC_BITGENDIST_VONMISES_64: int
    CUPYNUMERIC_BITGENDIST_ZIPF: int
    CUPYNUMERIC_BITGENDIST_GEOMETRIC: int
    CUPYNUMERIC_BITGENDIST_WALD_32: int
    CUPYNUMERIC_BITGENDIST_WALD_64: int
    CUPYNUMERIC_BITGENDIST_BINOMIAL: int
    CUPYNUMERIC_BITGENDIST_NEGATIVE_BINOMIAL: int
    CUPYNUMERIC_BITGENOP_CREATE: int
    CUPYNUMERIC_BITGENOP_DESTROY: int
    CUPYNUMERIC_BITGENOP_RAND_RAW: int
    CUPYNUMERIC_BITORDER_BIG: int
    CUPYNUMERIC_BITORDER_LITTLE: int
    CUPYNUMERIC_CHOOSE: int
    CUPYNUMERIC_CONTRACT: int
    CUPYNUMERIC_CONVERT: int
    CUPYNUMERIC_CONVERT_NAN_NOOP: int
    CUPYNUMERIC_CONVERT_NAN_PROD: int
    CUPYNUMERIC_CONVERT_NAN_SUM: int
    CUPYNUMERIC_CONVOLVE: int
    CUPYNUMERIC_CONVOLVE_AUTO: int
    CUPYNUMERIC_CONVOLVE_DIRECT: int
    CUPYNUMERIC_CONVOLVE_FFT: int
    CUPYNUMERIC_DIAG: int
    CUPYNUMERIC_DOT: int
    CUPYNUMERIC_EYE: int
    CUPYNUMERIC_FFT: int
    CUPYNUMERIC_FFT_C2C: int
    CUPYNUMERIC_FFT_C2R: int
    CUPYNUMERIC_FFT_D2Z: int
    CUPYNUMERIC_FFT_FORWARD: int
    CUPYNUMERIC_FFT_INVERSE: int
    CUPYNUMERIC_FFT_R2C: int
    CUPYNUMERIC_FFT_Z2D: int
    CUPYNUMERIC_FFT_Z2Z: int
    CUPYNUMERIC_FILL: int
    CUPYNUMERIC_FLIP: int
    CUPYNUMERIC_GEEV: int
    CUPYNUMERIC_GEMM: int
    CUPYNUMERIC_HISTOGRAM: int
    CUPYNUMERIC_HISTOGRAMDD: int
    CUPYNUMERIC_IN1D: int
    CUPYNUMERIC_LOAD_CUDALIBS: int
    CUPYNUMERIC_MATMUL: int
    CUPYNUMERIC_MATVECMUL: int
    CUPYNUMERIC_MAX_MAPPERS: int
    CUPYNUMERIC_MAX_REDOPS: int
    CUPYNUMERIC_MAX_TASKS: int
    CUPYNUMERIC_MP_POTRF: int
    CUPYNUMERIC_MP_QR: int
    CUPYNUMERIC_MP_SOLVE: int
    CUPYNUMERIC_NONZERO: int
    CUPYNUMERIC_PACKBITS: int
    CUPYNUMERIC_PAD: int
    CUPYNUMERIC_POTRF: int
    CUPYNUMERIC_PUTMASK: int
    CUPYNUMERIC_QR: int
    CUPYNUMERIC_RAND: int
    CUPYNUMERIC_READ: int
    CUPYNUMERIC_RED_ALL: int
    CUPYNUMERIC_RED_ANY: int
    CUPYNUMERIC_RED_ARGMAX: int
    CUPYNUMERIC_RED_ARGMIN: int
    CUPYNUMERIC_RED_CONTAINS: int
    CUPYNUMERIC_RED_COUNT_NONZERO: int
    CUPYNUMERIC_RED_MAX: int
    CUPYNUMERIC_RED_MIN: int
    CUPYNUMERIC_RED_NANARGMAX: int
    CUPYNUMERIC_RED_NANARGMIN: int
    CUPYNUMERIC_RED_NANMAX: int
    CUPYNUMERIC_RED_NANMIN: int
    CUPYNUMERIC_RED_NANPROD: int
    CUPYNUMERIC_RED_NANSUM: int
    CUPYNUMERIC_RED_PROD: int
    CUPYNUMERIC_RED_SUM: int
    CUPYNUMERIC_RED_SUM_SQUARES: int
    CUPYNUMERIC_RED_VARIANCE: int
    CUPYNUMERIC_REPEAT: int
    CUPYNUMERIC_SCALAR_UNARY_RED: int
    CUPYNUMERIC_SCAN_GLOBAL: int
    CUPYNUMERIC_SCAN_LOCAL: int
    CUPYNUMERIC_SCAN_PROD: int
    CUPYNUMERIC_SCAN_SUM: int
    CUPYNUMERIC_SEARCHSORTED: int
    CUPYNUMERIC_SELECT: int
    CUPYNUMERIC_SOLVE: int
    CUPYNUMERIC_SORT: int
    CUPYNUMERIC_SVD: int
    CUPYNUMERIC_SYEV: int
    CUPYNUMERIC_SYRK: int
    CUPYNUMERIC_TAKE: int
    CUPYNUMERIC_TILE: int
    CUPYNUMERIC_TRANSPOSE_COPY_2D: int
    CUPYNUMERIC_TRILU: int
    CUPYNUMERIC_TRSM: int
    CUPYNUMERIC_UNARY_OP: int
    CUPYNUMERIC_UNARY_RED: int
    CUPYNUMERIC_UNIQUE: int
    CUPYNUMERIC_UNIQUE_REDUCE: int
    CUPYNUMERIC_UNLOAD_CUDALIBS: int
    CUPYNUMERIC_UNPACKBITS: int
    CUPYNUMERIC_UOP_ABSOLUTE: int
    CUPYNUMERIC_UOP_ANGLE: int
    CUPYNUMERIC_UOP_ARCCOS: int
    CUPYNUMERIC_UOP_ARCCOSH: int
    CUPYNUMERIC_UOP_ARCSIN: int
    CUPYNUMERIC_UOP_ARCSINH: int
    CUPYNUMERIC_UOP_ARCTAN: int
    CUPYNUMERIC_UOP_ARCTANH: int
    CUPYNUMERIC_UOP_CBRT: int
    CUPYNUMERIC_UOP_CEIL: int
    CUPYNUMERIC_UOP_CLIP: int
    CUPYNUMERIC_UOP_CONJ: int
    CUPYNUMERIC_UOP_COPY: int
    CUPYNUMERIC_UOP_COS: int
    CUPYNUMERIC_UOP_COSH: int
    CUPYNUMERIC_UOP_DEG2RAD: int
    CUPYNUMERIC_UOP_EXP2: int
    CUPYNUMERIC_UOP_EXP: int
    CUPYNUMERIC_UOP_EXPM1: int
    CUPYNUMERIC_UOP_FLOOR: int
    CUPYNUMERIC_UOP_FREXP: int
    CUPYNUMERIC_UOP_GETARG: int
    CUPYNUMERIC_UOP_IMAG: int
    CUPYNUMERIC_UOP_INVERT: int
    CUPYNUMERIC_UOP_ISFINITE: int
    CUPYNUMERIC_UOP_ISINF: int
    CUPYNUMERIC_UOP_ISNAN: int
    CUPYNUMERIC_UOP_LOG10: int
    CUPYNUMERIC_UOP_LOG1P: int
    CUPYNUMERIC_UOP_LOG2: int
    CUPYNUMERIC_UOP_LOG: int
    CUPYNUMERIC_UOP_LOGICAL_NOT: int
    CUPYNUMERIC_UOP_MODF: int
    CUPYNUMERIC_UOP_NEGATIVE: int
    CUPYNUMERIC_UOP_POSITIVE: int
    CUPYNUMERIC_UOP_RAD2DEG: int
    CUPYNUMERIC_UOP_REAL: int
    CUPYNUMERIC_UOP_RECIPROCAL: int
    CUPYNUMERIC_UOP_RINT: int
    CUPYNUMERIC_UOP_ROUND: int
    CUPYNUMERIC_UOP_SIGN: int
    CUPYNUMERIC_UOP_SIGNBIT: int
    CUPYNUMERIC_UOP_SIN: int
    CUPYNUMERIC_UOP_SINH: int
    CUPYNUMERIC_UOP_SQRT: int
    CUPYNUMERIC_UOP_SQUARE: int
    CUPYNUMERIC_UOP_TAN: int
    CUPYNUMERIC_UOP_TANH: int
    CUPYNUMERIC_UOP_TRUNC: int
    CUPYNUMERIC_WHERE: int
    CUPYNUMERIC_WINDOW: int
    CUPYNUMERIC_WINDOW_BARLETT: int
    CUPYNUMERIC_WINDOW_BLACKMAN: int
    CUPYNUMERIC_WINDOW_HAMMING: int
    CUPYNUMERIC_WINDOW_HANNING: int
    CUPYNUMERIC_WINDOW_KAISER: int
    CUPYNUMERIC_WRAP: int
    CUPYNUMERIC_WRITE: int
    CUPYNUMERIC_ZIP: int

    @abstractmethod
    def cupynumeric_has_cusolvermp(self) -> bool: ...

    @abstractmethod
    def cupynumeric_cusolver_has_geev(self) -> bool: ...

    @abstractmethod
    def cupynumeric_max_eager_volume(self) -> int: ...

    @abstractmethod
    def cupynumeric_register_reduction_ops(
        self, code: int
    ) -> _ReductionOpIds: ...


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))


# Load the cuPyNumeric library first so we have a shard object that
# we can use to initialize all these configuration enumerations
class CuPyNumericLib:
    def __init__(self, name: str) -> None:
        self.name = name

        shared_lib_path = self.get_shared_library()
        assert shared_lib_path is not None
        header = self.get_c_header()
        ffi = cffi.FFI()
        if header is not None:
            ffi.cdef(header)
        # Don't use ffi.dlopen(), because that will call dlclose()
        # automatically when the object gets collected, thus removing
        # symbols that may be needed when destroying C++ objects later
        # (e.g. vtable entries, which will be queried for virtual
        # destructors), causing errors at shutdown.
        shared_lib = dlopen_no_autoclose(ffi, shared_lib_path)
        self.shared_object = cast(_CupynumericSharedLib, shared_lib)

    def register(self) -> None:
        from legate.core import get_legate_runtime

        # We need to make sure that the runtime is started
        get_legate_runtime()

        callback = getattr(
            self.shared_object, "cupynumeric_perform_registration"
        )
        callback()

    def get_shared_library(self) -> str:
        from .install_info import libpath

        return os.path.join(
            libpath, "libcupynumeric" + self.get_library_extension()
        )

    def get_c_header(self) -> str:
        from .install_info import header

        return header

    @staticmethod
    def get_library_extension() -> str:
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"
        raise RuntimeError(f"unknown platform {os_name!r}")


CUPYNUMERIC_LIB_NAME = "cupynumeric"
cupynumeric_lib = CuPyNumericLib(CUPYNUMERIC_LIB_NAME)
cupynumeric_lib.register()
_cupynumeric = cupynumeric_lib.shared_object


# Match these to CuPyNumericOpCode in cupynumeric_c.h
@unique
class CuPyNumericOpCode(IntEnum):
    ADVANCED_INDEXING = _cupynumeric.CUPYNUMERIC_ADVANCED_INDEXING
    ARANGE = _cupynumeric.CUPYNUMERIC_ARANGE
    ARGWHERE = _cupynumeric.CUPYNUMERIC_ARGWHERE
    BATCHED_CHOLESKY = _cupynumeric.CUPYNUMERIC_BATCHED_CHOLESKY
    BINARY_OP = _cupynumeric.CUPYNUMERIC_BINARY_OP
    BINARY_RED = _cupynumeric.CUPYNUMERIC_BINARY_RED
    BINCOUNT = _cupynumeric.CUPYNUMERIC_BINCOUNT
    BITGENERATOR = _cupynumeric.CUPYNUMERIC_BITGENERATOR
    CHOOSE = _cupynumeric.CUPYNUMERIC_CHOOSE
    CONTRACT = _cupynumeric.CUPYNUMERIC_CONTRACT
    CONVERT = _cupynumeric.CUPYNUMERIC_CONVERT
    CONVOLVE = _cupynumeric.CUPYNUMERIC_CONVOLVE
    DIAG = _cupynumeric.CUPYNUMERIC_DIAG
    DOT = _cupynumeric.CUPYNUMERIC_DOT
    EYE = _cupynumeric.CUPYNUMERIC_EYE
    FFT = _cupynumeric.CUPYNUMERIC_FFT
    FILL = _cupynumeric.CUPYNUMERIC_FILL
    FLIP = _cupynumeric.CUPYNUMERIC_FLIP
    GEEV = _cupynumeric.CUPYNUMERIC_GEEV
    GEMM = _cupynumeric.CUPYNUMERIC_GEMM
    HISTOGRAM = _cupynumeric.CUPYNUMERIC_HISTOGRAM
    HISTOGRAMDD = _cupynumeric.CUPYNUMERIC_HISTOGRAMDD
    IN1D = _cupynumeric.CUPYNUMERIC_IN1D
    LOAD_CUDALIBS = _cupynumeric.CUPYNUMERIC_LOAD_CUDALIBS
    MATMUL = _cupynumeric.CUPYNUMERIC_MATMUL
    MATVECMUL = _cupynumeric.CUPYNUMERIC_MATVECMUL
    MP_POTRF = _cupynumeric.CUPYNUMERIC_MP_POTRF
    MP_QR = _cupynumeric.CUPYNUMERIC_MP_QR
    MP_SOLVE = _cupynumeric.CUPYNUMERIC_MP_SOLVE
    NONZERO = _cupynumeric.CUPYNUMERIC_NONZERO
    PACKBITS = _cupynumeric.CUPYNUMERIC_PACKBITS
    PAD = _cupynumeric.CUPYNUMERIC_PAD
    POTRF = _cupynumeric.CUPYNUMERIC_POTRF
    PUTMASK = _cupynumeric.CUPYNUMERIC_PUTMASK
    QR = _cupynumeric.CUPYNUMERIC_QR
    RAND = _cupynumeric.CUPYNUMERIC_RAND
    READ = _cupynumeric.CUPYNUMERIC_READ
    REPEAT = _cupynumeric.CUPYNUMERIC_REPEAT
    SCALAR_UNARY_RED = _cupynumeric.CUPYNUMERIC_SCALAR_UNARY_RED
    SCAN_GLOBAL = _cupynumeric.CUPYNUMERIC_SCAN_GLOBAL
    SCAN_LOCAL = _cupynumeric.CUPYNUMERIC_SCAN_LOCAL
    SEARCHSORTED = _cupynumeric.CUPYNUMERIC_SEARCHSORTED
    SELECT = _cupynumeric.CUPYNUMERIC_SELECT
    SOLVE = _cupynumeric.CUPYNUMERIC_SOLVE
    SORT = _cupynumeric.CUPYNUMERIC_SORT
    SVD = _cupynumeric.CUPYNUMERIC_SVD
    SYRK = _cupynumeric.CUPYNUMERIC_SYRK
    SYEV = _cupynumeric.CUPYNUMERIC_SYEV
    TAKE = _cupynumeric.CUPYNUMERIC_TAKE
    TILE = _cupynumeric.CUPYNUMERIC_TILE
    TRANSPOSE_COPY_2D = _cupynumeric.CUPYNUMERIC_TRANSPOSE_COPY_2D
    TRILU = _cupynumeric.CUPYNUMERIC_TRILU
    TRSM = _cupynumeric.CUPYNUMERIC_TRSM
    UNARY_OP = _cupynumeric.CUPYNUMERIC_UNARY_OP
    UNARY_RED = _cupynumeric.CUPYNUMERIC_UNARY_RED
    UNIQUE = _cupynumeric.CUPYNUMERIC_UNIQUE
    UNIQUE_REDUCE = _cupynumeric.CUPYNUMERIC_UNIQUE_REDUCE
    UNLOAD_CUDALIBS = _cupynumeric.CUPYNUMERIC_UNLOAD_CUDALIBS
    UNPACKBITS = _cupynumeric.CUPYNUMERIC_UNPACKBITS
    WHERE = _cupynumeric.CUPYNUMERIC_WHERE
    WINDOW = _cupynumeric.CUPYNUMERIC_WINDOW
    WRAP = _cupynumeric.CUPYNUMERIC_WRAP
    WRITE = _cupynumeric.CUPYNUMERIC_WRITE
    ZIP = _cupynumeric.CUPYNUMERIC_ZIP


# Match these to CuPyNumericUnaryOpCode in cupynumeric_c.h
@unique
class UnaryOpCode(IntEnum):
    ABSOLUTE = _cupynumeric.CUPYNUMERIC_UOP_ABSOLUTE
    ANGLE = _cupynumeric.CUPYNUMERIC_UOP_ANGLE
    ARCCOS = _cupynumeric.CUPYNUMERIC_UOP_ARCCOS
    ARCCOSH = _cupynumeric.CUPYNUMERIC_UOP_ARCCOSH
    ARCSIN = _cupynumeric.CUPYNUMERIC_UOP_ARCSIN
    ARCSINH = _cupynumeric.CUPYNUMERIC_UOP_ARCSINH
    ARCTAN = _cupynumeric.CUPYNUMERIC_UOP_ARCTAN
    ARCTANH = _cupynumeric.CUPYNUMERIC_UOP_ARCTANH
    CBRT = _cupynumeric.CUPYNUMERIC_UOP_CBRT
    CEIL = _cupynumeric.CUPYNUMERIC_UOP_CEIL
    CLIP = _cupynumeric.CUPYNUMERIC_UOP_CLIP
    CONJ = _cupynumeric.CUPYNUMERIC_UOP_CONJ
    COPY = _cupynumeric.CUPYNUMERIC_UOP_COPY
    COS = _cupynumeric.CUPYNUMERIC_UOP_COS
    COSH = _cupynumeric.CUPYNUMERIC_UOP_COSH
    DEG2RAD = _cupynumeric.CUPYNUMERIC_UOP_DEG2RAD
    EXP = _cupynumeric.CUPYNUMERIC_UOP_EXP
    EXP2 = _cupynumeric.CUPYNUMERIC_UOP_EXP2
    EXPM1 = _cupynumeric.CUPYNUMERIC_UOP_EXPM1
    FLOOR = _cupynumeric.CUPYNUMERIC_UOP_FLOOR
    FREXP = _cupynumeric.CUPYNUMERIC_UOP_FREXP
    GETARG = _cupynumeric.CUPYNUMERIC_UOP_GETARG
    IMAG = _cupynumeric.CUPYNUMERIC_UOP_IMAG
    INVERT = _cupynumeric.CUPYNUMERIC_UOP_INVERT
    ISFINITE = _cupynumeric.CUPYNUMERIC_UOP_ISFINITE
    ISINF = _cupynumeric.CUPYNUMERIC_UOP_ISINF
    ISNAN = _cupynumeric.CUPYNUMERIC_UOP_ISNAN
    LOG = _cupynumeric.CUPYNUMERIC_UOP_LOG
    LOG10 = _cupynumeric.CUPYNUMERIC_UOP_LOG10
    LOG1P = _cupynumeric.CUPYNUMERIC_UOP_LOG1P
    LOG2 = _cupynumeric.CUPYNUMERIC_UOP_LOG2
    LOGICAL_NOT = _cupynumeric.CUPYNUMERIC_UOP_LOGICAL_NOT
    MODF = _cupynumeric.CUPYNUMERIC_UOP_MODF
    NEGATIVE = _cupynumeric.CUPYNUMERIC_UOP_NEGATIVE
    POSITIVE = _cupynumeric.CUPYNUMERIC_UOP_POSITIVE
    RAD2DEG = _cupynumeric.CUPYNUMERIC_UOP_RAD2DEG
    REAL = _cupynumeric.CUPYNUMERIC_UOP_REAL
    RECIPROCAL = _cupynumeric.CUPYNUMERIC_UOP_RECIPROCAL
    RINT = _cupynumeric.CUPYNUMERIC_UOP_RINT
    ROUND = _cupynumeric.CUPYNUMERIC_UOP_ROUND
    SIGN = _cupynumeric.CUPYNUMERIC_UOP_SIGN
    SIGNBIT = _cupynumeric.CUPYNUMERIC_UOP_SIGNBIT
    SIN = _cupynumeric.CUPYNUMERIC_UOP_SIN
    SINH = _cupynumeric.CUPYNUMERIC_UOP_SINH
    SQRT = _cupynumeric.CUPYNUMERIC_UOP_SQRT
    SQUARE = _cupynumeric.CUPYNUMERIC_UOP_SQUARE
    TAN = _cupynumeric.CUPYNUMERIC_UOP_TAN
    TANH = _cupynumeric.CUPYNUMERIC_UOP_TANH
    TRUNC = _cupynumeric.CUPYNUMERIC_UOP_TRUNC


# Match these to CuPyNumericUnaryRedCode in cupynumeric_c.h
@unique
class UnaryRedCode(IntEnum):
    ALL = _cupynumeric.CUPYNUMERIC_RED_ALL
    ANY = _cupynumeric.CUPYNUMERIC_RED_ANY
    ARGMAX = _cupynumeric.CUPYNUMERIC_RED_ARGMAX
    ARGMIN = _cupynumeric.CUPYNUMERIC_RED_ARGMIN
    CONTAINS = _cupynumeric.CUPYNUMERIC_RED_CONTAINS
    COUNT_NONZERO = _cupynumeric.CUPYNUMERIC_RED_COUNT_NONZERO
    MAX = _cupynumeric.CUPYNUMERIC_RED_MAX
    MIN = _cupynumeric.CUPYNUMERIC_RED_MIN
    NANARGMAX = _cupynumeric.CUPYNUMERIC_RED_NANARGMAX
    NANARGMIN = _cupynumeric.CUPYNUMERIC_RED_NANARGMIN
    NANMAX = _cupynumeric.CUPYNUMERIC_RED_NANMAX
    NANMIN = _cupynumeric.CUPYNUMERIC_RED_NANMIN
    NANPROD = _cupynumeric.CUPYNUMERIC_RED_NANPROD
    NANSUM = _cupynumeric.CUPYNUMERIC_RED_NANSUM
    PROD = _cupynumeric.CUPYNUMERIC_RED_PROD
    SUM = _cupynumeric.CUPYNUMERIC_RED_SUM
    SUM_SQUARES = _cupynumeric.CUPYNUMERIC_RED_SUM_SQUARES
    VARIANCE = _cupynumeric.CUPYNUMERIC_RED_VARIANCE


# Match these to CuPyNumericBinaryOpCode in cupynumeric_c.h
@unique
class BinaryOpCode(IntEnum):
    ADD = _cupynumeric.CUPYNUMERIC_BINOP_ADD
    ARCTAN2 = _cupynumeric.CUPYNUMERIC_BINOP_ARCTAN2
    BITWISE_AND = _cupynumeric.CUPYNUMERIC_BINOP_BITWISE_AND
    BITWISE_OR = _cupynumeric.CUPYNUMERIC_BINOP_BITWISE_OR
    BITWISE_XOR = _cupynumeric.CUPYNUMERIC_BINOP_BITWISE_XOR
    COPYSIGN = _cupynumeric.CUPYNUMERIC_BINOP_COPYSIGN
    DIVIDE = _cupynumeric.CUPYNUMERIC_BINOP_DIVIDE
    EQUAL = _cupynumeric.CUPYNUMERIC_BINOP_EQUAL
    FLOAT_POWER = _cupynumeric.CUPYNUMERIC_BINOP_FLOAT_POWER
    FLOOR_DIVIDE = _cupynumeric.CUPYNUMERIC_BINOP_FLOOR_DIVIDE
    FMOD = _cupynumeric.CUPYNUMERIC_BINOP_FMOD
    GCD = _cupynumeric.CUPYNUMERIC_BINOP_GCD
    GREATER = _cupynumeric.CUPYNUMERIC_BINOP_GREATER
    GREATER_EQUAL = _cupynumeric.CUPYNUMERIC_BINOP_GREATER_EQUAL
    HYPOT = _cupynumeric.CUPYNUMERIC_BINOP_HYPOT
    ISCLOSE = _cupynumeric.CUPYNUMERIC_BINOP_ISCLOSE
    LCM = _cupynumeric.CUPYNUMERIC_BINOP_LCM
    LDEXP = _cupynumeric.CUPYNUMERIC_BINOP_LDEXP
    LEFT_SHIFT = _cupynumeric.CUPYNUMERIC_BINOP_LEFT_SHIFT
    LESS = _cupynumeric.CUPYNUMERIC_BINOP_LESS
    LESS_EQUAL = _cupynumeric.CUPYNUMERIC_BINOP_LESS_EQUAL
    LOGADDEXP = _cupynumeric.CUPYNUMERIC_BINOP_LOGADDEXP
    LOGADDEXP2 = _cupynumeric.CUPYNUMERIC_BINOP_LOGADDEXP2
    LOGICAL_AND = _cupynumeric.CUPYNUMERIC_BINOP_LOGICAL_AND
    LOGICAL_OR = _cupynumeric.CUPYNUMERIC_BINOP_LOGICAL_OR
    LOGICAL_XOR = _cupynumeric.CUPYNUMERIC_BINOP_LOGICAL_XOR
    MAXIMUM = _cupynumeric.CUPYNUMERIC_BINOP_MAXIMUM
    MINIMUM = _cupynumeric.CUPYNUMERIC_BINOP_MINIMUM
    MOD = _cupynumeric.CUPYNUMERIC_BINOP_MOD
    MULTIPLY = _cupynumeric.CUPYNUMERIC_BINOP_MULTIPLY
    NEXTAFTER = _cupynumeric.CUPYNUMERIC_BINOP_NEXTAFTER
    NOT_EQUAL = _cupynumeric.CUPYNUMERIC_BINOP_NOT_EQUAL
    POWER = _cupynumeric.CUPYNUMERIC_BINOP_POWER
    RIGHT_SHIFT = _cupynumeric.CUPYNUMERIC_BINOP_RIGHT_SHIFT
    SUBTRACT = _cupynumeric.CUPYNUMERIC_BINOP_SUBTRACT


@unique
class WindowOpCode(IntEnum):
    BARLETT = _cupynumeric.CUPYNUMERIC_WINDOW_BARLETT
    BLACKMAN = _cupynumeric.CUPYNUMERIC_WINDOW_BLACKMAN
    HAMMING = _cupynumeric.CUPYNUMERIC_WINDOW_HAMMING
    HANNING = _cupynumeric.CUPYNUMERIC_WINDOW_HANNING
    KAISER = _cupynumeric.CUPYNUMERIC_WINDOW_KAISER


# Match these to RandGenCode in rand_util.h
@unique
class RandGenCode(IntEnum):
    UNIFORM = 1
    NORMAL = 2
    INTEGER = 3


# Match these to CuPyNumericScanCode in cupynumeric_c.h
@unique
class ScanCode(IntEnum):
    PROD = _cupynumeric.CUPYNUMERIC_SCAN_PROD
    SUM = _cupynumeric.CUPYNUMERIC_SCAN_SUM


# Match these to CuPyNumericConvertCode in cupynumeric_c.h
@unique
class ConvertCode(IntEnum):
    NOOP = _cupynumeric.CUPYNUMERIC_CONVERT_NAN_NOOP
    PROD = _cupynumeric.CUPYNUMERIC_CONVERT_NAN_PROD
    SUM = _cupynumeric.CUPYNUMERIC_CONVERT_NAN_SUM


# Match these to BitGeneratorOperation in cupynumeric_c.h
@unique
class BitGeneratorOperation(IntEnum):
    CREATE = _cupynumeric.CUPYNUMERIC_BITGENOP_CREATE
    DESTROY = _cupynumeric.CUPYNUMERIC_BITGENOP_DESTROY
    RAND_RAW = _cupynumeric.CUPYNUMERIC_BITGENOP_RAND_RAW
    DISTRIBUTION = _cupynumeric.CUPYNUMERIC_BITGENOP_DISTRIBUTION


# Match these to BitGeneratorType in cupynumeric_c.h
@unique
class BitGeneratorType(IntEnum):
    DEFAULT = _cupynumeric.CUPYNUMERIC_BITGENTYPE_DEFAULT
    XORWOW = _cupynumeric.CUPYNUMERIC_BITGENTYPE_XORWOW
    MRG32K3A = _cupynumeric.CUPYNUMERIC_BITGENTYPE_MRG32K3A
    MTGP32 = _cupynumeric.CUPYNUMERIC_BITGENTYPE_MTGP32
    MT19937 = _cupynumeric.CUPYNUMERIC_BITGENTYPE_MT19937
    PHILOX4_32_10 = _cupynumeric.CUPYNUMERIC_BITGENTYPE_PHILOX4_32_10


# Match these to BitGeneratorDistribution in cupynumeric_c.h
@unique
class BitGeneratorDistribution(IntEnum):
    INTEGERS_16 = _cupynumeric.CUPYNUMERIC_BITGENDIST_INTEGERS_16
    INTEGERS_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_INTEGERS_32
    INTEGERS_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_INTEGERS_64
    UNIFORM_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_UNIFORM_32
    UNIFORM_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_UNIFORM_64
    LOGNORMAL_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_LOGNORMAL_32
    LOGNORMAL_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_LOGNORMAL_64
    NORMAL_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_NORMAL_32
    NORMAL_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_NORMAL_64
    POISSON = _cupynumeric.CUPYNUMERIC_BITGENDIST_POISSON
    EXPONENTIAL_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_EXPONENTIAL_32
    EXPONENTIAL_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_EXPONENTIAL_64
    GUMBEL_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_GUMBEL_32
    GUMBEL_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_GUMBEL_64
    LAPLACE_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_LAPLACE_32
    LAPLACE_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_LAPLACE_64
    LOGISTIC_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_LOGISTIC_32
    LOGISTIC_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_LOGISTIC_64
    PARETO_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_PARETO_32
    PARETO_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_PARETO_64
    POWER_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_POWER_32
    POWER_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_POWER_64
    RAYLEIGH_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_RAYLEIGH_32
    RAYLEIGH_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_RAYLEIGH_64
    CAUCHY_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_CAUCHY_32
    CAUCHY_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_CAUCHY_64
    TRIANGULAR_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_TRIANGULAR_32
    TRIANGULAR_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_TRIANGULAR_64
    WEIBULL_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_WEIBULL_32
    WEIBULL_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_WEIBULL_64
    BYTES = _cupynumeric.CUPYNUMERIC_BITGENDIST_BYTES
    BETA_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_BETA_32
    BETA_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_BETA_64
    F_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_F_32
    F_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_F_64
    LOGSERIES = _cupynumeric.CUPYNUMERIC_BITGENDIST_LOGSERIES
    NONCENTRAL_F_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_NONCENTRAL_F_32
    NONCENTRAL_F_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_NONCENTRAL_F_64
    CHISQUARE_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_CHISQUARE_32
    CHISQUARE_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_CHISQUARE_64
    GAMMA_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_GAMMA_32
    GAMMA_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_GAMMA_64
    STANDARD_T_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_STANDARD_T_32
    STANDARD_T_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_STANDARD_T_64
    HYPERGEOMETRIC = _cupynumeric.CUPYNUMERIC_BITGENDIST_HYPERGEOMETRIC
    VONMISES_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_VONMISES_32
    VONMISES_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_VONMISES_64
    ZIPF = _cupynumeric.CUPYNUMERIC_BITGENDIST_ZIPF
    GEOMETRIC = _cupynumeric.CUPYNUMERIC_BITGENDIST_GEOMETRIC
    WALD_32 = _cupynumeric.CUPYNUMERIC_BITGENDIST_WALD_32
    WALD_64 = _cupynumeric.CUPYNUMERIC_BITGENDIST_WALD_64
    BINOMIAL = _cupynumeric.CUPYNUMERIC_BITGENDIST_BINOMIAL
    NEGATIVE_BINOMIAL = _cupynumeric.CUPYNUMERIC_BITGENDIST_NEGATIVE_BINOMIAL


# Match these to CuPyNumericConvolveMethod in cupynumeric_c.h
@unique
class ConvolveMethod(IntEnum):
    AUTO = _cupynumeric.CUPYNUMERIC_CONVOLVE_AUTO
    DIRECT = _cupynumeric.CUPYNUMERIC_CONVOLVE_DIRECT
    FFT = _cupynumeric.CUPYNUMERIC_CONVOLVE_FFT


@unique
class TransferType(IntEnum):
    DONATE = 0
    MAKE_COPY = 1
    SHARE = 2


# Match these to fftType in fft_util.h
class FFTType:
    def __init__(
        self,
        name: str,
        type_id: int,
        input_dtype: npt.DTypeLike,
        output_dtype: npt.DTypeLike,
        single_precision: bool,
        complex_type: FFTType | None = None,
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
    "C2C", _cupynumeric.CUPYNUMERIC_FFT_C2C, np.complex64, np.complex64, True
)

FFT_Z2Z = FFTType(
    "Z2Z",
    _cupynumeric.CUPYNUMERIC_FFT_Z2Z,
    np.complex128,
    np.complex128,
    False,
)

FFT_R2C = FFTType(
    "R2C",
    _cupynumeric.CUPYNUMERIC_FFT_R2C,
    np.float32,
    np.complex64,
    True,
    FFT_C2C,
)

FFT_C2R = FFTType(
    "C2R",
    _cupynumeric.CUPYNUMERIC_FFT_C2R,
    np.complex64,
    np.float32,
    True,
    FFT_C2C,
)

FFT_D2Z = FFTType(
    "D2Z",
    _cupynumeric.CUPYNUMERIC_FFT_D2Z,
    np.float64,
    np.complex128,
    False,
    FFT_Z2Z,
)

FFT_Z2D = FFTType(
    "Z2D",
    _cupynumeric.CUPYNUMERIC_FFT_Z2D,
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
    FORWARD = _cupynumeric.CUPYNUMERIC_FFT_FORWARD
    INVERSE = _cupynumeric.CUPYNUMERIC_FFT_INVERSE


# Match these to CuPyNumericBitorder in cupynumeric_c.h
@unique
class Bitorder(IntEnum):
    BIG = _cupynumeric.CUPYNUMERIC_BITORDER_BIG
    LITTLE = _cupynumeric.CUPYNUMERIC_BITORDER_LITTLE


@unique
class FFTNormalization(IntEnum):
    FORWARD = 1
    INVERSE = 2
    ORTHOGONAL = 3

    @staticmethod
    def from_string(in_string: str) -> FFTNormalization | None:
        if in_string == "forward":
            return FFTNormalization.FORWARD
        elif in_string == "ortho":
            return FFTNormalization.ORTHOGONAL
        elif in_string == "backward" or in_string is None:
            return FFTNormalization.INVERSE
        else:
            raise ValueError(
                f'Invalid norm value {in_string}; should be "backward",'
                '"ortho" or "forward".'
            )

    @staticmethod
    def reverse(in_string: str | None) -> str:
        if in_string == "forward":
            return "backward"
        elif in_string == "backward" or in_string is None:
            return "forward"
        else:
            return in_string
