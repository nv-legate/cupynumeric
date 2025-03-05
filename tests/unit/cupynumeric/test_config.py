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

import pytest

import cupynumeric.config as m  # module under test


class TestCuPyNumericLib:
    def test___init__(self) -> None:
        lib = m.CuPyNumericLib("foo")
        assert lib.name == "foo"

    def test_get_shared_library(self) -> None:
        lib = m.CuPyNumericLib("foo")
        result = lib.get_shared_library()
        assert isinstance(result, str)

        from cupynumeric.install_info import libpath

        assert result.startswith(libpath)

        assert "libcupynumeric" in result

        assert result.endswith(lib.get_library_extension())

    def test_get_c_header(self) -> None:
        lib = m.CuPyNumericLib("foo")

        from cupynumeric.install_info import header

        assert lib.get_c_header() == header


def test_CUPYNUMERIC_LIB_NAME() -> None:
    assert m.CUPYNUMERIC_LIB_NAME == "cupynumeric"


def test_cupynumeric_lib() -> None:
    assert isinstance(m.cupynumeric_lib, m.CuPyNumericLib)


def test_CuPyNumericOpCode() -> None:
    assert set(m.CuPyNumericOpCode.__members__) == {
        "ADVANCED_INDEXING",
        "ARANGE",
        "ARGWHERE",
        "BATCHED_CHOLESKY",
        "BINARY_OP",
        "BINARY_RED",
        "BINCOUNT",
        "BITGENERATOR",
        "CHOOSE",
        "CONTRACT",
        "CONVERT",
        "CONVOLVE",
        "DIAG",
        "DOT",
        "EYE",
        "FFT",
        "FILL",
        "FLIP",
        "GEEV",
        "GEMM",
        "HISTOGRAM",
        "LOAD_CUDALIBS",
        "MATMUL",
        "MATVECMUL",
        "MP_POTRF",
        "MP_SOLVE",
        "NONZERO",
        "PACKBITS",
        "POTRF",
        "PUTMASK",
        "QR",
        "RAND",
        "READ",
        "REPEAT",
        "SELECT",
        "SCALAR_UNARY_RED",
        "SCAN_GLOBAL",
        "SCAN_LOCAL",
        "SOLVE",
        "SORT",
        "SEARCHSORTED",
        "SVD",
        "SYRK",
        "TILE",
        "TRANSPOSE_COPY_2D",
        "TRILU",
        "TRSM",
        "UNARY_OP",
        "UNARY_RED",
        "UNIQUE",
        "UNIQUE_REDUCE",
        "UNLOAD_CUDALIBS",
        "UNPACKBITS",
        "WHERE",
        "WINDOW",
        "WRAP",
        "WRITE",
        "ZIP",
    }


def test_UnaryOpCode() -> None:
    assert (set(m.UnaryOpCode.__members__)) == {
        "ABSOLUTE",
        "ANGLE",
        "ARCCOS",
        "ARCCOSH",
        "ARCSIN",
        "ARCSINH",
        "ARCTAN",
        "ARCTANH",
        "CBRT",
        "CEIL",
        "CLIP",
        "CONJ",
        "COPY",
        "COS",
        "COSH",
        "DEG2RAD",
        "EXP",
        "EXP2",
        "EXPM1",
        "FLOOR",
        "FREXP",
        "GETARG",
        "IMAG",
        "INVERT",
        "ISFINITE",
        "ISINF",
        "ISNAN",
        "LOG",
        "LOG10",
        "LOG1P",
        "LOG2",
        "LOGICAL_NOT",
        "MODF",
        "NEGATIVE",
        "POSITIVE",
        "RAD2DEG",
        "REAL",
        "RECIPROCAL",
        "RINT",
        "ROUND",
        "SIGN",
        "SIGNBIT",
        "SIN",
        "SINH",
        "SQRT",
        "SQUARE",
        "TAN",
        "TANH",
        "TRUNC",
    }


def test_RandGenCode() -> None:
    assert (set(m.RandGenCode.__members__)) == {"UNIFORM", "NORMAL", "INTEGER"}


def test_ScanCode() -> None:
    assert (set(m.ScanCode.__members__)) == {"PROD", "SUM"}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
