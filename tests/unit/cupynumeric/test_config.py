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

import os
import sys
from unittest import mock

import pytest

import cupynumeric.config as m  # module under test
from cupynumeric.install_info import get_libpath


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

    def test_get_libpath_return(self) -> None:
        def fake_exists(path):
            return "cupynumeric/lib" in path or "cupynumeric/lib64" in path

        with mock.patch("os.path.exists", side_effect=fake_exists):
            result = get_libpath()
            assert "cupynumeric/lib" in result or "cupynumeric/lib64" in result

    def test_get_libpath_return2(self):
        def fake_exists(path):
            return (
                "lib" in path
                and "cupynumeric/lib" not in path
                and "build/lib" not in path
            )

        with mock.patch("os.path.exists", side_effect=fake_exists):
            result = get_libpath()
            assert result.endswith("lib") or result.endswith("lib64")

    def test_get_libpath_return3(self):
        def fake_exists(path):
            expected = os.path.join(
                os.path.dirname(os.path.dirname(sys.executable)), "lib"
            )
            expected64 = os.path.join(
                os.path.dirname(os.path.dirname(sys.executable)), "lib64"
            )
            return expected in path or expected64 in path

        with mock.patch("os.path.exists", side_effect=fake_exists):
            result = get_libpath()
            assert result.endswith("lib") or result.endswith("lib64")

    def test_get_libpath_all_fail(self):
        with mock.patch("os.path.exists", return_value=False):
            assert get_libpath() == ""

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
        "HISTOGRAMDD",
        "IN1D",
        "LOAD_CUDALIBS",
        "MATMUL",
        "MATVECMUL",
        "MP_POTRF",
        "MP_QR",
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
        "SYEV",
        "SYRK",
        "TAKE",
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
    sys.exit(pytest.main(sys.argv))
