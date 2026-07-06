# Copyright 2026 NVIDIA Corporation
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

import sys

import numpy as np
import pytest

from _benchmark.format_dtype import format_dtype


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int32, "int32"),
        (np.int64, "int64"),
        (np.uint8, "uint8"),
        (np.bool_, "bool"),
        (np.complex64, "complex64"),
        (np.complex128, "complex128"),
    ],
)
def test_format_dtype_numpy_scalar_types(dtype: type, expected: str) -> None:
    assert format_dtype(dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("float32", "float32"),
        ("float64", "float64"),
        ("int32", "int32"),
        ("uint16", "uint16"),
        ("complex128", "complex128"),
        ("bool", "bool"),
    ],
)
def test_format_dtype_string_aliases(dtype: str, expected: str) -> None:
    assert format_dtype(dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.dtype("float32"), "float32"),
        (np.dtype("int64"), "int64"),
        (np.dtype(np.complex64), "complex64"),
    ],
)
def test_format_dtype_dtype_instance(dtype: np.dtype, expected: str) -> None:
    assert format_dtype(dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(int, np.dtype(int).name), (float, np.dtype(float).name)],
)
def test_format_dtype_python_builtins(dtype: type, expected: str) -> None:
    assert format_dtype(dtype) == expected


def test_format_dtype_rejects_invalid_dtype() -> None:
    with pytest.raises(TypeError):
        format_dtype("not-a-real-dtype")


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
