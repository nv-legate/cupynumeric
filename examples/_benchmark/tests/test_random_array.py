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

from _benchmark.random_array import random_array


# ---------------------------------------------------------------------------
# Integer / unsigned dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16"]
)
def test_random_array_integer_dtype_preserved(dtype: str) -> None:
    arr = random_array(np, 100, dtype=dtype)
    assert arr.dtype == np.dtype(dtype)


def test_random_array_integer_default_range_is_binary() -> None:
    # shift=0, scale=1: lo=0, hi+1=2, so values are in {0, 1}.
    arr = random_array(np, 1000, dtype="int32")
    assert arr.min() >= 0
    assert arr.max() <= 1


def test_random_array_integer_scale_extends_upper_bound() -> None:
    # shift=0, scale=10: lo=0, hi=10, hi+1=11 -> values in {0..10}.
    arr = random_array(np, 1000, dtype="int32", scale=10)
    assert arr.min() >= 0
    assert arr.max() <= 10


def test_random_array_integer_shift_translates_range() -> None:
    # shift=-5, scale=2: lo=-(-5)*2=10, hi=(1-(-5))*2=12 -> values in {10..12}.
    arr = random_array(np, 1000, dtype="int32", shift=-5, scale=2)
    assert arr.min() >= 10
    assert arr.max() <= 12


def test_random_array_unsigned_bounded() -> None:
    arr = random_array(np, 1000, dtype="uint16", scale=255)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_random_array_integer_int_shape_gives_1d() -> None:
    arr = random_array(np, 50, dtype="int32")
    assert arr.shape == (50,)


def test_random_array_integer_tuple_shape_gives_nd() -> None:
    arr = random_array(np, (3, 4, 5), dtype="int32")
    assert arr.shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# Bool dtype (kind "b") follows the integer code path
# ---------------------------------------------------------------------------


def test_random_array_bool_dtype_preserved() -> None:
    arr = random_array(np, 100, dtype="bool")
    assert arr.dtype == np.dtype("bool")
    assert arr.shape == (100,)


# ---------------------------------------------------------------------------
# Float dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_random_array_float_dtype_preserved(dtype: str) -> None:
    arr = random_array(np, 100, dtype=dtype)
    assert arr.dtype == np.dtype(dtype)


def test_random_array_float_default_range_is_unit_interval() -> None:
    arr = random_array(np, 1000, dtype="float64")
    assert arr.min() >= 0.0
    assert arr.max() < 1.0


def test_random_array_float_shift_only() -> None:
    # r in [0, 1) -> r + shift -> [shift, 1 + shift).
    arr = random_array(np, 1000, dtype="float64", shift=10.0)
    assert arr.min() >= 10.0
    assert arr.max() < 11.0


def test_random_array_float_scale_only() -> None:
    # r in [0, 1) -> r * scale -> [0, scale).
    arr = random_array(np, 1000, dtype="float64", scale=5.0)
    assert arr.min() >= 0.0
    assert arr.max() < 5.0


def test_random_array_float_shift_then_scale() -> None:
    # shift is applied before scale, so r in [shift*scale, (1+shift)*scale).
    arr = random_array(np, 1000, dtype="float64", shift=2.0, scale=3.0)
    assert arr.min() >= 6.0
    assert arr.max() < 9.0


def test_random_array_float_tuple_shape() -> None:
    arr = random_array(np, (4, 5), dtype="float32")
    assert arr.shape == (4, 5)


# ---------------------------------------------------------------------------
# Complex dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_random_array_complex_dtype_preserved(dtype: str) -> None:
    arr = random_array(np, 100, dtype=dtype)
    assert arr.dtype == np.dtype(dtype)


def test_random_array_complex_default_has_zero_imaginary_part() -> None:
    # The implementation samples real values and casts to complex, so the
    # imaginary part should be exactly zero before shift/scale are applied.
    arr = random_array(np, 1000, dtype="complex128")
    assert (arr.imag == 0).all()
    assert arr.real.min() >= 0.0
    assert arr.real.max() < 1.0


def test_random_array_complex_shift_and_scale_affect_real_part() -> None:
    arr = random_array(np, 1000, dtype="complex128", shift=2.0, scale=3.0)
    assert (arr.imag == 0).all()
    assert arr.real.min() >= 6.0
    assert arr.real.max() < 9.0


def test_random_array_complex_tuple_shape() -> None:
    arr = random_array(np, (2, 3), dtype="complex64")
    assert arr.shape == (2, 3)


# ---------------------------------------------------------------------------
# Unsupported dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["datetime64[D]", "O", "U10"])
def test_random_array_unsupported_dtype_raises(dtype: str) -> None:
    with pytest.raises(RuntimeError, match="Unsupported dtype"):
        random_array(np, 10, dtype=dtype)


# ---------------------------------------------------------------------------
# Default dtype
# ---------------------------------------------------------------------------


def test_random_array_default_dtype_is_float64() -> None:
    arr = random_array(np, 10)
    assert arr.dtype == np.dtype("float64")


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
