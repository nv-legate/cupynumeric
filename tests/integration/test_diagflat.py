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

import numpy as np
import pytest
from utils.comparisons import allclose
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


def test_2d_array():
    shape = (100, 100)
    x = mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)


def test_1d_array_with_offset():
    shape = (1000,)
    x = mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape)

    expected = np.diagflat(x, k=1)
    result = num.diagflat(x_num, k=1)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128],
)
def test_dtypes(dtype):
    shape = (1000,)
    x = mk_seq_array(np, shape).astype(dtype)
    x_num = mk_seq_array(num, shape).astype(dtype)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)
    assert result.dtype == expected.dtype


def test_empty_array():
    x = np.array([])
    x_num = num.array(x)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)
    assert result.shape == expected.shape


def test_scalar():
    x = np.array(5)
    x_num = num.array(x)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)


def test_3d_array():
    shape = (10, 10, 10)
    x = mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)


@pytest.mark.parametrize("k", [-5, -2, -1, 0, 1, 2, 5])
def test_various_offsets(k):
    shape = (1000,)
    x = mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape)

    expected = np.diagflat(x, k=k)
    result = num.diagflat(x_num, k=k)
    assert allclose(result, expected)
    assert result.shape == expected.shape


def test_large_array():
    shape = (10000,)
    x = mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)


def test_boolean_array():
    shape = (1000,)
    x = (mk_seq_array(np, shape) % 2).astype(bool)
    x_num = (mk_seq_array(num, shape) % 2).astype(bool)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)
    assert result.dtype == expected.dtype


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_big_arrays(ndim):
    shape = (3,) * ndim
    x = mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape)
    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)
    assert result.shape == expected.shape


def test_non_contiguous_array():
    x = np.arange(10)[::2]
    x_num = num.array(x)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected)


def test_special_values():
    x = np.array([np.nan, np.inf, -np.inf])
    x_num = num.array(x)

    expected = np.diagflat(x)
    result = num.diagflat(x_num)
    assert allclose(result, expected, equal_nan=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
