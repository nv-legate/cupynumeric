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

import numpy as np
import pytest
from utils.comparisons import allclose
from utils.generators import mk_seq_array

import cupynumeric as num


@pytest.mark.parametrize("value", [5, 5.0, 5.0 + 0j, 5.0 + 1e-15j])
def test_scalar(value):
    result_np = np.real_if_close(value)
    result_num = num.real_if_close(value)
    assert allclose(result_np, result_num)


def test_0d_array():
    scalar = 1.0 + 1e-15j
    x_np_0d = np.array(scalar)
    x_num_0d = num.array(x_np_0d)

    result_np = np.real_if_close(x_np_0d)
    result_num = num.real_if_close(x_num_0d)
    assert allclose(result_np, result_num)


def test_integer_input():
    shape = (1000,)
    x_np = mk_seq_array(np, shape)
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)
    assert allclose(result_np, result_num)
    assert result_np.dtype == result_num.dtype


def test_empty_arrays():
    x_np = np.array([])
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize(
    "array",
    [
        [1.0 + 0j, 2.0 + 0j, 3.0 + 0j],
        [1e10 + 0j, 1e15 + 0j, 1e20 + 0j],
        [2.1 + 1e-10j, 5.2 + 1e-9j],
        [2.1 + 1e-3j, 5.2 + 1e-4j],
        [1.0 + 0j, 2.0 + 1e-15j, 3.0 + 1e-10j, 4.0 + 0j],
    ],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_imaginary(array, dtype):
    x_np = np.array(array, dtype=dtype)
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)
    assert allclose(result_np, result_num)
    assert result_np.dtype == result_num.dtype


@pytest.mark.parametrize("tol", [1e-10, 1e-15, 1e-20])
def test_small_absolute_tolerances(tol):
    x_np = np.array([2.1 + 1e-10j, 5.2 + 5e-11j])
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np, tol=tol)
    result_num = num.real_if_close(x_num, tol=tol)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("tol", [-1, -0.5, 0, 1, 10, 1000, 0.5, 1.0])
def test_various_tolerances(tol):
    x_np = np.array([2.1 + 4e-14j, 5.2 + 3e-15j])
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np, tol=tol)
    result_num = num.real_if_close(x_num, tol=tol)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize(
    "array",
    [
        [1.0 + 100 * np.finfo(np.float64).eps * 1j],
        [1.0 + 101 * np.finfo(np.float64).eps * 1j],
    ],
)
def test_edge_cases_tolerance(array):
    x_np = np.array(array)
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize(
    "dtype",
    [
        np.complex128,
        np.float64,
        np.int32,
        np.float32,
        np.complex64,
        np.int64,
        np.bool_,
    ],
)
def test_various_dtypes(dtype):
    shape = (1000,)
    x_np = mk_seq_array(np, shape).astype(dtype)
    x_num = mk_seq_array(num, shape).astype(dtype)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "array",
    [
        [np.nan + 0j, 1.0 + 0j],
        [1.0 + np.nan * 1j, 2.0 + 0j],
        [np.inf + 0j, 1.0 + 0j],
        [1.0 + np.inf * 1j, 2.0 + 0j],
    ],
)
def test_nan_inf_values(array, dtype):
    x_np = np.array(array, dtype=dtype)
    x_num = num.array(x_np)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)

    # Use numpy's allclose directly to handle NaN comparisons
    # (avoids cuPyNumeric's isclose limitation with equal_nan)
    assert np.allclose(result_np, np.array(result_num), equal_nan=True)


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (1, 1, 1, 5)])
def test_multidimensional_arrays(shape):
    eps = np.finfo(float).eps
    x_np = mk_seq_array(np, shape) + eps * 1j * mk_seq_array(np, shape)
    x_num = mk_seq_array(num, shape) + eps * 1j * mk_seq_array(num, shape)

    result_np = np.real_if_close(x_np)
    result_num = num.real_if_close(x_num)
    assert allclose(result_np, result_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
