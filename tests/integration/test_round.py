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

import numpy as np
import pytest
from utils.generators import mk_0to1_array
from utils.utils import MAX_DIM_RANGE

import cupynumeric as num

FLOAT = (np.float32, np.float64)

COMPLEX = (np.complex64, np.complex128)


@pytest.mark.parametrize("decimals", range(-5, 5))
def test_empty_array(decimals):
    res_np = np.round([], decimals=decimals)
    res_num = num.round([], decimals=decimals)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("decimals", range(-3, 3))
@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
def test_basic_float16(ndim, decimals):
    shape = (3,) * ndim
    np_arr = mk_0to1_array(np, shape, dtype=np.float16)
    num_arr = mk_0to1_array(num, shape, dtype=np.float16)

    res_np = np.round(np_arr, decimals=decimals)
    res_num = num.round(num_arr, decimals=decimals)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("decimals", range(-5, 5))
@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
@pytest.mark.parametrize("dtype", FLOAT)
def test_basic_float(dtype, ndim, decimals):
    shape = (3,) * ndim
    np_arr = mk_0to1_array(np, shape, dtype=dtype)
    num_arr = mk_0to1_array(num, shape, dtype=dtype)

    res_np = np.round(np_arr, decimals=decimals)
    res_num = num.round(num_arr, decimals=decimals)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("decimals", range(-5, 5))
@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
@pytest.mark.parametrize("dtype", FLOAT)
def test_randomized_float(dtype, ndim, decimals):
    shape = (3,) * ndim
    values = np.random.uniform(-10, 10, shape) * 10**6
    np_arr = np.array(values, dtype=dtype)
    num_arr = num.array(values, dtype=dtype)

    res_np = np.round(np_arr, decimals=decimals)
    res_num = num.round(num_arr, decimals=decimals)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("decimals", range(-5, 5))
@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
@pytest.mark.parametrize("dtype", COMPLEX)
def test_randomized_complex(dtype, ndim, decimals):
    shape = (1,) * ndim
    values = (
        np.random.uniform(-10, 10, shape) * 10**6
        + 1.0j * np.random.uniform(-10, 10, shape) * 10**6
    )
    np_arr = np.array(values, dtype=dtype)
    num_arr = num.array(values, dtype=dtype)

    res_np = np.round(np_arr, decimals=decimals)
    res_num = num.round(num_arr, decimals=decimals)

    assert np.array_equal(res_np, res_num)


def test_out_np_array():
    array = ((2.1, 3.7, 4.1), (-3.1, -4.7, 5), (6.2, 0, 12.9))
    np_arr = np.array(array)
    num_arr = num.array(array)

    out_np = np.empty(np_arr.shape)
    out_num = np.empty(np_arr.shape)

    np_arr.round(out=out_np)
    num_arr.round(out=out_num)
    assert np.array_equal(out_np, out_num)


def test_out_num_array():
    array = ((2.1, 3.7, 4.1), (-3.1, -4.7, 5), (6.2, 0, 12.9))
    np_arr = np.array(array)
    num_arr = num.array(array)

    out_np = np.empty(np_arr.shape)
    out_num = num.empty(np_arr.shape)

    np_arr.round(out=out_np)
    num_arr.round(out=out_num)
    assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
