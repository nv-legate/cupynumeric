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
from utils.utils import AxisError

import cupynumeric as num
from cupynumeric._utils import is_np2_1

# cupynumeric.count_nonzero(a: ndarray,
# axis: Optional[Union[int, tuple[int, ...]]] = None) → Union[int, ndarray]
# cupynumeric.nonzero(a: ndarray) → tuple[cupynumeric.array.ndarray, ...]
# cupynumeric.flatnonzero(a: ndarray) → ndarray

DIM = 5
EMPTY_SIZES = [(0,), (0, 1), (1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)]

NO_EMPTY_SIZE = [
    (1),
    (DIM),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

SIZES = NO_EMPTY_SIZE + EMPTY_SIZES


@pytest.mark.skipif(not is_np2_1, reason="numpy 1.0 does not raise")
@pytest.mark.parametrize("value", (0, 1, 2, 7))
def test_0d_error(value):
    with pytest.raises(ValueError):
        num.nonzero(value)


@pytest.mark.parametrize("size", EMPTY_SIZES)
def test_empty(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.count_nonzero(arr_np)
    res_num = num.count_nonzero(arr_num)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("arr", ([], [[], []], [[[], []], [[], []]]))
def test_empty_arr(arr):
    res_np = np.count_nonzero(arr)
    res_num = num.count_nonzero(arr)
    assert np.array_equal(res_np, res_num)


def assert_equal(numarr, nparr):
    for resultnp, resultnum in zip(nparr, numarr):
        assert np.array_equal(resultnp, resultnum)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.count_nonzero(arr_np)
    res_num = num.count_nonzero(arr_num)
    np.array_equal(res_np, res_num)


def test_axis_out_bound():
    arr = [-1, 0, 1, 2, 10]
    with pytest.raises(AxisError):
        num.count_nonzero(arr, axis=2)


@pytest.mark.xfail
@pytest.mark.parametrize("axis", ((-1, 1), (0, 1), (1, 2), (0, 2)))
def test_axis_tuple(axis):
    size = (5, 5, 5)
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    out_np = np.count_nonzero(arr_np, axis=axis)
    # Numpy passed all axis values
    out_num = num.count_nonzero(arr_num, axis=axis)
    # cuPyNumeric raises 'RuntimeError
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic_axis(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim + 1, ndim, 1):
        out_np = np.count_nonzero(arr_np, axis=axis)
        out_num = num.count_nonzero(arr_num, axis=axis)
        assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", EMPTY_SIZES)
def test_empty_axis(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim + 1, ndim, 1):
        out_np = np.count_nonzero(arr_np, axis=axis)
        out_num = num.count_nonzero(arr_num, axis=axis)
        assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_keepdims(size, keepdims):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim + 1, ndim, 1):
        out_np = np.count_nonzero(arr_np, axis=axis, keepdims=keepdims)
        out_num = num.count_nonzero(arr_num, axis=axis, keepdims=keepdims)
        assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", SIZES)
def test_nonzero(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.nonzero(arr_np)
    res_num = num.nonzero(arr_num)
    np.array_equal(res_np, res_num)


@pytest.mark.parametrize("size", SIZES)
def test_flatnonzero(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.flatnonzero(arr_np)
    res_num = num.flatnonzero(arr_num)
    np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
