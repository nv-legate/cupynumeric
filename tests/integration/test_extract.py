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
from utils.generators import mk_seq_array

import cupynumeric as num

DIM = 5
SIZES = [
    (0,),
    1,
    5,
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

VALUES = [
    [0],
    [42],
    [42 + 3j],
    [11, 12, 13],
    [True, False, False, True],
    [42.3, 42.3, 42.3, 42.3, 42.3],
    [np.inf],
]


@pytest.mark.xfail
def test_none_array():
    res_np = np.extract([0], None)  # return []
    res_num = num.extract(
        [0], None
    )  # AttributeError: 'NoneType' object has no attribute 'size'
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_empty_array():
    res_np = np.extract([0], [])  # return []
    res_num = num.extract(
        [0], []
    )  # ValueError: arr array and condition array must be of same size
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_none_condition():
    a = num.array([1, 2, 3, 4])
    res_np = np.extract(None, a)  # all return []
    res_num = num.extract(
        None, a
    )  # AttributeError: 'NoneType' object has no attribute 'size'
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "con", (-3, 0, 3, False, True, [2], [2, 3], [2, -3, 4], [1, 2, 3, 4, 5])
)
def test_negative_condition(con):
    a = num.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        num.extract(con, a)


@pytest.mark.xfail
def test_complex_condition():
    # when condition is complex type a+bj,
    # if a==0, cuPyNumeric take it as 0, while Numpy take it as 1
    a = np.array([1, 2, 3, 4])
    b = num.array([1, 2, 3, 4])
    condition = [1 + 2j, 2, 2, 5j]
    res_np = np.extract(condition, a)  # array([1, 2, 3, 4])
    res_num = num.extract(condition, b)  # array([1, 2, 3])
    assert np.array_equal(res_np, res_num)


ARR = [
    [
        [[1 + 2j, 54, 4], [4, 3 + 1j, 45]],
        [[5.5, 58.3, 0.6], [9, 0, 4]],
        [[0, 0, 0], [-9, 0, -4]],
    ],
    [[True, False], [True, True], [True, False]],
    [[]],
    [[], []],
    [
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1]],
    ],
]


def array_condition():
    arr_list = []
    for arr in ARR:
        arr_np = np.array(arr)
        condition_np = arr_np.copy()
        arr_list.append((condition_np, arr_np))
        arr_list.append((condition_np.flatten(), arr_np))
        arr_list.append((condition_np, arr_np.flatten()))
        arr_list.append(
            (condition_np.swapaxes(0, condition_np.ndim - 1), arr_np)
        )
        arr_list.append(
            (condition_np, arr_np.swapaxes(0, condition_np.ndim - 1))
        )
    return arr_list


def check_extract(condition_np, arr_np):
    arr_num = num.array(arr_np)
    condition_num = num.array(condition_np)
    result_np2 = arr_np[condition_np.reshape(arr_np.shape).astype(bool)]
    result_num = num.extract(condition_num, arr_num)
    assert np.array_equal(result_np2, result_num)


@pytest.mark.parametrize(
    "con, arr", (data for data in array_condition()), ids=str
)
def test_extract_nonzero1(con, arr):
    check_extract(con, arr)


@pytest.mark.parametrize("shape", SIZES, ids=str)
def test_extract_basic(shape):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    # make sure condition is between 0 and 1
    np_condition = np.array((mk_seq_array(np, shape) % 2).astype(bool))
    num_condition = num.array((mk_seq_array(num, shape) % 2).astype(bool))

    res_np = np.extract(np_condition, np_arr)
    res_num = num.extract(num_condition, num_arr)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("shape", SIZES, ids=str)
@pytest.mark.parametrize("vals", VALUES, ids=str)
def test_place_basic(shape, vals):
    arr_np = mk_seq_array(np, shape)
    arr_num = num.array(mk_seq_array(num, shape))

    mask_np = np.array((mk_seq_array(np, shape) % 2).astype(bool))
    mask_num = num.array((mk_seq_array(np, shape) % 2).astype(bool))

    vals_np = np.array(vals).astype(arr_np.dtype)
    vals_num = num.array(vals_np)

    np.place(arr_np, mask_np, vals_np)
    num.place(arr_num, mask_num, vals_num)

    assert np.array_equal(arr_np, arr_num)


@pytest.mark.xfail(reason="cupynumeric raises exception when vals is ndim")
@pytest.mark.parametrize("vals", VALUES, ids=str)
@pytest.mark.parametrize("ndim", range(2, DIM), ids=str)
def test_place_vals_ndim(vals, ndim):
    shape = (1, 2, 3)

    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    mask_np = (arr_np % 2).astype(bool)
    mask_num = (arr_np % 2).astype(bool)
    vals_np = np.array(vals, ndmin=ndim).astype(arr_np.dtype)
    vals_num = num.array(vals_np)

    # NumPy pass, array([[[2, 2, 2], [2, 2, 2]]])
    np.place(arr_np, mask_np, vals_np)
    # cuPyNumeric raises ValueError: vals array has to be 1-dimensional
    num.place(arr_num, mask_num, vals_num)
    assert np.array_equal(arr_np, arr_num)


@pytest.mark.parametrize("shape", SIZES, ids=str)
@pytest.mark.parametrize("vals", VALUES, ids=str)
def test_place_mask_reshape(shape, vals):
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)

    mask_np = np.arange(0, arr_np.size).astype(bool)
    mask_num = num.arange(0, arr_num.size).astype(bool)

    vals_np = np.array(vals).astype(arr_np.dtype)
    vals_num = num.array(vals_np)

    np.place(arr_np, mask_np, vals_np)
    num.place(arr_num, mask_num, vals_num)

    assert np.array_equal(arr_np, arr_num)


@pytest.mark.parametrize("dtype", (np.float32, np.complex64), ids=str)
def test_place_mask_dtype(dtype):
    shape = (3, 2, 3)
    vals = [42 + 3j]
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)

    mask_np = mk_seq_array(np, shape).astype(dtype)
    mask_num = mk_seq_array(num, shape).astype(dtype)

    vals_np = np.array(vals).astype(arr_np.dtype)
    vals_num = num.array(vals_np)

    np.place(arr_np, mask_np, vals_np)
    num.place(arr_num, mask_num, vals_num)

    assert np.array_equal(arr_np, arr_num)


@pytest.mark.parametrize("shape", SIZES, ids=str)
@pytest.mark.parametrize("vals", VALUES, ids=str)
def test_place_mask_zeros(shape, vals):
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)

    mask_np = np.zeros(shape, dtype=bool)
    mask_num = num.zeros(shape, dtype=bool)

    vals_np = np.array(vals).astype(arr_np.dtype)
    vals_num = num.array(vals).astype(arr_num.dtype)

    np.place(arr_np, mask_np, vals_np)
    num.place(arr_num, mask_num, vals_num)

    assert np.array_equal(arr_np, arr_num)


class TestPlaceErrors:
    def test_arr_mask_size(self):
        expected_exc = ValueError
        arr_shape = (1, 2, 3)
        mask_shape = (2, 2)
        vals = (11, 12, 13)

        arr_np = np.random.random(arr_shape)
        arr_num = num.array(arr_np)
        mask_np = np.random.random(mask_shape)
        mask_num = num.array(mask_np)
        vals_np = np.array(vals, dtype=arr_np.dtype)
        vals_num = num.array(vals, dtype=arr_np.dtype)

        with pytest.raises(expected_exc):
            np.place(arr_np, mask_np, vals_np)
        with pytest.raises(expected_exc):
            num.place(arr_num, mask_num, vals_num)

    def test_vals_empty(self):
        expected_exc = ValueError
        shape = (1, 2, 3)
        val = []

        arr_np = np.array(shape)
        arr_num = num.array(shape)
        mask_np = np.array(shape, dtype=bool)
        mask_num = num.array(shape, dtype=bool)
        vals_np = np.array(val, dtype=arr_np.dtype)
        vals_num = num.array(val, dtype=arr_num.dtype)

        with pytest.raises(expected_exc):
            np.place(arr_np, mask_np, vals_np)
        with pytest.raises(expected_exc):
            num.place(arr_num, mask_num, vals_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
