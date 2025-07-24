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
from utils.generators import mk_0to1_array, mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


def test_scalar():
    x = mk_seq_array(np, (3,))
    x_num = mk_seq_array(num, (3,))
    values = np.zeros((), dtype=np.int32)
    values_num = num.zeros((), dtype=np.int32)
    mask = (x % 2).astype(bool)
    mask_num = num.array(mask)
    np.putmask(x[:1], mask[2:], values)
    num.putmask(x_num[:1], mask_num[2:], values_num)
    assert np.array_equal(x_num, x)

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    mask = (x % 2).astype(bool)
    mask_num = num.array(mask)
    np.putmask(x, mask, 100)
    num.putmask(x_num, mask_num, 100)
    assert np.array_equal(x_num, x)

    x = np.zeros((), dtype=np.int32)
    x_num = num.zeros((), dtype=np.int32)
    mask = False
    mask_num = False
    np.putmask(x, mask, -1)
    num.putmask(x_num, mask_num, -1)
    assert np.array_equal(x_num, x)

    x = np.zeros((), dtype=np.int32)
    x_num = num.zeros((), dtype=np.int32)
    mask = True
    mask_num = True
    np.putmask(x, mask, -1)
    num.putmask(x_num, mask_num, -1)
    assert np.array_equal(x_num, x)

    # testing the case when indices is a scalar
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    mask = (x % 2).astype(bool)
    mask_num = num.array(mask)
    np.putmask(x, mask, values[:1])
    num.putmask(x_num, mask_num, values_num[:1])
    assert np.array_equal(x_num, x)

    # the case when every input is a scalar
    x = num.random.rand(3, 3)
    s = x.sum()
    num.putmask(s, True, 1.0)
    assert s == 1.0


def test_type_convert():
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    mask = x % 2
    mask_num = x_num % 2
    np.putmask(x, mask, values)
    num.putmask(x_num, mask_num, values_num)
    assert np.array_equal(x_num, x)

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    mask = np.zeros((3, 4, 5))
    mask_num = num.zeros((3, 4, 5))
    np.putmask(x, mask, values)
    num.putmask(x_num, mask_num, values_num)
    assert np.array_equal(x_num, x)

    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    x = x.astype(np.int32)
    x_num = x_num.astype(np.int32)
    mask = np.zeros((3, 4, 5))
    mask_num = num.zeros((3, 4, 5))
    np.putmask(x, mask, 11)
    num.putmask(x_num, mask_num, 11)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_ndim(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_mask = (np_arr % 2).astype(bool)
    num_mask = (num_arr % 2).astype(bool)
    # scalar_val
    np.putmask(np_arr, np_mask, -10)
    num.putmask(num_arr, num_mask, -10)
    assert np.array_equal(np_arr, num_arr)

    # val is the same shape:
    np_val = np_arr * 10
    num_val = num_arr * 10
    np.putmask(np_arr, np_mask, np_val)
    num.putmask(num_arr, num_mask, num_val)
    assert np.array_equal(np_arr, num_arr)

    # val is different shape, but the same size
    shape_val = (np_arr.size,)
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10
    np.putmask(np_arr, np_mask, np_values)
    num.putmask(num_arr, num_mask, num_values)
    assert np.array_equal(np_arr, num_arr)

    # val is different shape and smaller size for vals and array
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10
    np.putmask(np_arr, np_mask, np_values)
    num.putmask(num_arr, num_mask, num_values)
    assert np.array_equal(np_arr, num_arr)

    # val is different shape and bigger size for vals and array
    shape_val = (10,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10
    np.putmask(np_arr, np_mask, np_values)
    num.putmask(num_arr, num_mask, num_values)
    assert np.array_equal(np_arr, num_arr)


@pytest.mark.parametrize(
    "shape_val",
    (
        (1,),
        (4,),
        (5,),
        (1, 4),
        (2, 3),
        pytest.param((2, 3, 4), marks=pytest.mark.xfail),
        (3, 4, 5),
    ),
    ids=lambda shape_val: f"(shape_val={shape_val})",
)
def test_a_values_different_shapes(shape_val):
    # for (2, 3, 4),
    # In Numpy, pass
    # In cuPyNumeric, it raises ValueError
    shape_arr = (3, 4)
    np_arr = mk_seq_array(np, shape_arr)
    num_arr = mk_seq_array(num, shape_arr)
    np_mask = (np_arr % 2).astype(bool)
    num_mask = (num_arr % 2).astype(bool)
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10
    np.putmask(np_arr, np_mask, np_values)
    num.putmask(num_arr, num_mask, num_values)
    assert np.array_equal(np_arr, num_arr)


def test_empty_array():
    np_arr = np.array([])
    num_arr = num.array([])
    np_mask = np.array([])
    num_mask = num.array([])
    value = -1
    np.putmask(np_arr, np_mask, value)
    num.putmask(num_arr, num_mask, value)
    assert np.array_equal(np_arr, num_arr)


class TestPutmaskErrors:
    def test_invalid_mask_shape(self):
        expected_exc = ValueError
        shape_arr = (3, 4, 5)
        np_arr = mk_seq_array(np, shape_arr)
        num_arr = mk_seq_array(num, shape_arr)
        shape_mask = (3, 4, 1)
        np_mask = np.zeros(shape_mask)
        num_mask = num.zeros(shape_mask)
        value = -1
        with pytest.raises(expected_exc):
            np.putmask(np_arr, np_mask, value)
        with pytest.raises(expected_exc):
            num.putmask(num_arr, num_mask, value)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "dtype_val",
        (float, complex),
        ids=lambda dtype_val: f"(dtype_val={dtype_val})",
    )
    def test_a_values_different_dtype(self, dtype_val):
        # for both cases,
        # In Numpy, it raises TypeError
        # In cuPyNumeric, it pass
        expected_exc = TypeError
        shape = (3, 4)
        dtype_arr = int
        np_arr = mk_0to1_array(np, shape, dtype=dtype_arr)
        num_arr = mk_0to1_array(num, shape, dtype=dtype_arr)
        np_mask = (np_arr % 2).astype(bool)
        num_mask = (num_arr % 2).astype(bool)
        np_values = mk_0to1_array(np, shape, dtype=dtype_val) * 10
        num_values = mk_0to1_array(num, shape, dtype=dtype_val) * 10
        with pytest.raises(expected_exc):
            np.putmask(np_arr, np_mask, np_values)
        with pytest.raises(expected_exc):
            num.putmask(num_arr, num_mask, num_values)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
