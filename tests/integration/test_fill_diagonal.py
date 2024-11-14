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
from legate.core import LEGATE_MAX_DIM
from utils.generators import mk_seq_array

import cupynumeric as num

WRAP = [True, False]


@pytest.mark.parametrize("wrap", (None, -100, 0, 100, "hi", [2, 3]))
def test_wrap(wrap):
    shape = (3, 3, 3)
    val = 10
    np_array = mk_seq_array(np, shape)
    num_array = num.array(np_array)
    np.fill_diagonal(np_array, val, wrap)
    num.fill_diagonal(num_array, val, wrap)
    assert np.array_equal(np_array, num_array)


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("val_shape", ((0,), (3,), (6,), (2, 2), (2, 2, 6)))
@pytest.mark.parametrize("wrap", WRAP, ids=str)
def test_basic(ndim, val_shape, wrap):
    shape = (5,) * ndim
    np_array = mk_seq_array(np, shape)
    num_array = num.array(np_array)

    np_values = mk_seq_array(np, val_shape) * 100
    num_values = num.array(np_values)

    np.fill_diagonal(np_array, np_values, wrap)
    num.fill_diagonal(num_array, num_values, wrap)

    assert np.array_equal(np_array, num_array)


SHAPES = [
    (20, 10),
    (100, 2),
    (55, 11),
    (3, 0),
    (
        1,
        1,
    ),
]
VALUE_SHAPES = [
    (10, 10, 10),
    (5, 5),
    (5,),
    (9,),
    (
        0,
        3,
    ),
    (0,),
]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("vshape", VALUE_SHAPES, ids=str)
@pytest.mark.parametrize("wrap", WRAP, ids=str)
def test_tall_matrices(shape, vshape, wrap):
    a_np = np.ones(shape)
    v_np = np.full(vshape, 100)
    a_num = num.array(a_np)
    v_num = num.array(v_np)
    np.fill_diagonal(a_np, v_np, wrap)
    num.fill_diagonal(a_num, v_num, wrap)
    assert np.array_equal(a_np, a_num)


class TestFillDiagonalErrors:
    def test_dimension_mismatch(self):
        expected_exc = ValueError
        arr = np.empty((1, 2, 3))
        with pytest.raises(expected_exc):
            np.fill_diagonal(arr, 5)
        with pytest.raises(expected_exc):
            num.fill_diagonal(arr, 5)

    @pytest.mark.parametrize("arr", (-3, [0], (5)))
    def test_arr_invalid(self, arr):
        arr_np = np.array(arr)
        arr_num = num.array(arr)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.fill_diagonal(arr_np, 10)
        with pytest.raises(expected_exc):
            num.fill_diagonal(arr_num, 10)

    @pytest.mark.xfail
    def test_val_none(self):
        shape = (3, 3, 3)
        val = None
        np_array = mk_seq_array(np, shape)
        num_array = num.array(np_array)

        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.fill_diagonal(np_array, val)
        # Numpy raises TypeError: int() argument must be a string,
        # a bytes-like object or a real number, not 'NoneType'
        with pytest.raises(expected_exc):
            num.fill_diagonal(num_array, val)
        # cuPyNumeric raises AttributeError:
        # 'NoneType' object has no attribute 'size'


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
