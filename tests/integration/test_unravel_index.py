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


class TestUnravelIndexErrors:
    def test_none_array(self):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.unravel_index(None, (3,))
        with pytest.raises(expected_exc):
            num.unravel_index(None, (3,))

    def test_indices_wrong_type(self):
        indices = [
            1.0,
            2.0,
        ]
        shape = (
            3,
            3,
        )
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            num.unravel_index(indices, shape)

    def test_invalid_shape(self):
        # Test invalid shape specification (non-integer shape)
        index = 5
        shape = (3, "3")
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            num.unravel_index(index, shape)

    def test_invalid_index(self):
        # Test an invalid index value (negative index)
        index = -1
        shape = (3, 3)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.unravel_index(index, shape)

    def test_flat_index_out_of_bounds(self):
        # Test an index out of bounds
        index = 10
        shape = (2, 2)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.unravel_index(index, shape)

    def test_empty_shape(self):
        index = 1
        shape = (0,)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.unravel_index(index, shape)

    def test_empty_indices(self):
        # Test an empty indices array
        index = []
        shape = (3, 3)
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            num.unravel_index(index, shape)

    def test_wrong_order(self):
        # Test an empty indices array
        index = [1, 2]
        shape = (3, 3)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.unravel_index(index, shape, "K")


def test_empty_indices():
    # Test an empty indices array
    np_arr = mk_seq_array(np, 0)
    num_arr = mk_seq_array(num, 0)
    shape = (3, 3)
    res_num = num.unravel_index(num_arr, shape)
    res_np = np.unravel_index(np_arr, shape)
    assert np.array_equal(res_num, res_np)


def test_large_shape():
    # Test a large shape
    index = 123
    shape = (
        100,
        100,
        100,
    )
    res_np = np.unravel_index(index, shape)
    res_num = num.unravel_index(index, shape)
    assert np.array_equal(res_num, res_np)


def test_large_index():
    # Test large index values
    index = 1000000000
    shape = (1000000001,)
    res_np = np.unravel_index(index, shape)
    res_num = num.unravel_index(index, shape)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize(
    "order",
    (
        "F",
        "C",
    ),
)
def test_basic(ndim, order):
    shape = (6,) * ndim
    size = (6**ndim) % 2
    np_arr = mk_seq_array(np, size)
    num_arr = mk_seq_array(num, size)

    res_np = np.unravel_index(np_arr, shape, order)
    res_num = num.unravel_index(num_arr, shape, order)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize(
    "order",
    (
        "F",
        "C",
    ),
)
def test_uneven_shape(ndim, order):
    shape = np.random.randint(1, 6, ndim, dtype=int)
    size = ndim
    np_arr = mk_seq_array(np, size)
    num_arr = mk_seq_array(num, size)

    res_np = np.unravel_index(np_arr, shape, order)
    res_num = num.unravel_index(num_arr, shape, order)
    assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
