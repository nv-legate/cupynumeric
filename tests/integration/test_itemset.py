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
from utils.generators import generate_item

import cupynumeric as num
from cupynumeric._utils import is_np2

# itemset was removed in numpy 2.0, skip the entire module
if is_np2:
    pytestmark = pytest.mark.skip


@pytest.mark.xfail
def test_no_itemset():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = ValueError
    with pytest.raises(expected_exc):
        arr_num.itemset()
        # Numpy raises ValueError: itemset must have
        # at least one argument
    with pytest.raises(expected_exc):
        arr_np.itemset()
        # cuPyNumeric raises KeyError: 'itemset() requires
        # at least one argument'


@pytest.mark.xfail
def test_invalid_itemset():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = ValueError
    with pytest.raises(expected_exc):
        arr_np.itemset(8)
        # Numpy raises ValueError: can only convert an array of size 1
        # to a Python scalar
    with pytest.raises(expected_exc):
        arr_num.itemset(8)
        # cuPyNumeric raises KeyError: 'invalid key'


@pytest.mark.xfail
def test_out_of_index():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)
    expected_exc = IndexError
    with pytest.raises(expected_exc):
        arr_np.itemset(10, 4)
        # Numpy raises IndexError: index 10 is out of bounds for size 9
    with pytest.raises(expected_exc):
        arr_num.itemset(10, 4)
        # cuPyNumeric set the value of index 1 as 4
        # Original array:
        # [[193 212 238]
        #  [ 97 103 225]
        #  [197 107 174]]
        # new array
        # [[193   4 238]
        #  [ 97 103 225]
        #  [197 107 174]]


@pytest.mark.xfail
def test_tuple_out_of_index():
    shape = (3, 3)
    arr_num = num.random.randint(0, 300, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = IndexError
    with pytest.raises(expected_exc):
        arr_np.itemset((2, 3), 4)
        # Numpy raises IndexError: index 3 is out of bounds
        # for axis 1 with size 3
    with pytest.raises(expected_exc):
        arr_num.itemset((2, 2), 4)
        # cuPyNumeric raises ValueError: Out-of-bounds projection on
        # dimension 0 with index 3 for a store of shape Shape((3,))


@pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (4,) * ndim
    arr_num = num.random.randint(0, 30, size=shape)
    arr_np = np.array(arr_num)
    for itemset in generate_item(ndim):
        arr_num.itemset(itemset, 40)
        arr_np.itemset(itemset, 40)

        assert np.array_equal(arr_np, arr_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
