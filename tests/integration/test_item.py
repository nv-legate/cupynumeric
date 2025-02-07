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
from utils.generators import generate_item
from utils.utils import MAX_DIM_RANGE

import cupynumeric as num


@pytest.mark.xfail
def test_no_item():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    expected_exc = KeyError
    with pytest.raises(expected_exc):
        arr_np.item()
        # Numpy raises KeyError: 'invalid key'
    with pytest.raises(expected_exc):
        arr_num.item()
        # cuPyNumeric raises ValueError: can only convert an array
        # of size 1 to a Python scalar


@pytest.mark.xfail
def test_out_of_bound():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)
    expected_exc = IndexError
    with pytest.raises(expected_exc):
        arr_np.item(10)
        # Numpy raises IndexError: index 10 is out of bounds for size 9
    with pytest.raises(expected_exc):
        arr_num.item(10)
        # cuPyNumeric returns some value


@pytest.mark.xfail
def test_out_of_index():
    shape = (3, 3)
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)
    expected_exc = IndexError
    with pytest.raises(expected_exc):
        arr_np.item(2, 4)
        # Numpy raises IndexError: index 4 is out of bounds
        # for axis 1 with size 3
    with pytest.raises(expected_exc):
        arr_num.item(2, 4)
        # cuPyNumeric: ValueError: Out-of-bounds projection on dimension 0
        # with index 4 for a store of shape Shape((3,))


def test_empty_no_item():
    shape = ()
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    res_np = arr_np.item()
    res_num = arr_num.item()
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
def test_ndim(ndim):
    shape = (4,) * ndim
    arr_num = num.random.randint(0, 3, size=shape)
    arr_np = np.array(arr_num)

    for item in generate_item(ndim):
        res_num = arr_num.item(item)
        res_np = arr_np.item(item)
        assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
