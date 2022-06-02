# Copyright 2022 NVIDIA Corporation
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
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test():
    # test on the 1D array:
    a = mk_seq_array(np, (10,))
    a_num = num.array(a)
    res = np.compress([True, False, True], a, axis=0)
    res_num = num.compress([True, False, True], a_num, axis=0)
    assert np.array_equal(res_num, res)

    # testing compress on 2D array with different axis
    a = np.array([[1, 2], [3, 4], [5, 6]])
    num_a = num.array(a)
    res_np = np.compress([0, 1], a, axis=0)
    res_num = num.compress([0, 1], num_a, axis=0)
    assert np.array_equal(res_num, res_np)

    res_np = np.compress([0, 1], a, axis=1)
    res_num = num.compress([0, 1], num_a, axis=1)
    assert np.array_equal(res_num, res_np)

    # tesing when condition is a bool type
    res_np = np.compress([True, False], a, axis=1)
    res_num = num.compress([True, False], num_a, axis=1)
    assert np.array_equal(res_num, res_np)

    # testing with output array
    out_np = np.array([[1], [1], [1]])
    out_num = num.array(out_np)
    res_np = np.compress([0, 1], a, axis=1, out=out_np)
    res_num = num.compress([0, 1], num_a, axis=1, out=out_num)
    assert np.array_equal(res_num, res_np)
    assert np.array_equal(out_num, out_np)

    # the case when input and output arrays have different types
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    num_a = num.array(a)
    res_np = np.compress([0, 1], a, axis=1, out=out_np)
    res_num = num.compress([0, 1], num_a, axis=1, out=out_num)
    assert np.array_equal(res_num, res_np)
    assert np.array_equal(out_num, out_np)

    for ndim in range(1, LEGATE_MAX_DIM + 1):
        shape = (5,) * ndim
        np_arr = mk_seq_array(np, shape)
        num_arr = mk_seq_array(num, shape)
        # make sure condition is between 1 and 2
        np_condition = mk_seq_array(np, (5,)) % 2
        num_condition = mk_seq_array(num, (5,)) % 2
        res_np = np.compress(np_condition, np_arr)
        res_num = num.compress(num_condition, num_arr)
        assert np.array_equal(res_num, res_np)
        for axis in range(ndim):
            res_np = np.compress(np_condition, np_arr, axis)
            res_num = num.compress(num_condition, num_arr, axis)
            assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    test()
