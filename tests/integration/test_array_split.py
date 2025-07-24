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

import math

import numpy as np
import pytest
from utils.utils import check_module_function

import cupynumeric as num

DIM = 20

# test the array_split routines on empty, singleton, 2D and 3D arrays
# w/ integers, list of indicies. vsplit, hsplit, dsplit are included
# in the following loops(axis = 0: vsplit, 1: hsplit, 2: dsplit)
SIZES = [(0,), (0, 10), (1), (1, 1), (1, 1, 1), (DIM, DIM), (DIM, DIM, DIM)]


class TestArraySplitErrors:
    """
    this class is to test negative cases
    """

    def test_indices_negative(self):
        # negative indices should be not accepted
        ary = np.arange(10)
        msg = r"number sections must be larger than 0"
        with pytest.raises(ValueError, match=msg):
            num.array_split(ary, -2)

    def test_indices_0(self):
        # 0 indices should be not accepted
        ary = np.arange(10)
        msg = r"number sections must be larger than 0"
        with pytest.raises(ValueError, match=msg):
            num.array_split(ary, 0)

    def test_axis_bigger(self):
        ary = np.arange(9)
        with pytest.raises(ValueError):
            num.array_split(ary, len(ary) // 2, 2)

    def test_axis_negative(self):
        ary = np.arange(9)
        with pytest.raises(IndexError):
            num.array_split(ary, len(ary) // 2, -2)

    def test_indices_str_type(self):
        expected_exc = ValueError
        arr_np = np.arange(10)
        arr_num = num.arange(10)
        # Split points in the passed `indices` should be integer
        with pytest.raises(expected_exc):
            num.array_split(arr_np, ["a", "b"])
        with pytest.raises(expected_exc):
            num.array_split(arr_num, ["a", "b"])


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_split(size):
    a = np.random.randint(low=0, high=100, size=size)
    axis_list = list(range(a.ndim))
    axis_list.append(-1)
    for axis in axis_list:
        input_arr = []
        even_div = None
        uneven_div = None

        if a.shape[axis] > 1:
            for div in range(1, (int)(math.sqrt(a.shape[axis]) + 1)):
                if a.shape[axis] % div == 0:
                    even_div = div
                else:
                    uneven_div = div
                if even_div is not None and uneven_div is not None:
                    break
        else:
            even_div = 2
            uneven_div = 1

        # divisible integer
        input_arr.append(even_div)
        # indivisble integer
        input_arr.append(uneven_div)
        # integer larger than shape[axis]
        input_arr.append(a.shape[axis] + np.random.randint(1, 10))
        # indices array which has points
        # within the target dimension of the src array
        if a.shape[axis] > 1:
            input_arr.append(list(range(1, a.shape[axis], even_div)))
        # indices array which has points
        # out of the target dimension of the src array
        input_arr.append(
            list(
                range(
                    0,
                    a.shape[axis] + even_div * np.random.randint(1, 10),
                    even_div,
                )
            )
        )
        input_arr.append(
            list(
                range(
                    a.shape[axis] + even_div * np.random.randint(1, 10),
                    0,
                    -even_div,
                )
            )
        )

        for input_opt in input_arr:
            # test divisible integer or indices
            print_msg = f"np.array_split({a.shape}, {input_opt}, {axis})"
            # Check if both impls produce the error
            # for non-viable options

            check_module_function(
                "array_split", [a, input_opt], {"axis": axis}, print_msg
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
