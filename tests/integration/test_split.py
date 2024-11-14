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

import cupynumeric as num

# cupynumeric.split(a: ndarray, indices: Union[int, ndarray], axis: int = 0)
# → list[cupynumeric.array.ndarray]
# cupynumeric.vsplit(a: ndarray, indices: Union[int, ndarray])
# → list[cupynumeric.array.ndarray]    (axis=0)
# cupynumeric.hsplit(a: ndarray, indices: Union[int, ndarray])
# → list[cupynumeric.array.ndarray]    (axis=1)
# cupynumeric.dsplit(a: ndarray, indices: Union[int, ndarray])
# → list[cupynumeric.array.ndarray]    (axis=2)


DIM = 6
SIZES = [
    (0,),
    (1),
    (DIM),
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


SIZES_VSPLIT = [
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

SIZES_HSPLIT = [
    (DIM, 1),
    (DIM, DIM),
    (DIM, 1, 1),
    (DIM, DIM, DIM),
]


SIZES_DSPLIT = [
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

ARG_FUNCS = ("vsplit", "hsplit", "dsplit")


class TestSplitErrors:
    """
    this class is to test negative cases
    """

    @pytest.mark.parametrize("indices", (-2, 0, "hi", 1.0, None))
    def test_indices_negative(self, indices):
        ary = num.arange(10)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.split(ary, indices)
        with pytest.raises(expected_exc):
            np.split(ary, indices)

    def test_indices_divison(self):
        size = (3, 3, 3)
        ary = num.random.randint(low=0, high=100, size=size)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.split(ary, 2, 0)
        with pytest.raises(expected_exc):
            np.split(ary, 2, 0)

    def test_axis_negative(self):
        ary = num.arange(10)
        expected_exc = IndexError
        axis = -2
        with pytest.raises(expected_exc):
            num.split(ary, 1, axis=axis)
        with pytest.raises(expected_exc):
            np.split(ary, 1, axis=axis)

    def test_axis_bigger(self):
        ary = num.arange(10)
        axis = 2
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.split(ary, 5, axis=axis)
        with pytest.raises(expected_exc):
            np.split(ary, 5, axis=axis)

    @pytest.mark.parametrize("indices", (-2, 0, "hi", 1.0, None))
    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_indices_negative_different_split(self, func_name, indices):
        ary = num.arange(10)
        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)

        expected_exc = ValueError
        with pytest.raises(expected_exc):
            func_num(ary, indices)
        with pytest.raises(expected_exc):
            func_np(ary, indices)

    @pytest.mark.xfail
    def test_dimensions_vsplit(self):
        ary = []
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.vsplit(ary, 1)
            # cuPyNumeric returns [array([], dtype=float64)]
        with pytest.raises(expected_exc):
            np.vsplit(ary, 1)
            # Numpy raises
            # ValueError: vsplit only works on arrays of 2 or more dimensions

    @pytest.mark.xfail
    def test_dimensions_vsplit_1(self):
        ary = np.random.randint(low=0, high=100, size=(5))
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.vsplit(ary, 1)
            # cuPyNumeric returns the array
        with pytest.raises(expected_exc):
            np.vsplit(ary, 1)
            # Numpy raises
            # ValueError: vsplit only works on arrays of 2 or more dimensions

    @pytest.mark.xfail
    def test_dimensions_hsplit_0(self):
        ary = np.random.randint(low=0, high=100, size=(0,))
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.hsplit(ary, 1)
        with pytest.raises(expected_exc):
            np.hsplit(ary, 1)
            # Numpy returns  [array([], dtype=int64)]

    def test_dimensions_hsplit_1(self):
        ary = num.arange(10)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.hsplit(ary, 1)
        with pytest.raises(expected_exc):
            np.hsplit(ary, 1)

    @pytest.mark.parametrize("size", ((0,), (10,), (5, 5)))
    def test_dimensions_dsplit(self, size):
        ary = np.random.randint(low=0, high=100, size=size)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.dsplit(ary, 1)
        with pytest.raises(expected_exc):
            np.dsplit(ary, 1)


def compare_array(a, b):
    """
    Compare two array using zip method.
    """
    if len(a) != len(b):
        return False
    else:
        for each in zip(a, b):
            if not np.array_equal(*each):
                return False
    return True


def get_indices(arr, axis):
    """
    Generate the indices. split the array along axis.
    Include the divisible integer or a 1-D array of sorted integers.
    """
    indices_arr = []
    axis_size = arr.shape[axis]
    even_div = 1
    random_integer = np.random.randint(1, 10)

    if axis_size == 1:
        indices_arr.append(1)  # in index

    elif axis_size > 1:
        for div in range(2, int(math.sqrt(axis_size) + 1)):
            if axis_size % div == 0:
                indices_arr.append(div)  # add divisible integer
                even_div = div

        # add 1 and axis_size
        indices_arr.append(1)
        indices_arr.append(axis_size)

    # an index in the dimension of the array along axis
    indices_arr.append(list(range(1, axis_size, even_div)))

    # an index exceeds the dimension of the array along axis
    indices_arr.append(
        list(range(0, axis_size + even_div * random_integer, even_div))
    )
    indices_arr.append(
        list(range(axis_size + even_div * random_integer, 0, -even_div))
    )

    return indices_arr


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_split(size):
    a = np.random.randint(low=0, high=100, size=size)
    axis_list = list(range(a.ndim)) + [-1]
    for axis in axis_list:
        input_arr = get_indices(a, axis)
        for input_opt in input_arr:
            res_num = num.split(a, input_opt, axis=axis)
            res_np = np.split(a, input_opt, axis=axis)
            assert compare_array(res_num, res_np)


@pytest.mark.parametrize("size", SIZES_VSPLIT, ids=str)
def test_vsplit(size):
    a = np.random.randint(low=0, high=100, size=size)
    input_arr = get_indices(a, 0)
    for input_opt in input_arr:
        res_num = num.vsplit(a, input_opt)
        res_np = np.vsplit(a, input_opt)
        assert compare_array(res_num, res_np)


@pytest.mark.parametrize("size", SIZES_HSPLIT, ids=str)
def test_hsplit(size):
    a = np.random.randint(low=0, high=100, size=size)
    input_arr = get_indices(a, 1)
    for input_opt in input_arr:
        res_num = num.hsplit(a, input_opt)
        res_np = np.hsplit(a, input_opt)
        assert compare_array(res_num, res_np)


@pytest.mark.parametrize("size", SIZES_DSPLIT, ids=str)
def test_dsplit(size):
    a = np.random.randint(low=0, high=100, size=size)
    input_arr = get_indices(a, 2)
    for input_opt in input_arr:
        res_num = num.dsplit(a, input_opt)
        res_np = np.dsplit(a, input_opt)
        assert compare_array(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
