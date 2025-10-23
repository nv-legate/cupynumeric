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

import cupynumeric as num

DIM = 5
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

SORT_TYPES = ["quicksort", "mergesort", "heapsort", "stable"]


# cupynumeric.sort(a: ndarray, axis: int = -1,
# kind: SortType = 'quicksort', order: Optional = None) → ndarray
# ndarray.sort(axis=-1, kind=None, order=None)


class TestSort(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.sort(
            None
        )  # AxisError: axis -1 is out of bounds for array of dimension 0
        res_num = num.sort(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    def test_arr_empty(self, arr):
        res_np = np.sort(arr)
        res_num = num.sort(arr)
        assert np.array_equal(res_num, res_np)

    def test_axis_out_bound(self):
        arr = [-1, 0, 1, 2, 10]
        with pytest.raises(ValueError):
            num.sort(arr, axis=2)

    @pytest.mark.xfail
    def test_sorttype_invalid(self):
        size = (3, 3, 2)
        arr_np = np.random.randint(-3, 3, size)
        arr_num = num.array(arr_np)
        res_np = np.sort(arr_np, kind="negative")
        res_num = num.sort(arr_num, kind="negative")
        # Numpy raises "ValueError: sort kind must be one of 'quick', 'heap',
        # or 'stable' (got 'negative')"
        # cuPyNumeric passed. The code basically supports ‘stable’
        # or not ‘stable’.
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic_axis(self, size):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.sort(arr_np, axis=axis)
            res_num = num.sort(arr_num, axis=axis)
            assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", SORT_TYPES)
    def test_basic_axis_sort_type(self, size, sort_type):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.sort(arr_np, axis=axis, kind=sort_type)
            res_num = num.sort(arr_num, axis=axis, kind=sort_type)
            assert np.array_equal(res_num, res_np)

    @pytest.mark.skip
    @pytest.mark.parametrize("size", SIZES)
    def test_arr_basic_axis(self, size):
        # Set skip due to https://github.com/nv-legate/cupynumeric/issues/781
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_np_copy = arr_np
            arr_np_copy.sort(axis=axis)
            arr_num_copy = arr_num
            arr_num_copy.sort(axis=axis)
            assert np.array_equal(arr_np_copy, arr_num_copy)

    @pytest.mark.skip
    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", SORT_TYPES)
    def test_arr_basic_axis_sort(self, size, sort_type):
        # Set skip due to https://github.com/nv-legate/cupynumeric/issues/781
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_np_copy = arr_np
            arr_np_copy.sort(axis=axis, kind=sort_type)
            arr_num_copy = arr_num
            arr_num_copy.sort(axis=axis, kind=sort_type)
            assert np.array_equal(arr_np_copy, arr_num_copy)

    @pytest.mark.skip
    @pytest.mark.parametrize("size", SIZES)
    def test_compare_arr_axis(self, size):
        # Set skip due to https://github.com/nv-legate/cupynumeric/issues/781
        arr_num = num.random.randint(-100, 100, size)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_num_copy = arr_num
            res_num = num.sort(arr_num_copy, axis=axis)
            arr_num_copy.sort(axis=axis)
            assert np.array_equal(res_num, arr_num_copy)

    @pytest.mark.skip
    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", SORT_TYPES)
    def test_compare_arr_axis_sort(self, size, sort_type):
        # Set skip due to https://github.com/nv-legate/cupynumeric/issues/781
        arr_num = num.random.randint(-100, 100, size)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_num_copy = arr_num
            res_num = num.sort(arr_num_copy, axis=axis, kind=sort_type)
            arr_num_copy.sort(axis=axis, kind=sort_type)
            assert np.array_equal(res_num, arr_num_copy)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic_complex_axis(self, size):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        )
        arr_num = num.array(arr_np)
        print(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.sort(arr_np, axis=axis)
            res_num = num.sort(arr_num, axis=axis)
            assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", SORT_TYPES)
    def test_basic_complex_axis_sort(self, size, sort_type):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        )
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.sort(arr_np, axis=axis, kind=sort_type)
            res_num = num.sort(arr_num, axis=axis, kind=sort_type)
            assert np.array_equal(res_num, res_np)

    @pytest.mark.skip
    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", SORT_TYPES)
    def test_compare_complex_arr_axis_sort(self, size, sort_type):
        # Set skip due to https://github.com/nv-legate/cupynumeric/issues/781
        arr_num = (
            num.random.randint(-100, 100, size)
            + num.random.randint(-100, 100, size) * 1.0j
        )
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_num_copy = arr_num
            res_num = num.sort(arr_num_copy, axis=axis, kind=sort_type)
            arr_num_copy.sort(axis=axis, kind=sort_type)
            assert np.array_equal(res_num, arr_num_copy)

    def test_string_array_fallback(self):
        """Test that sort falls back to NumPy for unsupported dtypes"""
        arr_np = np.array(["zebra", "apple", "mango", "banana"])
        expected = np.sort(arr_np)

        with pytest.warns(
            RuntimeWarning,
            match="cuPyNumeric does not support.*falling back to NumPy",
        ):
            result = num.sort(arr_np)

        assert np.array_equal(result, expected)
        # Result may be NumPy array since cuPyNumeric doesn't support string dtype
        assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
