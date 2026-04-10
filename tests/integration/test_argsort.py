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
from utils.comparisons import allclose

# cupynumeric.argsort(a: ndarray, axis: int = -1, kind: SortType = 'quicksort',
# order: Optional = None) → ndarray

# ndarray.argsort(axis=-1, kind=None, order=None)

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

NO_EMPTY_SIZES = [
    (1),
    (DIM),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]

STABLE_SORT_TYPES = ["stable", "mergesort"]
UNSTABLE_SORT_TYPES = ["heapsort", "quicksort"]
SORT_TYPES = STABLE_SORT_TYPES + UNSTABLE_SORT_TYPES


class TestArgSort(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.argsort(
            None
        )  # AxisError: axis -1 is out of bounds for array of dimension 0
        res_num = num.argsort(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    def test_arr_empty(self, arr):
        res_np = np.argsort(arr)
        res_num = num.argsort(arr)
        assert allclose(res_num, res_np)

    @pytest.mark.xfail
    def test_structured_array_order(self):
        dtype = [("name", "S10"), ("height", float), ("age", int)]
        values = [
            ("Arthur", 1.8, 41),
            ("Lancelot", 1.9, 38),
            ("Galahad", 1.7, 38),
        ]
        a_np = np.array(values, dtype=dtype)
        a_num = num.array(values, dtype=dtype)

        res_np = np.argsort(a_np, order="height")
        res_num = num.argsort(a_num, order="height")
        # cuPyNumeric raises AssertionError in
        # function cupynumeric/cupynumeric/eager.py:to_deferred_array
        #     if self.deferred is None:
        #         if self.parent is None:
        #
        # > assert self.runtime.is_supported_dtype(self.array.dtype)
        # E
        # AssertionError
        #
        # Passed on Numpy.
        assert allclose(res_np, res_num)

        res_np = np.argsort(a_np, order=["age", "height"])
        res_num = num.argsort(a_num, order=["age", "height"])
        # same as above.
        assert allclose(res_np, res_num)

    def test_axis_out_bound(self):
        arr = [-1, 0, 1, 2, 10]
        with pytest.raises(ValueError):
            num.argsort(arr, axis=2)

    @pytest.mark.xfail
    def test_sort_type_invalid(self):
        size = (3, 3, 2)
        arr_np = np.random.randint(-3, 3, size)
        arr_num = num.array(arr_np)
        res_np = np.argsort(arr_np, kind="negative")
        res_num = num.argsort(arr_num, kind="negative")
        # Numpy raises "ValueError: sort kind must be one of 'quick',
        # 'heap', or 'stable' (got 'negative')"
        # cuPyNumeric passed. The code basically supports ‘stable’
        # or not ‘stable’.
        assert allclose(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic_axis(self, size):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.argsort(arr_np, axis=axis)
            res_num = num.argsort(arr_num, axis=axis)
            assert allclose(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", STABLE_SORT_TYPES)
    def test_basic_axis_sort_type(self, size, sort_type):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.argsort(arr_np, axis=axis, kind=sort_type)
            res_num = num.argsort(arr_num, axis=axis, kind=sort_type)
            assert allclose(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", UNSTABLE_SORT_TYPES)
    def test_basic_axis_sort_type_unstable(self, size, sort_type):
        # have to guarantee unique values in input
        # see https://github.com/nv-legate/cupynumeric/issues/782
        arr_np = np.arange(np.prod(size))
        np.random.shuffle(arr_np)
        arr_np = arr_np.reshape(size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.argsort(arr_np, axis=axis, kind=sort_type)
            res_num = num.argsort(arr_num, axis=axis, kind=sort_type)
            assert allclose(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    def test_arr_basic_axis(self, size):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_np_copy = arr_np
            arr_np_copy.argsort(axis=axis)
            arr_num_copy = arr_num
            arr_num_copy.argsort(axis=axis)
            assert allclose(arr_np_copy, arr_num_copy)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", STABLE_SORT_TYPES)
    def test_arr_basic_axis_sort(self, size, sort_type):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_np_copy = arr_np
            arr_np_copy.argsort(axis=axis, kind=sort_type)
            arr_num_copy = arr_num
            arr_num_copy.argsort(axis=axis, kind=sort_type)
            assert allclose(arr_np_copy, arr_num_copy)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", UNSTABLE_SORT_TYPES)
    def test_arr_basic_axis_sort_unstable(self, size, sort_type):
        # have to guarantee unique values in input
        # see https://github.com/nv-legate/cupynumeric/issues/782
        arr_np = np.arange(np.prod(size))
        np.random.shuffle(arr_np)
        arr_np = arr_np.reshape(size)
        arr_num = num.array(arr_np)
        for axis in range(-arr_num.ndim + 1, arr_num.ndim):
            arr_np_copy = arr_np
            arr_np_copy.argsort(axis=axis, kind=sort_type)
            arr_num_copy = arr_num
            arr_num_copy.argsort(axis=axis, kind=sort_type)
            assert allclose(arr_np_copy, arr_num_copy)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic_complex_axis(self, size):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        )
        arr_num = num.array(arr_np)
        for axis in range(-arr_np.ndim + 1, arr_np.ndim):
            res_np = np.sort(arr_np, axis=axis)
            res_num = num.sort(arr_num, axis=axis)
            assert allclose(res_num, res_np)

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
            assert allclose(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    def test_axis_none(self, size):
        """Test argsort with axis=None returns flat indices"""
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)

        res_np = np.argsort(arr_np, axis=None)
        res_num = num.argsort(arr_num, axis=None)

        # Verify shape is flattened
        assert res_num.shape == (np.prod(size),)
        assert res_np.shape == (np.prod(size),)

        # Verify indices are correct by comparing sorted values
        sorted_np = arr_np.flatten()[res_np]
        sorted_num = arr_num.flatten()[res_num]
        assert allclose(sorted_np, sorted_num)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("sort_type", SORT_TYPES)
    def test_axis_none_sort_type(self, size, sort_type):
        """Test argsort with axis=None works with all sort types"""
        arr_np = np.arange(np.prod(size))  # Unique values for unstable sorts
        np.random.shuffle(arr_np)
        arr_np = arr_np.reshape(size)
        arr_num = num.array(arr_np)

        res_np = np.argsort(arr_np, axis=None, kind=sort_type)
        res_num = num.argsort(arr_num, axis=None, kind=sort_type)

        assert res_num.shape == (np.prod(size),)
        sorted_np = arr_np.flatten()[res_np]
        sorted_num = arr_num.flatten()[res_num]
        assert allclose(sorted_np, sorted_num)

    @pytest.mark.xfail
    def test_0d_axis_none(self):
        """Test argsort on 0-D array with axis=None"""
        # cuPyNumeric returns shape () but NumPy returns shape (1,)
        # NumPy flattens 0-D arrays to 1-D when axis=None
        arr_np = np.array(42)  # 0-D array
        arr_num = num.array(42)

        # Verify input is 0-D
        assert arr_np.shape == ()
        assert arr_num.shape == ()

        res_np = np.argsort(arr_np, axis=None)
        res_num = num.argsort(arr_num, axis=None)

        # Verify output is 1-D with shape (1,) - axis=None flattens
        assert res_num.shape == (1,)
        assert res_np.shape == (1,)

        # Verify indices are correct
        assert np.array_equal(res_num, res_np)

    @pytest.mark.xfail
    def test_0d_invalid_axis(self):
        """Test argsort on 0-D array raises AxisError for axis=0"""
        # cuPyNumeric raises ValueError but NumPy raises AxisError
        from utils.utils import AxisError

        arr_np = np.array(42)
        arr_num = num.array(42)

        # NumPy behavior
        with pytest.raises(AxisError):
            np.argsort(arr_np, axis=0)

        # cuPyNumeric should match
        with pytest.raises(AxisError):
            num.argsort(arr_num, axis=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
