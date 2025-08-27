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
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


class TestMedianErrors:
    def test_none_array(self):
        expected_exc = TypeError
        msg = "unsupported operand type"
        with pytest.raises(expected_exc, match=msg):
            np.median(None)
        expected_exc = TypeError
        msg = "'None' is not suported input to 'median'"
        with pytest.raises(expected_exc, match=msg):
            num.median(None)

    def test_out(self):
        array = np.arange(0, 5)
        out_a = np.arange(0, 3)
        expected_exc = ValueError
        # shorten the error message as it vary depending on numpy version
        msg = "output parameter for reduction operation add"
        with pytest.raises(expected_exc, match=msg):
            np.median(array, out=out_a)
        msg = "wrong shape on output array"
        with pytest.raises(expected_exc, match=msg):
            num.median(array, out=out_a)

    def test_median_empty_array(self):
        expected_exc = IndexError
        # Numpy returns Warning instead of the Error
        # msg="invalid value encountered in scalar divide"
        # with pytest.raises(expected_exc, match=msg):
        #    np.median([])
        msg = "Cannot do a non-empty take\(\) from an empty axis."
        with pytest.raises(expected_exc, match=msg):
            num.median([])


class TestMedian:
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE[:-1])
    @pytest.mark.parametrize("keepdims", (False, True))
    def test_median_basic(self, ndim, keepdims):
        shape = np.random.randint(1, 4, ndim, dtype=int)
        size = 1
        for dim in shape:
            size *= dim
        np_arr = mk_seq_array(np, shape)
        num_arr = num.array(np_arr)
        for axis in range(0, ndim):
            np_res = np.median(np_arr, axis=axis, keepdims=keepdims)
            num_res = num.median(num_arr, axis=axis, keepdims=keepdims)
            assert np.array_equal(np_res, num_res)

        @pytest.mark.parametrize(
            "axis", (None, -2, [1, -1], [0, 1, 2, 3], [0, 3], 1)
        )
        def test_axis(self, axis):
            shape = np.random.randint(3, 10, 4, dtype=int)
            size = 1
            for dim in shape:
                size *= dim
            np_arr = mk_seq_array(np, size).reshape(shape)
            num_arr = mk_seq_array(num, size).reshape(shape)

            np_res = np.median(np_arr, axis=axis)
            num_res = num.median(num_arr, axis=axis)
            assert np.array_equal(np_res, num_res)

    def test_median_identical_values(self):
        assert num.median([5, 5, 5, 5]) == 5

    def test_median_nan_behavior(self):
        assert num.isnan(num.median([1, 2, np.nan]))

    def test_median_with_out_array(self):
        arr = num.array([[1, 3, 5, 7], [9, 11, 13, 15]])
        out = num.zeros((4,))
        num.median(arr, axis=0, out=out)
        assert np.array_equal(
            out, [5.0, 7.0, 9.0, 11.0]
        )  # Ensure result is written into `out` array


class TestNanMedianErrors:
    def test_none_array(self):
        expected_exc = TypeError
        msg = "unsupported operand type"
        with pytest.raises(expected_exc, match=msg):
            np.nanmedian(None)
        expected_exc = TypeError
        msg = "'None' is not suported input to 'nanmedian'"
        with pytest.raises(expected_exc, match=msg):
            num.nanmedian(None)

    def test_out(self):
        array = np.arange(0, 5)
        out_a = np.arange(0, 3)
        expected_exc = ValueError
        # shorten the error message as it vary depending on numpy version
        msg = "output parameter for reduction operation add"
        # with pytest.raises(expected_exc, match=msg):
        np.nanmedian(array, out=out_a)
        print("IRINA DEBUG out_a", out_a)
        msg = "data type <class 'numpy.int64'> not inexact"
        with pytest.raises(expected_exc, match=msg):
            num.nanmedian(array, out=out_a)

    def test_median_overwrite_input(self):
        arr = num.array([7, 1, 5, 3])
        median = num.median(arr, overwrite_input=True)
        assert median == 4  # Ensure correct median
        assert np.array_equal(arr, [1, 3, 5, 7])  # Input array is modified

        arr_2d = num.array([[1, 3], [5, 7]])
        median_axis0 = num.median(arr_2d, axis=0, overwrite_input=True)
        assert np.array_equal(median_axis0, [3, 5])
        assert np.array_equal(
            arr_2d, [[1, 3], [5, 7]]
        )  # Input array is not always modified along axis


class TestNanmedian:
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    def test_nanmedian_basic(self, ndim):
        shape = np.random.randint(2, 5, ndim, dtype=int)
        size = 1
        for dim in shape:
            size *= dim
        np_arr = mk_seq_array(np, shape).astype(float)
        np.putmask(np_arr, np_arr > 2, np.nan)
        num_arr = num.array(np_arr)
        for axis in range(0, ndim):
            np_res = np.nanmedian(np_arr, axis=axis)
            num_res = num.nanmedian(num_arr, axis=axis)
            assert np.array_equal(np_res, num_res, equal_nan=True)

    def test_nanmedian_identical_values_with_nans(self):
        assert num.nanmedian([np.nan, np.nan, 5, np.nan]) == 5

    def test_median_with_out_array(self):
        arr = num.array([[1, 3, 5, 7, np.nan], [9, np.nan, 11, 13, 15]])
        out = num.zeros((5,))
        num.nanmedian(arr, axis=0, out=out)
        assert np.array_equal(
            out, [5.0, 3.0, 8.0, 10.0, 15.0]
        )  # Ensure result is written into `out` array

    def test_nanmedian_overwrite_input(self):
        arr = num.array([7, 1, np.nan, 3])
        median = num.nanmedian(arr, overwrite_input=True)
        assert median == 3  # Ensure correct median
        # FIXME: doesn't work
        # assert np.array_equal(arr, [1, 3, 7, np.nan], equal_nan=True)

        arr_2d = num.array([[1, np.nan], [5, 7]])
        median_axis0 = num.nanmedian(arr_2d, axis=0, overwrite_input=True)
        assert np.array_equal(median_axis0, [3, 7])
        assert np.array_equal(
            arr_2d, [[1, np.nan], [5, 7]], equal_nan=True
        )  # Check if input array is altered


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
