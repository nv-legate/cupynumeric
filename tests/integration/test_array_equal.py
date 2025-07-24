#!/usr/bin/env python
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


@pytest.mark.parametrize(
    "arr", (1, [], [[]], [1], [[1, 2], [3, 4]]), ids=lambda arr: f"(arr={arr})"
)
def test_equal_arrays(arr):
    res_np = np.array_equal(arr, arr)
    res_num = num.array_equal(arr, arr)
    assert res_np is bool(res_num) is True


ARRAYS = (
    (1, 2),
    (1, [1]),
    (1, []),
    ([], [1]),
    ([], [[]]),
    ([1], [2]),
    ([1, 2], [1, 3]),
    ([1, 2], [1, 2, 3]),
    ([1, 2], [[1, 2]]),
)


@pytest.mark.parametrize(("arr1", "arr2"), ARRAYS, ids=str)
def test_unequal_arrays(arr1, arr2):
    res_np = np.array_equal(arr1, arr2)
    res_num = num.array_equal(arr1, arr2)
    assert res_np is bool(res_num) is False

    res_np_swapped = np.array_equal(arr2, arr1)
    res_num_swapped = num.array_equal(arr2, arr1)
    assert res_np_swapped is bool(res_num_swapped) is False


DTYPES = ((np.int32, np.float64), (np.float64, np.complex128))


@pytest.mark.parametrize(("dtype1", "dtype2"), DTYPES, ids=str)
def test_equal_values_with_different_dtype(dtype1, dtype2):
    array = [1, 2, 3]
    np_arr1 = np.array(array, dtype=dtype1)
    np_arr2 = np.array(array, dtype=dtype2)
    num_arr1 = num.array(array, dtype=dtype1)
    num_arr2 = num.array(array, dtype=dtype2)

    res_np = np.array_equal(np_arr1, np_arr2)
    res_num = num.array_equal(num_arr1, num_arr2)
    assert res_np == res_num

    res_np_swapped = np.array_equal(np_arr2, np_arr1)
    res_num_swapped = num.array_equal(num_arr2, num_arr1)
    assert res_np_swapped == res_num_swapped


@pytest.mark.parametrize(
    "equal_nan", (False, pytest.param(True, marks=pytest.mark.xfail))
)
@pytest.mark.parametrize(
    "arr",
    ([np.nan], [1, 2, np.nan], [[1, 2], [3, np.nan]]),
    ids=lambda arr: f"(arr={arr})",
)
def test_equal_nan_basic(arr, equal_nan):
    # If equal_nan is True,
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    res_np = np.array_equal(arr, arr, equal_nan=equal_nan)
    res_num = num.array_equal(arr, arr, equal_nan=equal_nan)
    assert res_np == res_num


@pytest.mark.parametrize(
    "equal_nan", (False, pytest.param(True, marks=pytest.mark.xfail))
)
def test_equal_nan_complex_values(equal_nan):
    # If equal_nan is True,
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    a = np.array([1, 1 + 1j])
    b = a.copy()
    a.real = np.nan
    b.imag = np.nan

    res_np = np.array_equal(a, b, equal_nan=equal_nan)
    res_num = num.array_equal(a, b, equal_nan=equal_nan)
    assert res_np == res_num


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
