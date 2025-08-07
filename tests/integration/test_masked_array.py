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
from legate.core import LEGATE_MAX_DIM

import cupynumeric as num

SCALARS = (0, -10.5, 1 + 1j)

ARRAYS = ([], (1, 2), ((1, 2),), [(1, 2), (3, 4.1)], ([1, 2.1], [3, 4 + 4j]))


def strict_type_equal(a, b):
    return np.array_equal(a, b) and a.dtype == b.dtype


@pytest.mark.parametrize(
    "obj", SCALARS + ARRAYS, ids=lambda obj: f"(object={obj})"
)
def test_masked_array_basic(obj):
    res_np = np.ma.masked_array(obj)
    res_num = num.ma.masked_array(obj)
    assert strict_type_equal(res_np, res_num)


def test_masked_array_ndarray():
    obj = [[1, 2], [3, 4]]
    res_np = np.ma.masked_array(np.array(obj))
    res_num = num.ma.masked_array(num.array(obj))
    assert strict_type_equal(res_np, res_num)


def test_masked_array_with_mask():
    obj = np.array([10, 20, 30, 40, 50])
    mask = np.array([False, True, False, True, False])

    res_np = np.ma.masked_array(obj, mask=mask)
    res_num = num.ma.masked_array(obj, mask=mask)

    assert strict_type_equal(res_np, res_num)


DTYPES = (np.int32, np.float64, np.complex128)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: f"(dtype={dtype})")
@pytest.mark.parametrize(
    "obj",
    (0, -10.5, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda obj: f"(object={obj})",
)
def test_masked_array_dtype(obj, dtype):
    res_np = np.ma.masked_array(obj, dtype=dtype)
    res_num = num.ma.masked_array(obj, dtype=dtype)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize(
    "ndmin",
    range(-1, LEGATE_MAX_DIM + 1),
    ids=lambda ndmin: f"(ndmin={ndmin})",
)
@pytest.mark.parametrize(
    "obj",
    (0, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda obj: f"(object={obj})",
)
def test_masked_array_ndmin(obj, ndmin):
    res_np = np.ma.masked_array(obj, ndmin=ndmin)
    res_num = num.ma.masked_array(obj, ndmin=ndmin)
    assert strict_type_equal(res_np, res_num)


class TestArrayErrors:
    @pytest.mark.parametrize(
        "dtype", (np.int32, np.float64), ids=lambda dtype: f"(dtype={dtype})"
    )
    @pytest.mark.parametrize(
        "obj", (1 + 1j, [1, 2, 3.0, 4 + 4j]), ids=lambda obj: f"(obj={obj})"
    )
    def test_invalid_dtype(self, obj, dtype):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.array(obj, dtype=dtype)
        with pytest.raises(expected_exc):
            num.array(obj, dtype=dtype)


def test_asarray_ndarray():
    obj = [[1, 2], [3, 4]]
    res_np = np.asarray(np.ma.masked_array(obj))
    res_num = num.asarray(num.ma.masked_array(obj))
    assert strict_type_equal(res_np, res_num)


def test_masked_array_size_and_shape() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[False, False, True], [False, True, False]])
    m_arr = num.ma.MaskedArray(data, mask=mask)

    assert m_arr.shape == (2, 3)
    assert m_arr.size == 6
    

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
