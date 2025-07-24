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

UNSUPPORTED_OBJECTS = (
    None,
    "somestr",
    ["one", "two"],
    [("name", "S10"), ("height", float), ("age", int)],
)


def strict_type_equal(a, b):
    return np.array_equal(a, b) and a.dtype == b.dtype


@pytest.mark.parametrize(
    "obj", SCALARS + ARRAYS, ids=lambda obj: f"(object={obj})"
)
def test_array_basic(obj):
    res_np = np.array(obj)
    res_num = num.array(obj)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize("obj", UNSUPPORTED_OBJECTS)
def test_array_unsupported(obj):
    with pytest.raises(TypeError, match="cuPyNumeric does not support dtype"):
        num.array(obj)


def test_array_ndarray():
    obj = [[1, 2], [3, 4]]
    res_np = np.array(np.array(obj))
    res_num = num.array(num.array(obj))
    assert strict_type_equal(res_np, res_num)


DTYPES = (np.int32, np.float64, np.complex128)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: f"(dtype={dtype})")
@pytest.mark.parametrize(
    "obj",
    (0, -10.5, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda obj: f"(object={obj})",
)
def test_array_dtype(obj, dtype):
    res_np = np.array(obj, dtype=dtype)
    res_num = num.array(obj, dtype=dtype)
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
def test_array_ndmin(obj, ndmin):
    res_np = np.array(obj, ndmin=ndmin)
    res_num = num.array(obj, ndmin=ndmin)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize(
    "copy", (True, False), ids=lambda copy: f"(copy={copy})"
)
def test_array_copy(copy):
    x = [[1, 2, 3], [4, 5, 6]]
    x_np = np.array(x)
    xc_np = np.array(x_np, copy=copy)
    x_np[0, :] = [7, 8, 9]

    x_num = num.array(x)
    xc_num = num.array(x_num, copy=copy)
    x_num[0, :] = [7, 8, 9]

    assert strict_type_equal(xc_np, xc_num)


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


@pytest.mark.parametrize(
    "obj", SCALARS + ARRAYS, ids=lambda obj: f"(object={obj})"
)
def test_asarray_basic(obj):
    res_np = np.asarray(obj)
    res_num = num.asarray(obj)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize("obj", UNSUPPORTED_OBJECTS)
def test_asarray_unsupported(obj):
    with pytest.raises(TypeError, match="cuPyNumeric does not support dtype"):
        num.array(obj)


def test_asarray_ndarray():
    obj = [[1, 2], [3, 4]]
    res_np = np.asarray(np.array(obj))
    res_num = num.asarray(num.array(obj))
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: f"(dtype={dtype})")
@pytest.mark.parametrize(
    "obj",
    (0, -10.5, [], [1, 2], [[1, 2], [3, 4.1]]),
    ids=lambda obj: f"(object={obj})",
)
def test_asarray_dtype(obj, dtype):
    res_np = np.asarray(obj, dtype=dtype)
    res_num = num.asarray(obj, dtype=dtype)
    assert strict_type_equal(res_np, res_num)


@pytest.mark.parametrize(
    "src_dtype, tgt_dtype",
    ((np.int32, np.complex128), (np.float64, np.int64)),
    ids=str,
)
@pytest.mark.parametrize("func", ("array", "asarray"), ids=str)
def test_ndarray_dtype(src_dtype, tgt_dtype, func):
    shape = (1, 3, 1)
    arr_np = np.ndarray(shape, dtype=src_dtype)
    arr_num = num.array(arr_np)
    res_np = getattr(np, func)(arr_np, dtype=tgt_dtype)
    res_num = getattr(num, func)(arr_num, dtype=tgt_dtype)
    assert strict_type_equal(res_np, res_num)


class TestAsArrayErrors:
    @pytest.mark.parametrize(
        "dtype", (np.int32, np.float64), ids=lambda dtype: f"(dtype={dtype})"
    )
    @pytest.mark.parametrize(
        "obj", (1 + 1j, [1, 2, 3.0, 4 + 4j]), ids=lambda obj: f"(object={obj})"
    )
    def test_invalid_dtype(self, obj, dtype):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.asarray(obj, dtype=dtype)
        with pytest.raises(expected_exc):
            num.asarray(obj, dtype=dtype)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
