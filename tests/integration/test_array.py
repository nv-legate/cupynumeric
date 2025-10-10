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


class TestAsArrayBugFixes:
    @pytest.mark.parametrize(
        "obj", SCALARS + ARRAYS, ids=lambda obj: f"(object={obj})"
    )
    def test_asarray_basic(self, obj):
        res_np = np.asarray(obj)
        res_num = num.asarray(obj)
        assert strict_type_equal(res_np, res_num)

    @pytest.mark.parametrize("obj", UNSUPPORTED_OBJECTS)
    def test_asarray_unsupported(self, obj):
        with pytest.raises(
            TypeError, match="cuPyNumeric does not support dtype"
        ):
            num.array(obj)

    def test_asarray_ndarray(self):
        obj = [[1, 2], [3, 4]]
        res_np = np.asarray(np.array(obj))
        res_num = num.asarray(num.array(obj))
        assert strict_type_equal(res_np, res_num)

    @pytest.mark.parametrize(
        "dtype", DTYPES, ids=lambda dtype: f"(dtype={dtype})"
    )
    @pytest.mark.parametrize(
        "obj",
        (0, -10.5, [], [1, 2], [[1, 2], [3, 4.1]]),
        ids=lambda obj: f"(object={obj})",
    )
    def test_asarray_dtype(self, obj, dtype):
        res_np = np.asarray(obj, dtype=dtype)
        res_num = num.asarray(obj, dtype=dtype)
        assert strict_type_equal(res_np, res_num)

    @pytest.mark.parametrize(
        "src_dtype, tgt_dtype",
        ((np.int32, np.complex128), (np.float64, np.int64)),
        ids=str,
    )
    @pytest.mark.parametrize("func", ("array", "asarray"), ids=str)
    def test_ndarray_dtype(self, src_dtype, tgt_dtype, func):
        """Test converting to different dtypes"""
        shape = (1, 3, 1)
        arr_np = np.ndarray(shape, dtype=src_dtype)
        arr_num = num.array(arr_np)
        res_np = getattr(np, func)(arr_np, dtype=tgt_dtype)
        res_num = getattr(num, func)(arr_num, dtype=tgt_dtype)
        assert strict_type_equal(res_np, res_num)

    def test_asarray_transpose(self):
        """Test with transposed arrays"""
        x = np.ones((4, 5))
        y = x.transpose()
        res_np = np.asarray(y)
        res_num = num.asarray(y)
        assert strict_type_equal(res_np, res_num)
        assert res_num.shape == (5, 4)
        assert res_num.dtype == np.float64

    def test_asarray_boolean_indexing(self):
        """Test asarray with boolean indexing and broadcasting"""
        x = np.zeros((3, 10))
        mask = np.array(
            [False, False, False, False, True, True, False, True, True, True]
        )
        y = x[:, mask]
        z = y - np.zeros((3,))[:, np.newaxis]
        res_np = np.asarray(z)
        res_num = num.asarray(z)
        assert strict_type_equal(res_np, res_num)
        assert res_num.shape == (3, 5)  # 5 True values in mask
        assert res_num.dtype == np.float64

    def test_asarray_squeeze_transpose(self):
        """Test asarray with squeezed and transposed arrays"""
        x = np.zeros((64, 128, 1, 1))
        y = np.squeeze(x).T
        res_np = np.asarray(y)
        res_num = num.asarray(y)
        assert strict_type_equal(res_np, res_num)
        assert res_num.shape == (128, 64)  # squeezed then transposed
        assert res_num.dtype == np.float64

    def test_asarray_empty_squeeze(self):
        """Test asarray with empty arrays after squeeze"""
        x = np.zeros((0, 1, 64, 1, 1))
        y = np.squeeze(x)
        res_np = np.asarray(y)
        res_num = num.asarray(y)
        assert strict_type_equal(res_np, res_num)
        assert res_num.shape == (0, 64)  # squeezed empty array
        assert res_num.dtype == np.float64

    def test_asarray_empty_arrays_various_shapes(self):
        """Test asarray with various empty array shapes"""
        x1 = np.zeros((0, 5, 3))
        res_np1 = np.asarray(x1)
        res_num1 = num.asarray(x1)
        assert strict_type_equal(res_np1, res_num1)
        assert res_num1.shape == (0, 5, 3)

        x2 = np.zeros((3, 0, 5))
        res_np2 = np.asarray(x2)
        res_num2 = num.asarray(x2)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.shape == (3, 0, 5)

        x3 = np.zeros((3, 5, 0))
        res_np3 = np.asarray(x3)
        res_num3 = num.asarray(x3)
        assert strict_type_equal(res_np3, res_num3)
        assert res_num3.shape == (3, 5, 0)

    def test_asarray_squeezed_arrays_various_shapes(self):
        """Test asarray with squeezed arrays of various shapes"""
        x1 = np.zeros((5, 1, 3))
        y1 = np.squeeze(x1)
        res_np1 = np.asarray(y1)
        res_num1 = num.asarray(y1)
        assert strict_type_equal(res_np1, res_num1)
        assert res_num1.shape == (5, 3)

        x2 = np.zeros((5, 1, 3, 1))
        y2 = np.squeeze(x2)
        res_np2 = np.asarray(y2)
        res_num2 = num.asarray(y2)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.shape == (5, 3)

        x3 = np.zeros((1, 1, 1))
        y3 = np.squeeze(x3)
        res_np3 = np.asarray(y3)
        res_num3 = num.asarray(y3)
        assert strict_type_equal(res_np3, res_num3)
        assert res_num3.shape == ()

    def test_asarray_view_operations(self):
        """Test asarray with various view operations"""
        # Slice view
        x1 = np.arange(24).reshape(4, 6)
        y1 = x1[1:3, 2:5]
        res_np1 = np.asarray(y1)
        res_num1 = num.asarray(y1)
        assert strict_type_equal(res_np1, res_num1)
        assert res_num1.shape == (2, 3)

        # Strided view
        x2 = np.arange(20).reshape(4, 5)
        y2 = x2[::2, ::2]
        res_np2 = np.asarray(y2)
        res_num2 = num.asarray(y2)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.shape == (2, 3)

        # Negative stride view
        x3 = np.arange(12).reshape(3, 4)
        y3 = x3[::-1, ::-1]
        res_np3 = np.asarray(y3)
        res_num3 = num.asarray(y3)
        assert strict_type_equal(res_np3, res_num3)
        assert res_num3.shape == (3, 4)

    def test_asarray_reshaped_arrays(self):
        """Test asarray with reshaped arrays"""
        x = np.arange(24).reshape(4, 6)
        y = x.reshape(2, 12)
        res_np = np.asarray(y)
        res_num = num.asarray(y)
        assert strict_type_equal(res_np, res_num)
        assert res_num.shape == (2, 12)

        # Reshape with -1
        y2 = x.reshape(-1)
        res_np2 = np.asarray(y2)
        res_num2 = num.asarray(y2)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.shape == (24,)

    def test_asarray_advanced_indexing(self):
        """Test asarray with advanced indexing operations"""
        x = np.arange(24).reshape(4, 6)

        # Integer array indexing
        indices = np.array([0, 2, 3])
        y1 = x[indices, :]
        res_np1 = np.asarray(y1)
        res_num1 = num.asarray(y1)
        assert strict_type_equal(res_np1, res_num1)
        assert res_num1.shape == (3, 6)

        # Boolean array indexing
        mask = np.array([True, False, True, False, True, False])
        y2 = x[:, mask]
        res_np2 = np.asarray(y2)
        res_num2 = num.asarray(y2)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.shape == (4, 3)

    def test_asarray_mixed_operations(self):
        """Test asarray with complex combinations of operations"""
        # Transpose + slice + squeeze
        x = np.arange(48).reshape(4, 3, 4)
        y = x.transpose(2, 0, 1)[:, 1:3, :]
        z = np.squeeze(y)
        res_np = np.asarray(z)
        res_num = num.asarray(z)
        assert strict_type_equal(res_np, res_num)
        assert res_num.shape == (4, 2, 3)

        # Boolean indexing + transpose + squeeze
        x2 = np.arange(60).reshape(5, 4, 3)
        mask = np.array([True, False, True, False])
        y2 = x2[:, mask, :].transpose(2, 0, 1)
        z2 = np.squeeze(y2)
        res_np2 = np.asarray(z2)
        res_num2 = num.asarray(z2)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.shape == (3, 5, 2)

    def test_asarray_dtype_preservation(self):
        """Test that asarray preserves dtypes correctly"""
        dtypes = [
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]

        for dtype in dtypes:
            x = np.arange(12, dtype=dtype).reshape(3, 4)
            y = x.transpose()
            res_np = np.asarray(y)
            res_num = num.asarray(y)
            assert strict_type_equal(res_np, res_num)
            assert res_num.dtype == dtype

    def test_asarray_edge_cases(self):
        """Test asarray with edge cases"""
        # Zero-sized array
        x1 = np.zeros((0, 0, 0))
        res_np1 = np.asarray(x1)
        res_num1 = num.asarray(x1)
        assert strict_type_equal(res_np1, res_num1)
        assert res_num1.shape == (0, 0, 0)

        # Arrays with zero strides
        x4 = np.zeros((3, 4))
        y4 = x4[0:1, :]  # This creates a view with zero stride in first dim
        res_np4 = np.asarray(y4)
        res_num4 = num.asarray(y4)
        assert strict_type_equal(res_np4, res_num4)
        assert res_num4.shape == (1, 4)

    def test_asarray_with_dtype_conversion(self):
        """Test asarray with dtype conversion"""
        x = np.arange(12, dtype=np.int32).reshape(3, 4)
        y = x.transpose()

        # Convert to different dtype
        res_np = np.asarray(y, dtype=np.float64)
        res_num = num.asarray(y, dtype=np.float64)
        assert strict_type_equal(res_np, res_num)
        assert res_num.dtype == np.float64
        assert res_num.shape == (4, 3)

        # Convert complex array
        x2 = np.arange(12, dtype=np.complex64).reshape(3, 4)
        y2 = x2.transpose()
        res_np2 = np.asarray(y2, dtype=np.complex128)
        res_num2 = num.asarray(y2, dtype=np.complex128)
        assert strict_type_equal(res_np2, res_num2)
        assert res_num2.dtype == np.complex128

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
