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
from utils.comparisons import allclose
from utils.utils import AxisError

import cupynumeric as num
from cupynumeric._utils import is_np2

# numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<no value>,
# initial=<no value>, where=<no value>)

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
SIZES_E2 = [
    (DIM),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]
SIZE_E = [
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
]
NO_EMPTY_SIZE = [
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

ARR = ([], [[]], [[], []], np.inf, -10.3, 0, 200, 5 + 8j)

DTYPE = ("l", "L", "f", "e", "d")
INTEGER_DTYPE = ("h", "i", "H", "I", "?", "b", "B")


def to_dtype(s):
    return str(np.dtype(s))


class TestProdNegative(object):
    """
    this class is to test negative cases
    """

    @pytest.mark.parametrize("arr", ARR)
    def test_array(self, arr):
        assert allclose(np.prod(arr), num.prod(arr))

    def test_axis_out_bound(self):
        expected_exc = AxisError
        arr = [-1, 0, 1, 2, 10]
        with pytest.raises(expected_exc):
            np.prod(arr, axis=2)
        with pytest.raises(expected_exc):
            num.prod(arr, axis=2)

    def test_out_negative(self):
        expected_exc = ValueError
        in_shape = (2, 3, 4)
        out_shape = (2, 3, 3)
        arr_np = np.ndarray(in_shape)
        out_np = np.ndarray(out_shape)
        arr_num = num.ndarray(in_shape)
        out_num = num.ndarray(out_shape)
        with pytest.raises(expected_exc):
            np.prod(arr_np, out=out_np, axis=2)
        with pytest.raises(expected_exc):
            num.prod(arr_num, out=out_num, axis=2)

    def test_keepdims(self):
        in_shape = (2, 3, 4)
        arr_num = num.random.random(in_shape) * 10
        arr_np = np.array(arr_num)
        out_np = np.prod(arr_np, axis=2, keepdims=True)
        out_num = num.prod(arr_num, axis=2, keepdims=True)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("initial", ([2, 3], [3]), ids=str)
    def test_initial_list(self, initial):
        expected_exc = TypeError if is_np2 else ValueError
        arr = [[1, 2], [3, 4]]
        with pytest.raises(expected_exc):
            np.prod(arr, initial=initial)
        with pytest.raises(expected_exc):
            num.prod(arr, initial=initial)

    def test_initial_empty_array(self):
        size = (1, 0)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        initial_value = np.random.uniform(-20.0, 20.0)
        out_num = num.prod(arr_num, initial=initial_value)
        out_np = np.prod(arr_np, initial=initial_value)
        assert allclose(out_np, out_num)

    def test_where(self):
        arr = [[1, 2], [3, 4]]
        out_np = np.prod(arr, where=[False, True])
        out_num = num.prod(arr, where=[False, True])
        assert allclose(out_np, out_num)

        # where is boolean
        out_np = np.prod(arr, where=True)
        out_num = num.prod(arr, where=True)
        assert allclose(out_np, out_num)

        out_np = np.prod(arr, where=False)
        out_num = num.prod(arr, where=False)
        assert allclose(out_np, out_num)


class TestProdPositive(object):
    """
    this class is to test positive cases
    """

    @pytest.mark.parametrize("size", SIZES)
    def test_basic(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)
        out_num = np.prod(arr_num)
        assert allclose(out_np, out_num)
        assert allclose(out_num, arr_num.prod())

    @pytest.mark.parametrize("dtype", DTYPE, ids=to_dtype)
    def test_dtype(self, dtype):
        size = (5, 5, 5)
        arr = np.random.random(size) * 10
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np)
        out_num = num.prod(arr_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("dtype", INTEGER_DTYPE, ids=to_dtype)
    def test_dtype_integer_precision(self, dtype):
        arr_np = np.arange(0, 5).astype(dtype)
        arr_num = num.arange(0, 5).astype(dtype)
        out_np = np.prod(arr_np)
        out_num = num.prod(arr_num)
        assert allclose(out_num, arr_num.prod())
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize(
        "dtype",
        (
            "F",
            pytest.param("D", marks=pytest.mark.xfail),
            pytest.param("G", marks=pytest.mark.xfail),
        ),
        ids=to_dtype,
    )
    def test_dtype_complex(self, dtype):
        arr = (np.random.rand(5, 5) * 10 + 2) + (
            np.random.rand(5, 5) * 10 * 1.0j + 0.2j
        )
        arr_np = np.array(arr, dtype=dtype)
        arr_num = num.array(arr, dtype=dtype)
        out_np = np.prod(arr_np)
        # cupynumeric always returns [1+0.j] when LEGATE_TEST=1
        out_num = num.prod(arr_num)
        # When running tests with CUPYNUMERIC_TEST=1 and dtype is complex256,
        # allclose hits assertion error:
        # File "/legate/cupynumeric/cupynumeric/eager.py", line 293,
        # in to_deferred_array
        #   assert self.runtime.is_supported_dtype(self.array.dtype)
        #   AssertionError
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("axis", (_ for _ in range(-2, 3, 1)))
    def test_axis_basic(self, axis):
        size = (5, 5, 5)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        out_num = num.prod(arr_num, axis=axis)
        out_np = np.prod(arr_np, axis=axis)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize(
        "axis", ((-1, 1), (0, 1), (1, 2), (0, 2)), ids=str
    )
    def test_axis_tuple(self, axis):
        size = (5, 5, 5)
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        out_np = np.prod(arr_np, axis=axis)
        out_num = num.prod(arr_num, axis=axis)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_out_basic(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        out_np = np.random.random(())
        out_num = num.random.random(())
        np.prod(arr_np, out=out_np)
        num.prod(arr_num, out=out_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_out_axis(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_shape = ()
            if isinstance(size, tuple):
                out_shape_list = list(size)
                del out_shape_list[axis]
                out_shape = tuple(out_shape_list)
            out_np = np.random.random(out_shape)
            out_num = num.random.random(out_shape)
            np.prod(arr_np, out=out_np, axis=axis)
            num.prod(arr_num, out=out_num, axis=axis)
            assert allclose(out_np, out_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", SIZES_E2)
    def test_out_axis_dtype(self, size):
        arr = np.random.random(size) * 10
        arr_np = np.array(arr, dtype=to_dtype("f"))
        arr_num = num.array(arr, dtype=to_dtype("f"))

        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_shape = ()
            if isinstance(size, tuple):
                out_shape_list = list(size)
                del out_shape_list[axis]
                out_shape = tuple(out_shape_list)
            out = np.random.random(out_shape)

            out_np = np.array(out, dtype=to_dtype("i"))
            out_num = num.array(out, dtype=to_dtype("i"))

            np.prod(arr_np, out=out_np, axis=axis)
            num.prod(arr_num, out=out_num, axis=axis)

            assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZES)
    def test_axis_keepdims_false(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_np = np.prod(arr_np, axis=axis, keepdims=False)
            out_num = num.prod(arr_num, axis=axis, keepdims=False)
            assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", SIZE_E)
    def test_axis_keepdims_true(self, size):
        arr_np = np.random.random(size)
        arr_num = num.array(arr_np)
        ndim = arr_np.ndim
        for axis in range(-ndim + 1, ndim, 1):
            out_np = np.prod(arr_np, axis=axis, keepdims=True)
            out_num = num.prod(arr_num, axis=axis, keepdims=True)
            assert allclose(out_np, out_num)

    @pytest.mark.parametrize("size", NO_EMPTY_SIZE)
    def test_initial(self, size):
        arr_np = np.random.random(size) * 10
        arr_num = num.array(arr_np)
        initial_value = np.random.uniform(-20.0, 20.0)
        out_num = num.prod(arr_num, initial=initial_value)
        out_np = np.prod(arr_np, initial=initial_value)

        assert allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
