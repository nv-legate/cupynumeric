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
    1,
    DIM,
    (0,),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, DIM - 1, DIM),
    (2, DIM - 1, DIM),
    (DIM, DIM, DIM),
]


class TestModule:
    @pytest.mark.xfail
    def test_none_array_compare(self):
        res_num = num.transpose(None)  # AttributeError: 'NoneType'
        res_np = np.transpose(None)  # return None
        assert np.array_equal(res_num, res_np, equal_nan=True)

    def test_none_array(self):
        # numpy returned None
        msg = r"NoneType"
        with pytest.raises(AttributeError, match=msg):
            num.transpose(None)

    @pytest.mark.parametrize(
        "axes", ((1, 1, 1), (1, 2, 3), (1, 2), (1, 2, 0, 1))
    )
    def test_invalid_axis(self, axes):
        size = (2, 3, 4)
        a = num.random.randint(low=-10, high=10, size=size)
        with pytest.raises(ValueError):
            num.transpose(a, axes=axes)

    def test_int_axis(self):
        size = (2, 3, 4)
        a = num.random.randint(low=-10, high=10, size=size)
        # numpy raises "ValueError: axes don't match array".
        # cupynumeric raises "TypeError".
        with pytest.raises(TypeError):
            num.transpose(a, axes=2)

    @pytest.mark.xfail
    def test_int_axis_compare(self):
        size = (2, 3, 4)
        a = num.random.randint(low=-10, high=10, size=size)
        # numpy raises "ValueError: axes don't match array".
        # cupynumeric raises "TypeError".
        with pytest.raises(ValueError):
            num.transpose(a, axes=2)

    @pytest.mark.parametrize("size", SIZES, ids=str)
    def test_round(self, size):
        a = num.random.randint(low=-10, high=10, size=size)
        b = num.transpose(a)
        c = num.transpose(b)
        assert num.array_equal(c, a)

    @pytest.mark.parametrize("size", SIZES, ids=str)
    def test_basic(self, size):
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = np.transpose(a)
        res_num = num.transpose(b)
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", (0, 1, DIM))
    def test_axes_1d(self, size):
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = np.transpose(a, axes=0)
        res_num = num.transpose(b, axes=0)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", (0, 1, DIM))
    @pytest.mark.parametrize("axes", (-3, 3))
    def test_axes_1d_int(self, size, axes):
        # For cupynumeric, if array.dim==1, it returns the array itself
        # directly, no matter what the axes value is.
        # For numpy, it raises
        # "AxisError: axis * is out of bounds for array of dimension 1".
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = np.transpose(a, axes=axes)
        res_num = num.transpose(b, axes=axes)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", (0, 1, DIM))
    @pytest.mark.parametrize("axes", ((1,), (3, 1)))
    def test_axes_1d_tuple(self, size, axes):
        # For cupynumeric, if array.dim==1, it returns the array itself
        # directly, no matter what the axes value is.
        # For numpy, it raises "ValueError: axes don't match array".
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = np.transpose(a, axes=axes)
        res_num = num.transpose(b, axes=axes)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.parametrize(
        "size",
        ((1, 0), (1, 1), (1, DIM), (DIM, 1), (DIM - 1, DIM - 2), (DIM, DIM)),
    )
    @pytest.mark.parametrize("axes", ((0, 1), (1, 0)))
    def test_axes_2d(self, size, axes):
        a = num.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = np.transpose(a, axes=axes)
        res_num = num.transpose(b, axes=axes)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.parametrize(
        "size",
        (
            (1, 0, 1),
            (1, 1, 1),
            (DIM, DIM - 1, 1),
            (1, 1, DIM),
            (2, 3, 4),
            (DIM, DIM, DIM),
        ),
    )
    @pytest.mark.parametrize(
        "axes", ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    )
    def test_axes_3d(self, size, axes):
        a = num.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = np.transpose(a, axes=axes)
        res_num = num.transpose(b, axes=axes)
        assert num.array_equal(res_num, res_np)


class TestArrayMethod:
    @pytest.mark.parametrize(
        "axes", ((1, 1, 1), (1, 2, 3), (1, 2), (1, 2, 0, 1))
    )
    def test_invalid_axis(self, axes):
        size = (2, 3, 4)
        a = num.random.randint(low=-10, high=10, size=size)
        with pytest.raises(ValueError):
            a.transpose(axes=axes)

    def test_int_axis(self):
        size = (2, 3, 4)
        a = num.random.randint(low=-10, high=10, size=size)
        # numpy raises "ValueError: axes don't match array".
        # cupynumeric raises "TypeError".
        with pytest.raises(TypeError):
            a.transpose(axes=2)

    @pytest.mark.xfail
    def test_int_axis_compare(self):
        size = (2, 3, 4)
        a = num.random.randint(low=-10, high=10, size=size)
        # numpy raises "ValueError: axes don't match array".
        # cupynumeric raises "TypeError".
        with pytest.raises(ValueError):
            a.transpose(axes=2)

    @pytest.mark.parametrize("size", SIZES, ids=str)
    def test_round(self, size):
        a = num.random.randint(low=-10, high=10, size=size)
        b = a.transpose()
        c = b.transpose()
        assert num.array_equal(c, a)

    @pytest.mark.parametrize("size", SIZES, ids=str)
    def test_basic(self, size):
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = a.transpose()
        res_num = b.transpose()
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", (0, 1, DIM))
    def test_axes_1d(self, size):
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = a.transpose(0)
        res_num = b.transpose(0)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", (0, 1, DIM))
    @pytest.mark.parametrize("axes", (-3, 3))
    def test_axes_1d_int(self, size, axes):
        # For cupynumeric, if array.dim==1, it returns the array itself
        # directly, no matter what the axes value is.
        # For Numpy, it raises
        # "AxisError: axis * is out of bounds for array of dimension 1".
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = a.transpose(axes)
        res_num = b.transpose(axes)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.xfail
    @pytest.mark.parametrize("size", (0, 1, DIM))
    @pytest.mark.parametrize("axes", ((1,), (3, 1)))
    def test_axes_1d_tuple(self, size, axes):
        # For cupynumeric, if array.dim==1, it returns the array itself
        # directly, no matter what the axes value is.
        # For Numpy, it raises "ValueError: axes don't match array".
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = a.transpose(axes)
        res_num = b.transpose(axes)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.parametrize(
        "size",
        ((1, 0), (1, 1), (1, DIM), (DIM, 1), (DIM - 1, DIM - 2), (DIM, DIM)),
    )
    @pytest.mark.parametrize("axes", ((0, 1), (1, 0)))
    def test_axes_2d(self, size, axes):
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = a.transpose(axes)
        res_num = b.transpose(axes)
        assert num.array_equal(res_num, res_np)

    @pytest.mark.parametrize(
        "size",
        (
            (1, 0, 1),
            (1, 1, 1),
            (DIM, DIM - 1, 1),
            (1, 1, DIM),
            (2, 3, 4),
            (DIM, DIM, DIM),
        ),
    )
    @pytest.mark.parametrize(
        "axes", ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    )
    def test_axes_3d(self, size, axes):
        a = np.random.randint(low=-10, high=10, size=size)
        b = num.array(a)
        res_np = a.transpose(axes)
        res_num = b.transpose(axes)
        assert num.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
