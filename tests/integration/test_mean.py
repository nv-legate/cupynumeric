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

DIM = 7

NO_EMPTY_SIZE = (
    (1,),
    (DIM,),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
)


def gen_out_shape(size, axis):
    if axis is None:
        return ()
    if axis < 0:
        axis += len(size)
    if axis >= 0 and axis < len(size):
        return size[:axis] + size[axis + 1 :]
    else:
        return -1


@pytest.mark.parametrize("arr", ([], [[], []]))
def test_empty_arr(arr):
    res_np = np.mean(arr)
    res_num = num.mean(arr)
    assert np.isnan(res_np) and np.isnan(res_num)


@pytest.mark.parametrize("val", (0.0, 10.0, -5, 1 + 1j))
def test_scalar(val):
    res_np = np.mean(val)
    res_num = num.mean(val)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("val", (0.0, 10.0, -5, 1 + 1j))
def test_scalar_where(val):
    res_np = np.mean(val, where=True)
    res_num = num.mean(val, where=True)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    res_np = np.mean(arr_np)
    res_num = num.mean(arr_num)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic_where(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    where_np = arr_np % 2
    where_np = arr_np.astype(bool)
    where_num = num.array(where_np)
    res_np = np.mean(arr_np, where=where_np)
    res_num = num.mean(arr_num, where=where_num)
    assert np.array_equal(res_np, res_num, equal_nan=True)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_where_broadcast(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    where_np = np.zeros((1,), bool)
    where_num = num.array(where_np)
    res_np = np.mean(arr_np, where=where_np)
    res_num = num.mean(arr_num, where=where_num)
    assert np.array_equal(res_np, res_num, equal_nan=True)


@pytest.mark.xfail
@pytest.mark.parametrize("axis", ((-3, -1), (-1, 0), (-2, 2), (0, 2)))
def test_axis_tuple(axis):
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    size = (3, 4, 7)
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    out_np = np.mean(arr_np, axis=axis)
    out_num = num.mean(arr_num, axis=axis)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_axis_keepdims(size, keepdims):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim, ndim):
        out_np = np.mean(arr_np, axis=axis, keepdims=keepdims)
        out_num = num.mean(arr_num, axis=axis, keepdims=keepdims)
        assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_axis_where(size):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    where_np = arr_np % 2
    where_np = arr_np.astype(bool)
    where_num = num.array(where_np)
    ndim = arr_np.ndim
    for axis in range(-ndim, ndim):
        out_np = np.mean(arr_np, axis=axis, where=where_np)
        out_num = num.mean(arr_num, axis=axis, where=where_num)
        assert np.array_equal(out_np, out_num, equal_nan=True)


@pytest.mark.parametrize("array_dt", (np.int32, np.float32, np.complex64))
@pytest.mark.parametrize("dt", (np.int32, np.float32, np.complex64))
@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_dtype(size, array_dt, dt):
    arr_np = np.random.randint(-5, 5, size=size).astype(array_dt)
    arr_num = num.array(arr_np)
    res_np = np.mean(arr_np, dtype=dt)
    res_num = num.mean(arr_num, dtype=np.dtype(dt))
    np.array_equal(res_np, res_num)


@pytest.mark.parametrize("out_dt", (np.int32, np.float32, np.complex128))
@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_out(size, out_dt):
    arr_np = np.random.randint(-5, 5, size=size)
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in (-1, ndim - 1, None):
        out_shape = gen_out_shape(size, axis)
        out_np = np.empty(out_shape, dtype=out_dt)
        out_num = num.empty(out_shape, dtype=out_dt)
        np.mean(arr_np, axis=axis, out=out_np)
        num.mean(arr_num, axis=axis, out=out_num)
        np.array_equal(out_np, out_num)


class TestMeanErrors:
    def setup_method(self):
        size = (3, 4, 5)
        self.arr_np = np.random.randint(-5, 5, size=size)
        self.arr_num = num.array(self.arr_np)

    @pytest.mark.parametrize(
        "axis", (-4, 3), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound(self, axis):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.mean(self.arr_np, axis=axis)
        with pytest.raises(expected_exc):
            num.mean(self.arr_num, axis=axis)

    @pytest.mark.parametrize(
        "axis_out_shape",
        ((None, (1,)), (1, (3, 4))),
        ids=lambda axis_out_shape: f"(axis_out_shape={axis_out_shape})",
    )
    def test_out_invalid_shape(self, axis_out_shape):
        axis, out_shape = axis_out_shape
        expected_exc = ValueError
        out_np = np.empty(out_shape)
        out_num = num.empty(out_shape)
        with pytest.raises(expected_exc):
            np.mean(self.arr_np, axis=axis, out=out_np)
        with pytest.raises(expected_exc):
            num.mean(self.arr_num, axis=axis, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
