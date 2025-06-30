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
    res_np = np.nanmean(arr)
    res_num = num.nanmean(arr)
    assert np.isnan(res_np) and np.isnan(res_num)


@pytest.mark.parametrize("val", (np.nan, 0.0, 10.0, -5, 1 + 1j))
def test_scalar(val):
    res_np = np.nanmean(val)
    res_num = num.nanmean(val)
    assert np.array_equal(res_np, res_num, equal_nan=True)


@pytest.mark.parametrize("val", (np.nan, 0.0, 10.0, -5, 1 + 1j))
def test_scalar_where(val):
    res_np = np.nanmean(val, where=True)
    res_num = num.nanmean(val, where=True)
    assert np.array_equal(res_np, res_num, equal_nan=True)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic(size):
    arr_np = np.random.randint(-5, 5, size=size).astype(float)
    arr_np[arr_np % 2 == 0] = np.nan
    arr_num = num.array(arr_np)
    res_np = np.nanmean(arr_np)
    res_num = num.nanmean(arr_num)
    assert np.array_equal(res_np, res_num, equal_nan=True)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_basic_where(size):
    arr_np = np.random.randint(-5, 5, size=size).astype(float)
    arr_np[arr_np % 2 == 0] = np.nan
    arr_num = num.array(arr_np)
    where_np = arr_np % 2
    where_np = arr_np.astype(bool)
    where_num = num.array(where_np)
    res_np = np.nanmean(arr_np, where=where_np)
    res_num = num.nanmean(arr_num, where=where_num)
    assert np.array_equal(res_np, res_num, equal_nan=True)


@pytest.mark.xfail
@pytest.mark.parametrize("axis", ((-3, -1), (-1, 0), (-2, 2), (0, 2)))
def test_axis_tuple(axis):
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    size = (3, 4, 7)
    arr_np = np.random.randint(-5, 5, size=size).astype(float)
    arr_np[arr_np % 2 == 1] = np.nan
    arr_num = num.array(arr_np)
    out_np = np.nanmean(arr_np, axis=axis)
    out_num = num.nanmean(arr_num, axis=axis)
    assert np.array_equal(out_np, out_num, equal_nan=True)


@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_axis_keepdims(size, keepdims):
    arr_np = np.random.randint(-5, 5, size=size).astype(float)
    arr_np[arr_np % 2 == 1] = np.nan
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in range(-ndim, ndim):
        out_np = np.nanmean(arr_np, axis=axis, keepdims=keepdims)
        out_num = num.nanmean(arr_num, axis=axis, keepdims=keepdims)
        assert np.array_equal(out_np, out_num, equal_nan=True)


@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_axis_where(size):
    arr_np = np.random.randint(-5, 5, size=size).astype(float)
    arr_np[arr_np % 2 == 0] = np.nan
    arr_num = num.array(arr_np)
    where_np = arr_np[arr_np % 2 == 1] % 2
    where_np = arr_np.astype(bool)
    where_num = num.array(where_np)
    ndim = arr_np.ndim
    for axis in range(-ndim, ndim):
        out_np = np.nanmean(arr_np, axis=axis, where=where_np)
        out_num = num.nanmean(arr_num, axis=axis, where=where_num)
        assert np.array_equal(out_np, out_num, equal_nan=True)


@pytest.mark.parametrize("out_dt", (np.float32, np.complex128))
@pytest.mark.parametrize("size", NO_EMPTY_SIZE)
def test_out(size, out_dt):
    arr_np = np.random.randint(-5, 5, size=size).astype(float)
    arr_np[arr_np % 2 == 0] = np.nan
    arr_num = num.array(arr_np)
    ndim = arr_np.ndim
    for axis in (-1, ndim - 1, None):
        out_shape = gen_out_shape(size, axis)
        out_np = np.empty(out_shape, dtype=out_dt)
        out_num = num.empty(out_shape, dtype=out_dt)
        np.nanmean(arr_np, axis=axis, out=out_np)
        num.nanmean(arr_num, axis=axis, out=out_num)
        np.array_equal(out_np, out_num, equal_nan=True)


def test_nanmean_integer_dtype() -> None:
    arr_num = num.array([1, 2, 3, 4])
    arr_np = np.array([1, 2, 3, 4])
    res_num = num.nanmean(arr_num, dtype=np.int32)
    res_np = np.nanmean(arr_np, dtype=np.int32)
    assert res_num == res_np


def test_nanmean_bool_dtype() -> None:
    arr_np = np.array([True, False, True])
    arr_num = num.array([True, False, True])
    res_num = num.nanmean(arr_num)
    res_np = np.mean(arr_np)
    assert res_num == res_np


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
