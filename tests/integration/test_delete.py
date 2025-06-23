# Copyright 2025 NVIDIA Corporation
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

import re

import numpy as np
import pytest
from utils.comparisons import allclose
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num

# Scalars


@pytest.mark.parametrize("obj", [-2, 1, 2])
def test_scalar_index_error(obj):
    message = f"index {obj} is out of bounds for axis 0 with size 1"
    with pytest.raises(
        IndexError,
        match=message,
    ):
        np.delete(5, obj)
    with pytest.raises(
        IndexError,
        match=message,
    ):
        num.delete(5, obj)


@pytest.mark.parametrize("obj", [-1, 0])
def test_scalar_negative_index(obj):
    res_np = np.delete(3, obj)
    res_num = num.delete(3, obj)
    assert allclose(res_np, res_num)


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_scalar_axis_error(axis):
    message = f"axis {axis} is out of bounds for array of dimension 0"
    with pytest.raises(
        IndexError,
        match=message,
    ):
        np.delete(5, 0, axis=axis)
    with pytest.raises(
        IndexError,
        match=message,
    ):
        num.delete(5, 0, axis=axis)


# Empty


@pytest.mark.parametrize("obj", [-1, 0, 1, 2])
def test_empty(obj):
    message = f"index {obj} is out of bounds for axis 0 with size 0"
    with pytest.raises(
        IndexError,
        match=message,
    ):
        np.delete([], obj)
    with pytest.raises(
        IndexError,
        match=message,
    ):
        num.delete([], obj)


def test_empty_index():
    arr = np.array([1, 2, 3])
    arr_num = num.array(arr)
    indices = []

    res_np = np.delete(arr, indices)
    res_num = num.delete(arr_num, indices)
    assert allclose(res_np, res_num)


def test_empty_with_axis():
    arr = np.empty((0, 3))
    arr_num = num.array(arr)

    message = "index 0 is out of bounds for axis 0 with size 0"
    with pytest.raises(IndexError, match=message):
        np.delete(arr, 0, axis=0)
    with pytest.raises(IndexError, match=message):
        num.delete(arr_num, 0, axis=0)


# Indexing


def test_batch_delete():
    arr = mk_seq_array(np, (1000,))
    arr_num = mk_seq_array(num, (1000,))
    indices = [0, 2, 5, 7, 100, 500, 999]

    res_np = np.delete(arr, indices)
    res_num = num.delete(arr_num, indices)
    assert allclose(res_np, res_num)


def test_duplicate_indices():
    arr = mk_seq_array(np, (1000,))
    arr_num = mk_seq_array(num, (1000,))
    indices = [0, 0, 2, 5, 5, 100, 100, 999]

    res_np = np.delete(arr, indices)
    res_num = num.delete(arr_num, indices)
    assert allclose(res_np, res_num)


# Shape


def test_row():
    arr = mk_seq_array(np, (200, 300))
    arr_num = mk_seq_array(num, (200, 300))
    index = 0

    res_np = np.delete(arr, index)
    res_num = num.delete(arr_num, index)
    assert allclose(res_np, res_num)


def test_column():
    arr = mk_seq_array(np, (200, 300))
    arr_num = mk_seq_array(num, (200, 300))
    index = 1

    res_np = np.delete(arr, index)
    res_num = num.delete(arr_num, index)
    assert allclose(res_np, res_num)


def test_non_array_index():
    arr = mk_seq_array(np, (1000,))
    arr_num = mk_seq_array(num, (1000,))
    index = 1

    res_np = np.delete(arr, index)
    res_num = num.delete(arr_num, index)
    assert allclose(res_np, res_num)


def test_boolean_mask():
    arr = mk_seq_array(np, (8,))
    arr_num = mk_seq_array(num, (8,))
    mask = [True, False, True, False, True, False, True, False]

    res_np = np.delete(arr, mask)
    res_num = num.delete(arr_num, mask)
    assert allclose(res_np, res_num)


# Test that is returns a new arr


def test_bool_mask_returns_copy():
    a = mk_seq_array(num, (8,))
    mask = num.array([False, True, False, False, True, False, False, True])
    b = num.delete(a, mask)
    assert not np.shares_memory(a, b)


def test_row_returns_copy():
    a = mk_seq_array(num, (200, 300))
    b = num.delete(a, 0)
    assert not np.shares_memory(a, b)


def test_column_returns_copy():
    a = mk_seq_array(num, (200, 300))
    b = num.delete(a, 1)
    assert not np.shares_memory(a, b)


def test_non_array_index_returns_copy():
    a = mk_seq_array(num, (1000,))
    b = num.delete(a, 1)
    assert not np.shares_memory(a, b)


# Other


def test_axis_negative():
    arr = mk_seq_array(np, (1000, 10, 5))
    arr_num = mk_seq_array(num, (1000, 10, 5))

    res_np = np.delete(arr, 1, axis=-1)
    res_num = num.delete(arr_num, 1, axis=-1)
    assert allclose(res_np, res_num)


def test_boolean_mask_wrong_size():
    arr = mk_seq_array(np, (8,))
    arr_num = mk_seq_array(num, (8,))
    mask = [True, False, True, False, True, False, True, False, True, False]
    message = (
        "boolean array argument obj to delete must be one dimensional"
        " and match the axis length of 8"
    )
    with pytest.raises(
        ValueError,
        match=message,
    ):
        np.delete(arr, mask)
    with pytest.raises(
        ValueError,
        match=message,
    ):
        num.delete(arr_num, mask)


def test_out_of_bounds_boolean_mask():
    arr = mk_seq_array(np, (8,))
    arr_num = mk_seq_array(num, (8,))
    mask = [True, False, True, False, True, False, True, False, True, False]

    message = (
        "boolean array argument obj to delete must be one dimensional"
        " and match the axis length of 8"
    )
    with pytest.raises(
        ValueError,
        match=message,
    ):
        np.delete(arr, mask)
    with pytest.raises(
        ValueError,
        match=message,
    ):
        num.delete(arr_num, mask)


def test_non_integer_indices():
    arr = mk_seq_array(np, (1000,))
    arr_num = mk_seq_array(num, (1000,))
    indices = [0.5, 1.5]

    with pytest.raises(
        IndexError,
        match=re.escape(
            "arrays used as indices must be of integer (or boolean) type"
        ),
    ):
        np.delete(arr, indices)
    with pytest.raises(
        TypeError, match="index arrays should be int or bool type"
    ):
        num.delete(arr_num, indices)


def test_all_indices():
    arr = mk_seq_array(np, (1000,))
    arr_num = mk_seq_array(num, (1000,))
    indices = list(range(1000))

    res_np = np.delete(arr, indices)
    res_num = num.delete(arr_num, indices)
    assert allclose(res_np, res_num)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_ndim_axis(ndim, axis):
    shape = tuple(4 for _ in range(ndim))
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)

    res_np = np.delete(arr_np, 1, axis=axis)
    res_num = num.delete(arr_num, 1, axis=axis)
    assert allclose(res_np, res_num)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_slice_obj(ndim, axis):
    shape = tuple(4 for _ in range(ndim))
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    # Use a slice to delete a range of elements along the axis
    obj = slice(1, 3)
    res_np = np.delete(arr_np, obj, axis=axis)
    res_num = num.delete(arr_num, obj, axis=axis)
    assert allclose(res_np, res_num)


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_slice_out_of_bounds(axis):
    arr = mk_seq_array(np, (1000, 1000))
    arr_num = mk_seq_array(num, (1000, 1000))
    res_np = np.delete(arr, slice(1500, 2000, 1), axis=axis)
    res_num = num.delete(arr_num, slice(1500, 2000, 1), axis=axis)
    assert allclose(res_np, res_num)


def test_slice_axis_out_of_bounds():
    arr_np = mk_seq_array(np, (1000, 1000))
    arr_num = mk_seq_array(num, (1000, 1000))
    obj = slice(1, 2)
    axis = 2

    with pytest.raises(
        IndexError, match="axis 2 is out of bounds for array of" " dimension 2"
    ):
        np.delete(arr_np, obj, axis=axis)
    with pytest.raises(
        IndexError, match="axis 2 is out of bounds for array of" " dimension 2"
    ):
        num.delete(arr_num, obj, axis=axis)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
