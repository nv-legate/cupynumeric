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
from utils.generators import mk_seq_array
from utils.utils import AxisError

import cupynumeric as num


@pytest.mark.parametrize(
    "array", (None, [], 4, [2, 3], mk_seq_array(num, (3, 4, 2)))
)
def test_repeats_none(array):
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        np.repeat(array, None)
    with pytest.raises(expected_exc):
        num.repeat(array, None)


@pytest.mark.parametrize("repeats", (-3, 0, 3, 4.7, [], [-3], [0], [3], [4.7]))
def test_array_empty_repeats_valid(repeats):
    res_np = np.repeat([], repeats)
    res_num = num.repeat([], repeats)
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
@pytest.mark.parametrize("repeats", ([3, 4], [1, 2, 3]))
def test_array_empty_repeats_invalid_negative(repeats):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat([], repeats)
        # numpy raises:  ValueError: operands could not be broadcast
        # together with shape (0,) (2,)
    with pytest.raises(expected_exc):
        num.repeat([], repeats)
        # while cupynumeric is pass with the result []


@pytest.mark.xfail
@pytest.mark.parametrize("repeats", ([3, 4], [1, 2, 3]))
def test_array_empty_repeats_invalid(repeats):
    res_np = np.repeat([], repeats)
    res_num = num.repeat([], repeats)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("repeats", (-3, 0, 3, 4.7, [], [-3], [0], [3], [4.7]))
def test_array_empty_axis_valid(repeats):
    res_np = np.repeat([], repeats, axis=0)
    res_num = num.repeat([], repeats, axis=0)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("repeats", (-3, 0, 3, 4.7, [], [-3], [0], [3], [4.7]))
def test_array_empty_axis_invalid(repeats):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat([], repeats, axis=1)
    with pytest.raises(expected_exc):
        num.repeat([], repeats, axis=1)


@pytest.mark.parametrize("axis", (-3, 3))
def test_array_int_axis_negative(axis):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(3, 3, axis=axis)
    with pytest.raises(expected_exc):
        num.repeat(3, 3, axis=axis)


@pytest.mark.parametrize("repeats", (-3, [-3]))
def test_array_int_repeats_negative(repeats):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(3, repeats)
    with pytest.raises(expected_exc):
        num.repeat(3, repeats)


@pytest.mark.parametrize("repeats", (0, 3, 4.7, [0], [3], [4.7]))
def test_array_int_repeats_valid(repeats):
    res_np = np.repeat(3, repeats)
    res_num = num.repeat(3, repeats)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("repeats", ([], [1, 2]))
def test_array_int_repeats_invalid(repeats):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(3, repeats)
    with pytest.raises(expected_exc):
        num.repeat(3, repeats)


@pytest.mark.parametrize("repeats", (0, 3, 4.7, [0], [3], [4.7], [2, 3, 4]))
def test_array_1d_repeats_valid(repeats):
    anp = np.array([1, 2, 3])
    res_np = np.repeat(anp, repeats)
    res_num = num.repeat(anp, repeats)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("repeats", ([], [2, 3]))
def test_array_1d_repeats_invalid(repeats):
    anp = np.array([1, 2, 3])
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(anp, repeats)
    with pytest.raises(expected_exc):
        num.repeat(anp, repeats)


@pytest.mark.parametrize("repeats", (0, [0], 3, 4.7, [3], [4.7]))
def test_array_2d_repeats_valid(repeats):
    anp = np.array([[1, 3], [2, 4]])
    res_np = np.repeat(anp, repeats)
    res_num = num.repeat(anp, repeats)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("repeats", ([], [2, 3]))
def test_array_2d_repeats_invalid(repeats):
    anp = np.array([[1, 3], [2, 4]])
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(anp, repeats)
    with pytest.raises(expected_exc):
        num.repeat(anp, repeats)


@pytest.mark.skip()
@pytest.mark.parametrize("arr", ([1, 2, 3], [[1, 3], [2, 4]]))
@pytest.mark.parametrize("repeats", (-3, [-3]))
def test_array_1d_repeats_fatal_error(arr, repeats):
    anp = np.array(arr)
    anum = num.array(arr)
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(anp, repeats)
        # numpy raises "ValueError: negative dimensions are not allowed"
    with pytest.raises(expected_exc):
        num.repeat(anum, repeats)
        # cuPyNumeric got "Fatal Python error: Aborted"


@pytest.mark.parametrize("arr", (None, [], 3, [1, 2, 3], [[1, 3], [2, 4]]))
@pytest.mark.parametrize(
    "repeats",
    ([[2, 3], [3, 3]], np.random.randint(low=-10.0, high=10, size=(3, 3, 3))),
)
def test_repeats_nd(arr, repeats):
    anp = np.array(arr)
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.repeat(anp, repeats)
    with pytest.raises(expected_exc):
        num.repeat(anp, repeats)


@pytest.mark.parametrize(("arr", "repeats"), ((3, 3), ([1, 2, 3], [1, 2, 3])))
@pytest.mark.parametrize("axis", ("hello", 0.9))
def test_axis_string(arr, repeats, axis):
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        np.repeat(arr, repeats, axis=axis)
    with pytest.raises(expected_exc):
        num.repeat(arr, repeats, axis=axis)


def test_array_axis_out_bound():
    anp = np.array([1, 2, 3, 4, 5])
    expected_exc = AxisError
    with pytest.raises(expected_exc):
        np.repeat(anp, 4, 2)
    with pytest.raises(expected_exc):
        num.repeat(anp, 4, 2)


def test_array_axis_negative_equal():
    anp = np.array([1, 2, 3, 4, 5])
    res_np = np.repeat(anp, 4, -1)
    res_num = num.repeat(anp, 4, -1)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "ndim",
    (
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_nd_basic(ndim):
    a_shape = tuple(np.random.randint(1, 9) for _ in range(ndim))
    np_array = mk_seq_array(np, a_shape)
    num_array = mk_seq_array(num, a_shape)
    repeats = np.random.randint(0, 15)
    res_num = num.repeat(num_array, repeats)
    res_np = np.repeat(np_array, repeats)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "ndim",
    (
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_nd_axis(ndim):
    for axis in range(0, ndim):
        a_shape = tuple(np.random.randint(1, 9) for _ in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        repeats = np.random.randint(0, 15)
        res_num2 = num.repeat(num_array, repeats, axis)
        res_np2 = np.repeat(np_array, repeats, axis)
        assert np.array_equal(res_num2, res_np2)


@pytest.mark.parametrize(
    "ndim",
    (
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_nd_repeats(ndim):
    a_shape = tuple(np.random.randint(1, 9) for _ in range(ndim))
    np_array = mk_seq_array(np, a_shape)
    num_array = mk_seq_array(num, a_shape)
    for axis in range(0, ndim):
        rep_shape = (a_shape[axis],)
        rep_arr_np = mk_seq_array(np, rep_shape)
        rep_arr_num = mk_seq_array(num, rep_shape)
        res_num3 = num.repeat(num_array, rep_arr_num, axis)
        res_np3 = np.repeat(np_array, rep_arr_np, axis)
        assert np.array_equal(res_num3, res_np3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
