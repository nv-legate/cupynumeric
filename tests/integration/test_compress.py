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
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


@pytest.mark.xfail
def test_none_array():
    res_np = np.compress([0], None)  # numpy return []
    # cuPyNumeric raises:
    # AttributeError: 'NoneType' object has no attribute 'compress'
    res_num = num.compress([0], None)
    assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
def test_empty_array():
    res_np = np.compress([0], [])  # numpy return []
    # cuPyNumeric raises: ValueError:
    # Shape mismatch: condition contains entries that are out of bounds
    res_num = num.compress([0], [])
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("con", (-3, 0, 3, None, False, True))
def test_negative_condition(con):
    a = num.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        num.compress(con, a)


def test_condition_out_bound():
    a = num.array([1, 2, 3, 4])
    msg = r"bounds"
    with pytest.raises(ValueError, match=msg):
        num.compress([1, 2, 3, 4, 5], a)


def test_axis_out_bound():
    a = num.array([1, 2, 3, 4])
    msg = r"bounds"
    with pytest.raises(ValueError, match=msg):
        num.compress([1, 2, 3, 4], a, axis=1)


@pytest.mark.parametrize(
    "con", ([True, True], [True, True, True, True, True, True])
)
def test_out_bounds(con):
    a = num.array([1, 2, 3, 4])
    b = num.array([-1, -2, -3, -4])
    with pytest.raises(ValueError):
        num.compress(con, a, out=b)


@pytest.mark.xfail
def test_dtype_out1():
    a = mk_seq_array(np, (4,))
    b = mk_seq_array(num, (4,))
    out_np = np.random.random((4,))
    out_num = num.random.random((4,))
    # for Numpy, it will raise TypeError:
    # "Cannot cast array data from dtype('float64') to dtype('int64')
    # according to the rule 'safe'".
    # cuPyNumeric passed.
    np.compress([True, True, True, True], a, out=out_np)
    num.compress([True, True, True, True], b, out=out_num)
    assert np.array_equal(out_np, out_num)


def test_dtype_out2():
    # both Numpy and cuPyNumeric turn float into int
    a = np.random.random((4,)) * 10
    b = num.array(a)
    out_np = np.random.randint(1, 10, (4,))
    out_num = num.random.randint(-10, -1, (4,))
    np.compress([True, True, True, True], a, out=out_np)
    num.compress([True, True, True, True], b, out=out_num)
    assert np.array_equal(out_np, out_num)


@pytest.mark.xfail
def test_out_parameter():
    a = mk_seq_array(np, (4,))
    b = mk_seq_array(num, (4,))
    out_np = np.random.randint(1, 5, (4,))
    out_num = np.random.randint(1, 5, (4,))
    np.compress([True, True, True, True], a, 0, out_np)
    num.compress([True, True, True, True], b, 0, out_num)
    # for cuPyNumeric, the last parameter 'out',
    # it should be written as 'out=out_num'
    # otherwise it raises error
    assert np.array_equal(out_num, out_np)


def test_bool_condition():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    num_a = num.array(a)

    res_np = np.compress([True, False], a, axis=1)
    res_num = num.compress([True, False], num_a, axis=1)

    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_ndim_basic(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    # make sure condition is between 0 and 1
    np_condition = np.array((mk_seq_array(np, (5,)) % 2).astype(bool))
    num_condition = num.array((mk_seq_array(num, (5,)) % 2).astype(bool))

    res_np = np.compress(np_condition, np_arr)
    res_num = num.compress(num_condition, num_arr)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_ndim_axis(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    # make sure condition is between 0 and 1
    np_condition = np.array((mk_seq_array(np, (5,)) % 2).astype(bool))
    num_condition = num.array((mk_seq_array(num, (5,)) % 2).astype(bool))

    for axis in range(ndim):
        res_np = np.compress(np_condition, np_arr, axis)
        res_num = num.compress(num_condition, num_arr, axis)
        assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_ndim_out(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    # make sure condition is between 0 and 1
    np_condition = np.array((mk_seq_array(np, (5,)) % 2).astype(bool))
    num_condition = num.array((mk_seq_array(num, (5,)) % 2).astype(bool))

    for axis in range(ndim):
        shape_list = list(shape)
        shape_list[axis] = 3
        shape_new = tuple(shape_list)

        out_np = np.random.randint(1, 10, shape_new)
        out_num = np.random.randint(-10, -1, shape_new)

        np.compress(np_condition, np_arr, axis, out_np)
        num.compress(num_condition, num_arr, axis, out=out_num)

        assert np.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
