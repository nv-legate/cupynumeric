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

import cupynumeric as num

x = mk_seq_array(np, (3, 4, 5))
x_num = mk_seq_array(num, (3, 4, 5))
indices = mk_seq_array(np, (8,))
indices_num = num.array(indices)
indices2 = mk_seq_array(np, (3,))
indices2_num = num.array(indices2)


def test_no_axis():
    res = np.take(x, indices)
    res_num = num.take(x_num, indices_num)

    assert np.array_equal(res_num, res)


@pytest.mark.parametrize("mode", ("clip", "wrap"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_different_axis_mode(axis, mode):
    res = np.take(x, indices, axis=axis, mode=mode)
    res_num = num.take(x_num, indices_num, axis=axis, mode=mode)
    assert np.array_equal(res_num, res)


def test_different_axis_default_mode():
    res = np.take(x, indices2, axis=1)
    res_num = num.take(x_num, indices2_num, axis=1)

    assert np.array_equal(res_num, res)


def test_different_axis_raise_mode():
    res = np.take(x, indices2, axis=2, mode="raise")
    res_num = num.take(x_num, indices2_num, axis=2, mode="raise")
    assert np.array_equal(res_num, res)


def test_with_out_array():
    out = np.ones((3, 4, 3), dtype=int)
    out_num = num.array(out)
    np.take(x, indices2, axis=2, mode="raise", out=out)
    num.take(x_num, indices2_num, axis=2, mode="raise", out=out_num)
    assert np.array_equal(out_num, out)


@pytest.mark.parametrize(
    "indices",
    (-3, 2),
    ids=lambda indices: f"(indices={indices})",
)
def test_scalar_indices_default_mode(indices):
    res = np.take(x, indices, axis=0)
    res_num = num.take(x_num, indices, axis=0)

    assert np.array_equal(res_num, res)


@pytest.mark.parametrize("mode", ("clip", "wrap"))
@pytest.mark.parametrize(
    "indices",
    (-4, 2, 7),
    ids=lambda indices: f"(indices={indices})",
)
def test_scalar_indices_mode(mode, indices):
    res = np.take(x, indices, axis=0, mode=mode)
    res_num = num.take(x_num, indices, axis=0, mode=mode)

    assert np.array_equal(res_num, res)


def test_empty_array_and_indices():
    np_arr = mk_seq_array(np, (0,))
    num_arr = mk_seq_array(num, (0,))
    np_indices = np.array([], dtype=int)
    num_indices = num.array([], dtype=int)

    res_np = np.take(np_arr, np_indices)
    res_num = num.take(num_arr, num_indices)
    assert np.array_equal(res_num, res_np)

    axis = 0
    res_np = np.take(np_arr, np_indices, axis=axis)
    res_num = num.take(num_arr, num_indices, axis=axis)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "shape_in",
    ((4,), (0,), pytest.param((2, 2), marks=pytest.mark.xfail)),
    ids=lambda shape_in: f"(shape_in={shape_in})",
)
@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim_default_mode(ndim, shape_in):
    # for shape_in=(2, 2) and ndim=4,
    # In Numpy, pass
    # In cuPyNumeric, it raises ValueError:
    # Point cannot exceed 4 dimensions set from LEGATE_MAX_DIM
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_indices = mk_seq_array(np, shape_in)
    num_indices = mk_seq_array(num, shape_in)

    res_np = np.take(np_arr, np_indices)
    res_num = num.take(num_arr, num_indices)
    assert np.array_equal(res_num, res_np)

    for axis in range(ndim):
        res_np = np.take(np_arr, np_indices, axis=axis)
        res_num = num.take(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("mode", ("clip", "wrap"))
@pytest.mark.parametrize(
    "shape_in",
    ((8,), pytest.param((3, 4), marks=pytest.mark.xfail)),
    ids=lambda shape_in: f"(shape_in={shape_in})",
)
@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim_mode(ndim, mode, shape_in):
    # for shape_in=(3, 4) and ndim=4,
    # In Numpy, pass
    # In cuPyNumeric, it raises ValueError:
    # Point cannot exceed 4 dimensions set from LEGATE_MAX_DIM
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_indices = mk_seq_array(np, shape_in)
    num_indices = mk_seq_array(num, shape_in)

    res_np = np.take(np_arr, np_indices, mode=mode)
    res_num = num.take(num_arr, num_indices, mode=mode)
    assert np.array_equal(res_num, res_np)

    for axis in range(ndim):
        res_np = np.take(np_arr, np_indices, axis=axis, mode=mode)
        res_num = num.take(num_arr, num_indices, axis=axis, mode=mode)
        assert np.array_equal(res_num, res_np)


class TestTakeErrors:
    def setup_method(self):
        self.A_np = mk_seq_array(np, (3, 4, 5))
        self.A_num = mk_seq_array(num, (3, 4, 5))

    @pytest.mark.parametrize(
        "indices",
        (-5, 4),
        ids=lambda indices: f"(indices={indices})",
    )
    def test_indices_invalid_scalar(self, indices):
        expected_exc = IndexError
        axis = 1
        with pytest.raises(expected_exc):
            np.take(self.A_np, indices, axis=axis)
        with pytest.raises(expected_exc):
            num.take(self.A_num, indices, axis=axis)

    @pytest.mark.parametrize(
        "indices",
        ([-5, 0, 2], [0, 4, 2]),
        ids=lambda indices: f"(indices={indices})",
    )
    def test_indices_invalid_array(self, indices):
        expected_exc = IndexError
        axis = 1
        with pytest.raises(expected_exc):
            np.take(self.A_np, np.array(indices), axis=axis)
        with pytest.raises(expected_exc):
            num.take(self.A_num, num.array(indices), axis=axis)

    def test_invalid_indices_for_empty_array(self):
        expected_exc = IndexError
        A_np = mk_seq_array(np, (0,))
        A_num = mk_seq_array(num, (0,))
        indices = [0]
        axis = 0
        mode = "clip"
        with pytest.raises(expected_exc):
            np.take(A_np, np.array(indices), axis=axis, mode=mode)
        with pytest.raises(expected_exc):
            num.take(A_num, num.array(indices), axis=axis, mode=mode)

    @pytest.mark.parametrize(
        "axis",
        (-4, 3),
        ids=lambda axis: f"(axis={axis})",
    )
    def test_axis_out_of_bound(self, axis):
        expected_exc = ValueError
        indices = 0
        with pytest.raises(expected_exc):
            np.take(self.A_np, indices, axis=axis)
        with pytest.raises(expected_exc):
            num.take(self.A_num, indices, axis=axis)

    def test_axis_float(self):
        expected_exc = TypeError
        indices = 0
        axis = 0.0
        with pytest.raises(expected_exc):
            np.take(self.A_np, indices, axis=axis)
        with pytest.raises(expected_exc):
            num.take(self.A_num, indices, axis=axis)

    def test_invalid_mode(self):
        expected_exc = ValueError
        indices = 0
        axis = 1
        mode = "unknown"
        with pytest.raises(expected_exc):
            np.take(self.A_np, indices, axis=axis, mode=mode)
        with pytest.raises(expected_exc):
            num.take(self.A_num, indices, axis=axis, mode=mode)

    @pytest.mark.parametrize(
        "shape",
        ((2,), (3, 2), (3, 2, 4), (3, 4, 5)),
        ids=lambda shape: f"(shape={shape})",
    )
    def test_out_invalid_shape(self, shape):
        expected_exc = ValueError
        indices = [1, 0]
        axis = 1
        out_np = np.zeros(shape, dtype=np.int64)
        out_num = num.zeros(shape, dtype=np.int64)
        with pytest.raises(expected_exc):
            np.take(self.A_np, np.array(indices), axis=axis, out=out_np)
        with pytest.raises(expected_exc):
            num.take(self.A_num, num.array(indices), axis=axis, out=out_num)

    @pytest.mark.parametrize(
        "dtype",
        (np.float32, pytest.param(np.int32, marks=pytest.mark.xfail)),
        ids=lambda dtype: f"(dtype={dtype})",
    )
    def test_out_invalid_dtype(self, dtype):
        # In Numpy,
        # for np.float32, it raises TypeError
        # for np.int64 and np.int32, it pass
        # In cuPyNumeric,
        # for np.float32, it raises ValueError
        # for np.int32, it raises ValueError
        # for np.int64, it pass
        expected_exc = TypeError
        indices = [1, 0]
        axis = 1
        out_np = np.zeros((3, 2, 5), dtype=dtype)
        out_num = num.zeros((3, 2, 5), dtype=dtype)
        with pytest.raises(expected_exc):
            np.take(self.A_np, np.array(indices), axis=axis, out=out_np)
        with pytest.raises(expected_exc):
            num.take(self.A_num, num.array(indices), axis=axis, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
