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
from utils.generators import (
    broadcasts_to,
    broadcasts_to_along_axis,
    mk_seq_array,
)

import cupynumeric as num


def equivalent_shapes_gen(shape):
    yield shape
    for i in range(len(shape) - 1):
        if shape[i] == 1:
            i += 1
            yield shape[i:]
        else:
            break


def test_axis_None():
    x = mk_seq_array(np, (256,))
    x_num = mk_seq_array(num, (256,))

    indices = mk_seq_array(np, (125,))
    indices_num = num.array(indices)

    np.put_along_axis(x, indices, -10, None)
    num.put_along_axis(x_num, indices_num, -10, None)
    assert np.array_equal(x_num, x)


N = 10


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_ndim(ndim):
    shape = (N,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = num.array(np_arr)

    shape_idx = (1,) * ndim
    np_indices = mk_seq_array(np, shape_idx) % N
    num_indices = mk_seq_array(num, shape_idx) % N
    for axis in range(-1, ndim):
        np_a = np_arr.copy()
        num_a = num_arr.copy()
        np.put_along_axis(np_a, np_indices, 8, axis=axis)
        num.put_along_axis(num_a, num_indices, 8, axis=axis)
        assert np.array_equal(np_a, num_a)


@pytest.mark.parametrize(
    "axis", range(-1, 3), ids=lambda axis: f"(axis={axis})"
)
def test_full(axis):
    shape = (3, 4, 5)
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    size = shape[axis]
    axis_values = (0, size - 1, size * 2)

    for shape_idx in broadcasts_to_along_axis(shape, axis, axis_values):
        np_indices = mk_seq_array(np, shape_idx) % shape[axis]
        num_indices = mk_seq_array(num, shape_idx) % shape[axis]
        np_a = np_arr.copy()
        num_a = num_arr.copy()
        np.put_along_axis(np_a, np_indices, 100, axis=axis)
        num.put_along_axis(num_a, num_indices, 100, axis=axis)
        assert np.array_equal(np_a, num_a)


def test_values():
    shape = (3, 4, 5)
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_idx = (3, 4, 5)
    axis = 0
    np_indices = mk_seq_array(np, shape_idx) % shape[axis]
    num_indices = mk_seq_array(num, shape_idx) % shape[axis]

    for shape_values in broadcasts_to(shape_idx):
        for s in equivalent_shapes_gen(shape_values):
            np_values = mk_seq_array(np, s)
            num_values = mk_seq_array(num, s)
            np_a = np_arr.copy()
            num_a = num_arr.copy()
            np.put_along_axis(np_a, np_indices, np_values, axis=axis)
            num.put_along_axis(num_a, num_indices, num_values, axis=axis)
            assert np.array_equal(np_a, num_a)


def test_empty_indice():
    x = mk_seq_array(np, (10,))
    x_num = mk_seq_array(num, (10,))

    indices = np.array([], dtype=int)
    indices_num = num.array([], dtype=int)

    np.put_along_axis(x, indices, 99, axis=0)
    num.put_along_axis(x_num, indices_num, 99, axis=0)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize(
    "indices",
    [
        np.array([], dtype=int),
        pytest.param(
            np.array((0,)),
            marks=pytest.mark.xfail(
                reason="NumPy: IndexError, cuPyNumeric: return None"
            ),
        ),
    ],
    ids=["empty index", "out of bound index"],
)
def test_empty_array(indices):
    arr_np = np.array([])
    arr_num = num.array([])
    np.put_along_axis(arr_np, indices, 1, None)
    num.put_along_axis(arr_num, indices, 1, None)
    assert np.array_equal(arr_np, arr_num)


class TestPutAlongAxisErrors:
    def setup_method(self):
        self.a = num.ones((3, 3))
        self.ai = num.ones((3, 3), dtype=int)

    @pytest.mark.parametrize("dtype", (bool, float), ids=str)
    def test_indices_bad_type(self, dtype):
        ai = num.ones((3, 3), dtype=dtype)
        msg = "`indices` must be an integer array"
        with pytest.raises(TypeError, match=msg):
            num.put_along_axis(self.a, ai, 100, axis=0)

    @pytest.mark.parametrize(
        "shape", ((1,), (3, 3, 1)), ids=lambda shape: f"(shape={shape})"
    )
    def test_indices_bad_dims(self, shape):
        ai = num.ones(shape, dtype=int)
        msg = "`indices` and `a` must have the same number of dimensions"
        with pytest.raises(ValueError, match=msg):
            num.put_along_axis(self.a, ai, 100, axis=0)

    @pytest.mark.parametrize(
        "value", (-4, 3), ids=lambda value: f"(value={value})"
    )
    def test_indices_out_of_bound(self, value):
        ai = num.full((3, 3), value, dtype=int)
        msg = "out of bounds"
        with pytest.raises(IndexError, match=msg):
            num.put_along_axis(self.a, ai, 100, axis=0)

    @pytest.mark.parametrize(
        "axis", (2, -3), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound(self, axis):
        msg = "out of bounds"
        # In Numpy, it raises AxisError
        with pytest.raises(ValueError, match=msg):
            num.put_along_axis(self.a, self.ai, 100, axis=axis)

    def test_axis_float(self):
        axis = 0.0
        msg = "integer argument expected"
        with pytest.raises(TypeError, match=msg):
            num.put_along_axis(self.a, self.ai, 100, axis=axis)

    def test_axis_none_indice_not_1d(self):
        axis = None
        msg = "indices must be 1D if axis=None"
        with pytest.raises(ValueError, match=msg):
            num.put_along_axis(self.a, self.ai, 100, axis=axis)

    def test_axis_none_andim_greater_than_one(self):
        ai = num.ones((3 * 3), dtype=int)
        axis = None
        msg = "a.ndim>1 case is not supported when axis=None"
        with pytest.raises(ValueError, match=msg):
            num.put_along_axis(self.a, ai, 100, axis=axis)

    @pytest.mark.parametrize(
        "shape",
        ((1, 2), (4, 1), (0,), (2,), (4,), (1, 0)),
        ids=lambda shape: f"(shape={shape})",
    )
    def test_values_bad_shape(self, shape):
        values = num.ones(shape)
        with pytest.raises(ValueError):
            num.put_along_axis(self.a, self.ai, values, axis=0)

    def test_values_bad_shape2(self):
        shape = (3, 3, 1)
        values = num.ones(shape)
        with pytest.raises(ValueError):
            num.put_along_axis(self.a, self.ai, values, axis=0)

    @pytest.mark.parametrize(
        "shape", ((0,), (5,), (4, 5)), ids=lambda shape: f"(shape={shape})"
    )
    def test_values_axis_none(self, shape):
        np_arr = mk_seq_array(np, (10,))
        num_arr = mk_seq_array(num, (10,))

        indices = mk_seq_array(np, (7,))
        indices_num = mk_seq_array(num, (7,))

        values = mk_seq_array(np, shape)
        values_num = mk_seq_array(num, shape)

        np.put_along_axis(np_arr, indices, values, None)
        num.put_along_axis(num_arr, indices_num, values_num, None)
        assert np.array_equal(np_arr, num_arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
