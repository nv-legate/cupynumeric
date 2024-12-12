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
from utils.generators import broadcasts_to_along_axis, mk_seq_array

import cupynumeric as num

N = 10


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
def test_ndim(ndim):
    shape = (N,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_idx = (1,) * ndim
    np_indices = mk_seq_array(np, shape_idx) % N
    num_indices = mk_seq_array(num, shape_idx) % N
    for axis in range(-1, ndim):
        res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
        res_num = num.take_along_axis(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)
    np_indices = mk_seq_array(np, (3,))
    num_indices = mk_seq_array(num, (3,))
    res_np = np.take_along_axis(np_arr, np_indices, None)
    res_num = num.take_along_axis(num_arr, num_indices, None)
    assert np.array_equal(res_num, res_np)


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
        res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
        res_num = num.take_along_axis(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)


def test_empty_indice():
    np_arr = mk_seq_array(np, (10,))
    num_arr = mk_seq_array(num, (10,))
    np_indices = np.array([], dtype=int)
    num_indices = num.array([], dtype=int)
    res_np = np.take_along_axis(np_arr, np_indices, axis=0)
    res_num = num.take_along_axis(num_arr, num_indices, axis=0)
    assert np.array_equal(res_num, res_np)


class TestTakeAlongAxisErrors:
    def setup_method(self):
        self.a = num.ones((3, 3))
        self.ai = num.ones((3, 3), dtype=int)

    @pytest.mark.parametrize("dtype", (bool, float), ids=str)
    def test_indices_bad_type(self, dtype):
        ai = num.ones((3, 3), dtype=dtype)
        msg = "`indices` must be an integer array"
        with pytest.raises(TypeError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "shape", ((3, 2), (3, 0)), ids=lambda shape: f"(shape={shape})"
    )
    def test_indices_bad_shape(self, shape):
        # In Numpy, it raises IndexError.
        # In cuPyNumeric, it raises ValueError.
        ai = num.ones(shape, dtype=int)
        msg = "shape mismatch: indexing arrays could not be broadcast"
        with pytest.raises(IndexError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.parametrize(
        "shape", ((1,), (3, 3, 1)), ids=lambda shape: f"(shape={shape})"
    )
    def test_indices_bad_dims(self, shape):
        ai = num.ones(shape, dtype=int)
        msg = "`indices` and `a` must have the same number of dimensions"
        with pytest.raises(ValueError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.parametrize(
        "value", (-4, 3), ids=lambda value: f"(value={value})"
    )
    def test_indices_out_of_bound(self, value):
        ai = num.full((3, 3), value, dtype=int)
        msg = "out of bounds"
        with pytest.raises(IndexError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.parametrize(
        "axis", (2, -3), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound(self, axis):
        msg = "out of bounds"
        # In Numpy, it raises AxisError
        with pytest.raises(ValueError, match=msg):
            num.take_along_axis(self.a, self.ai, axis=axis)

    def test_axis_float(self):
        axis = 0.0
        msg = "integer argument expected"
        with pytest.raises(TypeError, match=msg):
            num.take_along_axis(self.a, self.ai, axis=axis)

    def test_axis_none_indice_not_1d(self):
        axis = None
        msg = "indices must be 1D if axis=None"
        with pytest.raises(ValueError, match=msg):
            num.take_along_axis(self.a, self.ai, axis=axis)

    def test_a_none(self):
        ai = num.array([1, 1, 1])
        msg = "object has no attribute 'ndim'"
        with pytest.raises(AttributeError, match=msg):
            num.take_along_axis(None, ai, axis=0)

    def test_indice_none(self):
        msg = "'NoneType' object has no attribute 'dtype'"
        with pytest.raises(AttributeError, match=msg):
            num.take_along_axis(self.a, None, axis=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
