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

INDICES_VALUES = (
    (0, 10),
    (0, [10, 20, 30]),
    ([0], 10),
    ([0, 1, 2.5, 1], 10),
)


@pytest.mark.parametrize(
    "indices_values",
    INDICES_VALUES,
    ids=lambda indices_values: f"(indices_values={indices_values})",
)
def test_scalar_indices_values(indices_values):
    indices, values = indices_values
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    np.put(x, indices, values)
    num.put(x_num, indices, values)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize("mode", ("wrap", "clip"))
@pytest.mark.parametrize(
    "values", (10, [10, 20]), ids=lambda values: f"(values={values})"
)
@pytest.mark.parametrize(
    "indices", (100, -100), ids=lambda indices: f"(indices={indices})"
)
def test_scalar_indices_values_mode(indices, values, mode):
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    np.put(x, indices, values, mode=mode)
    num.put(x_num, indices, values, mode=mode)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize(
    "values", (10, [10], [10, 20]), ids=lambda values: f"(values={values})"
)
@pytest.mark.parametrize(
    "indices", (0, [-1]), ids=lambda indices: f"(indices={indices})"
)
def test_scalar_arr(indices, values):
    x = np.zeros((), dtype=int)
    x_num = num.zeros((), dtype=int)
    np.put(x, indices, values)
    num.put(x_num, indices, values)
    assert np.array_equal(x_num, x)


@pytest.mark.parametrize("mode", ("wrap", "clip"))
@pytest.mark.parametrize(
    "indices",
    (-1, 1, [-1, 0], [-1, 0, 1, 2]),
    ids=lambda indices: f"(indices={indices})",
)
def test_scalar_arr_mode(indices, mode):
    x = np.zeros((), dtype=int)
    x_num = num.zeros((), dtype=int)
    values = 10
    np.put(x, indices, values, mode=mode)
    num.put(x_num, indices, values, mode=mode)
    assert np.array_equal(x_num, x)


def test_indices_type_convert():
    x = mk_seq_array(np, (3, 4, 5))
    x_num = mk_seq_array(num, (3, 4, 5))
    values = mk_seq_array(np, (6,)) * 10
    values_num = num.array(values)
    indices = np.array([-2, 2], dtype=np.int32)
    indices_num = num.array(indices)
    np.put(x, indices, values)
    num.put(x_num, indices_num, values_num)
    assert np.array_equal(x_num, x)


INDICES_VALUES_SHAPE = (
    ((0,), (1,)),
    ((2,), (0,)),
    ((2,), (1,)),
    ((2,), (2,)),
    ((2,), (3,)),
    ((2,), (2, 1)),
    ((2,), (3, 2)),
    ((2, 2), (1,)),
    ((2, 2), (4,)),
    ((2, 2), (5,)),
    ((2, 2), (2, 1)),
    ((2, 2), (2, 2)),
    ((2, 2), (3, 3)),
)


@pytest.mark.parametrize(
    "indices_values_shape",
    INDICES_VALUES_SHAPE,
    ids=lambda indices_values_shape: f"(in_val_shape={indices_values_shape})",
)
@pytest.mark.parametrize(
    "shape", ((2, 3, 4), (6,)), ids=lambda shape: f"(arr_shape={shape})"
)
def test_indices_array_and_shape_array(shape, indices_values_shape):
    shape_in, shape_val = indices_values_shape
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    np_indices = mk_seq_array(np, shape_in)
    num_indices = mk_seq_array(num, shape_in)
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, np_indices, np_values)
    num.put(num_arr, num_indices, num_values)
    assert np.array_equal(np_arr, num_arr)


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
def test_ndim_default_mode(ndim):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_in = (3,) * ndim
    np_indices = mk_seq_array(np, shape_in)
    num_indices = mk_seq_array(num, shape_in)
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, np_indices, np_values)
    num.put(num_arr, num_indices, num_values)
    assert np.array_equal(np_arr, num_arr)


INDICES = ([1, 2, 3.2, 100], [[2, 1], [3, 100]], [1], [100])


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
@pytest.mark.parametrize("mode", ("wrap", "clip"))
@pytest.mark.parametrize(
    "indices", INDICES, ids=lambda indices: f"(indices={indices})"
)
def test_ndim_mode(ndim, mode, indices):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_val = (2,) * ndim
    np_values = mk_seq_array(np, shape_val) * 10
    num_values = mk_seq_array(num, shape_val) * 10

    np.put(np_arr, indices, np_values, mode=mode)
    num.put(num_arr, indices, num_values, mode=mode)
    assert np.array_equal(np_arr, num_arr)


def test_empty_array():
    x = np.array([])
    x_num = num.array([])
    values = 10
    indices = np.array([], dtype=int)
    indices_num = num.array([])
    np.put(x, indices, values)
    num.put(x_num, indices_num, values)
    assert np.array_equal(x_num, x)


class TestPutErrors:
    @pytest.mark.parametrize(
        "indices",
        (-13, 12, [0, 1, 12]),
        ids=lambda indices: f"(indices={indices})",
    )
    def test_indices_out_of_bound(self, indices):
        expected_exc = IndexError
        shape = (3, 4)
        x_np = mk_seq_array(np, shape)
        x_num = mk_seq_array(num, shape)
        values = 10
        with pytest.raises(expected_exc):
            np.put(x_np, indices, values)
        with pytest.raises(expected_exc):
            num.put(x_num, indices, values)

    @pytest.mark.parametrize(
        "indices",
        (-2, 1, [1]),
        ids=lambda indices: f"(indices={indices})",
    )
    def test_indices_out_of_bound_arr_is_scalar(self, indices):
        expected_exc = IndexError
        x_np = np.zeros((), dtype=int)
        x_num = num.zeros((), dtype=int)
        values = 10
        with pytest.raises(expected_exc):
            np.put(x_np, indices, values)
        with pytest.raises(expected_exc):
            num.put(x_num, indices, values)

    def test_invalid_mode(self):
        expected_exc = ValueError
        shape = (3, 4)
        x_np = mk_seq_array(np, shape)
        x_num = mk_seq_array(num, shape)
        indices = 0
        values = 10
        mode = "unknown"
        with pytest.raises(expected_exc):
            np.put(x_np, indices, values, mode=mode)
        with pytest.raises(expected_exc):
            num.put(x_num, indices, values, mode=mode)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
