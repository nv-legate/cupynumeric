# Copyright 2021-2022 NVIDIA Corporation
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

from itertools import chain

import numpy as np
import pytest
from utils.generators import broadcasts_to, mk_seq_array

import cunumeric as num


def equivalent_shapes_gen(shape):
    """
    Generate more equivalent shapes by removing
    leading singleton dimensions from `shape`.
    e.g., shape=(1, 4, 1), yield (1, 4, 1), (4, 1)
    shape=(1, 1, 5), yield (1, 1, 5), (1, 5), (5,)
    """
    yield shape
    for i in range(len(shape) - 1):
        if shape[i] == 1:
            i += 1
            yield shape[i:]
        else:
            break


@pytest.mark.parametrize(
    "endpoint", (True, False), ids=lambda endpoint: f"(endpoint={endpoint})"
)
@pytest.mark.parametrize(
    "number", (0, 1, 10), ids=lambda number: f"(num={number})"
)
@pytest.mark.parametrize(
    "values",
    ((10, -5.5), (2.0, 3.0), (0, 0), (1 + 2.5j, 10 + 5j), (0j, 10)),
    ids=lambda values: f"(values={values})",
)
def test_scalar_basic(values, number, endpoint):
    start, stop = values
    x = np.linspace(start, stop, num=number, endpoint=endpoint)
    y = num.linspace(start, stop, num=number, endpoint=endpoint)
    assert np.array_equal(x, y)


@pytest.mark.parametrize(
    "endpoint", (True, False), ids=lambda endpoint: f"(endpoint={endpoint})"
)
@pytest.mark.parametrize(
    "number", (0, 1, 10), ids=lambda number: f"(num={number})"
)
@pytest.mark.parametrize(
    "values",
    ((10, -5.5), (2.0, 3.0), (0, 0), (1 + 2.5j, 10 + 5j), (0j, 10)),
    ids=lambda values: f"(values={values})",
)
def test_scalar_basic_retstep(values, number, endpoint):
    start, stop = values
    x = np.linspace(start, stop, num=number, endpoint=endpoint, retstep=True)
    y = num.linspace(start, stop, num=number, endpoint=endpoint, retstep=True)

    assert np.array_equal(x[0], y[0])
    if not (np.isnan(x[1]) and np.isnan(y[1])):
        assert x[1] == y[1]


@pytest.mark.parametrize(
    "endpoint", (True, False), ids=lambda endpoint: f"(endpoint={endpoint})"
)
def test_arrays_basic(endpoint):
    shape = (2, 2, 3)
    np_start = mk_seq_array(np, shape)
    num_start = mk_seq_array(num, shape)
    np_stop = mk_seq_array(np, shape) + 10
    num_stop = mk_seq_array(num, shape) + 10
    x = np.linspace(np_start, np_stop, num=5, endpoint=endpoint)
    y = np.linspace(num_start, num_stop, num=5, endpoint=endpoint)
    assert np.array_equal(x, y)


@pytest.mark.parametrize(
    "endpoint", (True, False), ids=lambda endpoint: f"(endpoint={endpoint})"
)
def test_arrays_basic_retstep(endpoint):
    shape = (2, 2, 3)
    np_start = mk_seq_array(np, shape)
    num_start = mk_seq_array(num, shape)
    np_stop = mk_seq_array(np, shape) + 10
    num_stop = mk_seq_array(num, shape) + 10
    x = np.linspace(np_start, np_stop, num=5, endpoint=endpoint, retstep=True)
    y = np.linspace(
        num_start, num_stop, num=5, endpoint=endpoint, retstep=True
    )
    assert np.array_equal(x[0], y[0])
    assert np.array_equal(x[1], y[1])


shape_start = (2, 2, 3)
shape_stops = (equivalent_shapes_gen(s) for s in broadcasts_to(shape_start))


@pytest.mark.parametrize(
    "shape_stop",
    chain.from_iterable(shape_stops),
    ids=lambda shape_stop: f"(shape_stop={shape_stop})",
)
def test_array_broadcast_stops(shape_stop):
    np_start = mk_seq_array(np, shape_start)
    num_start = mk_seq_array(num, shape_start)

    np_stop = mk_seq_array(np, shape_stop) + 5
    num_stop = mk_seq_array(num, shape_stop) + 5
    x = np.linspace(np_start, np_stop, num=5)
    y = num.linspace(num_start, num_stop, num=5)
    assert np.array_equal(x, y)


def test_arrays_both_start_and_stop_broadcast():
    shape_start = (1, 3)
    np_start = mk_seq_array(np, shape_start)
    num_start = mk_seq_array(num, shape_start)
    shape_stop = (2, 1)
    np_stop = mk_seq_array(np, shape_stop) + 5
    num_stop = mk_seq_array(num, shape_stop) + 5

    x = np.linspace(np_start, np_stop, num=5)
    y = num.linspace(num_start, num_stop, num=5)
    assert np.array_equal(x, y)


@pytest.mark.parametrize(
    "shape", ((0,), (3,), (2, 1)), ids=lambda shape: f"(shape={shape})"
)
def test_array_with_scalar(shape):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    scalar = 10

    x1 = np.linspace(np_arr, scalar, num=5)
    y1 = num.linspace(num_arr, scalar, num=5)
    assert np.array_equal(x1, y1)

    x2 = np.linspace(scalar, np_arr, num=5)
    y2 = num.linspace(scalar, num_arr, num=5)
    assert np.array_equal(x2, y2)


@pytest.mark.parametrize(
    "endpoint", (True, False), ids=lambda endpoint: f"(endpoint={endpoint})"
)
@pytest.mark.parametrize(
    "shape", ((0,), (2, 1)), ids=lambda shape: f"(shape={shape})"
)
def test_empty_array(shape, endpoint):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    x1 = np.linspace(np_arr, [], num=5, endpoint=endpoint)
    y1 = num.linspace(num_arr, [], num=5, endpoint=endpoint)
    assert np.array_equal(x1, y1)

    x2 = np.linspace([], np_arr, num=5, endpoint=endpoint)
    y2 = num.linspace([], num_arr, num=5, endpoint=endpoint)
    assert np.array_equal(x2, y2)


@pytest.mark.parametrize(
    "endpoint", (True, False), ids=lambda endpoint: f"(endpoint={endpoint})"
)
@pytest.mark.parametrize(
    "shape", ((0,), (2, 1)), ids=lambda shape: f"(shape={shape})"
)
def test_empty_array_retstep(shape, endpoint):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    x1 = np.linspace(np_arr, [], num=5, endpoint=endpoint, retstep=True)
    y1 = num.linspace(num_arr, [], num=5, endpoint=endpoint, retstep=True)
    assert np.array_equal(x1[0], y1[0])
    assert np.array_equal(x1[1], y1[1])

    x2 = np.linspace([], np_arr, num=5, endpoint=endpoint, retstep=True)
    y2 = num.linspace([], num_arr, num=5, endpoint=endpoint, retstep=True)
    assert np.array_equal(x2[0], y2[0])
    assert np.array_equal(x2[1], y2[1])


@pytest.mark.xfail
@pytest.mark.parametrize(
    "number", (0, 1, 10), ids=lambda number: f"(num={number})"
)
@pytest.mark.parametrize(
    "axis", range(-3, 3), ids=lambda axis: f"(axis={axis})"
)
def test_arrays_axis(axis, number):
    # In cuNumeric, if axis < -1, raise ValueError
    # 'Point cannot exceed 4 dimensions set from LEGATE_MAX_DIM'
    # In Numpy, if axis is -2 or -3, also pass
    # In cuNumeric, for axis >= -1, if num=0, raise IndexError:
    # tuple index out of range
    # In Numpy, if num=0, pass and returns empty array
    x = np.array([[0, 1], [2, 3]])
    y = np.array([[4, 5], [6, 7]])
    xp = num.array(x)
    yp = num.array(y)

    z = np.linspace(x, y, num=number, axis=axis)
    w = num.linspace(xp, yp, num=number, axis=axis)
    assert np.array_equal(z, w)


@pytest.mark.parametrize(
    "axis", range(-1, 1), ids=lambda axis: f"(axis={axis})"
)
def test_scalar_axis(axis):
    start = 2.0
    stop = 3.0
    x = np.linspace(start, stop, num=5, axis=axis)
    y = num.linspace(start, stop, num=5, axis=axis)
    assert np.array_equal(x, y)


@pytest.mark.parametrize(
    "dtype", (None, int, float, bool), ids=lambda dtype: f"(dtype={dtype})"
)
def test_dtype(dtype):
    start = 2.0
    stop = 3.0
    x = np.linspace(start, stop, num=5, dtype=dtype)
    y = num.linspace(start, stop, num=5, dtype=dtype)
    assert np.array_equal(x, y)


class TestLinspaceErrors:
    def setup_method(self):
        self.start = mk_seq_array(num, (2, 3))
        self.stop = mk_seq_array(num, (2, 3)) + 10
        self.num = 5

    @pytest.mark.xfail
    def test_num_float(self):
        # In Numpy, raise TypeError
        # In cuNumeric, pass
        msg = "cannot be interpreted as an integer"
        with pytest.raises(TypeError, match=msg):
            num.linspace(0, 10, num=4.5)

    def test_num_negative(self):
        msg = "must be non-negative"
        with pytest.raises(ValueError, match=msg):
            num.linspace(0, 10, num=-1)

    def test_num_none(self):
        msg = "not supported between instances of 'NoneType' and 'int'"
        with pytest.raises(TypeError, match=msg):
            num.linspace(0, 10, num=None)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "axis", (-4, 3), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound_array(self, axis):
        # In cuNumeric, if axis < -1, raise ValueError
        # 'Point cannot exceed 4 dimensions set from LEGATE_MAX_DIM'
        msg = "out of bounds"
        # In Numpy, it raises AxisError
        with pytest.raises(ValueError, match=msg):
            num.linspace(self.start, self.stop, axis=axis)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "axis", (-2, 1), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound_scalar(self, axis):
        # In cuNumeric, it pass and the result equals when axis=0
        # In Numpy, it raises AxisError
        msg = "out of bounds"
        with pytest.raises(ValueError, match=msg):
            num.linspace(2.0, 3.0, axis=axis)

    def test_axis_float(self):
        axis = 1.0
        msg = "can't multiply sequence by non-int of type 'float'"
        with pytest.raises(TypeError, match=msg):
            num.linspace(self.start, self.stop, axis=axis)

    @pytest.mark.xfail
    def test_axis_none(self):
        # In cuNumeric, pass and treat it as axis=0
        # In Numpy, raises TypeError
        axis = None
        msg = "'NoneType' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            num.linspace(self.start, self.stop, axis=axis)

    @pytest.mark.parametrize(
        "shape", ((0,), (2,), (3, 3)), ids=lambda shape: f"(shape={shape})"
    )
    def test_array_bad_shape(self, shape):
        msg = "shape mismatch"
        stop = mk_seq_array(num, shape)
        with pytest.raises(ValueError, match=msg):
            num.linspace(self.start, stop)

    def test_start_none(self):
        with pytest.raises(Exception):
            num.linspace(None, 10, num=5)

    def test_stop_none(self):
        with pytest.raises(Exception):
            num.linspace(0, None, num=5)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_meshgrid(sparse, indexing):
    xnp = np.linspace(0.0, 1.0, 10)
    ynp = np.linspace(0.5, 1.5, 10)
    Xnp, Ynp = np.meshgrid(xnp, ynp, sparse=sparse, indexing=indexing)

    xnum = num.linspace(0.0, 1.0, 10)
    ynum = num.linspace(0.5, 1.5, 10)
    Xnum, Ynum = num.meshgrid(xnum, ynum, sparse=sparse, indexing=indexing)

    assert num.array_equal(Xnum, Xnp)
    assert num.array_equal(Ynum, Ynp)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
