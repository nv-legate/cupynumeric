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

import cupynumeric as num

CONDITIONS = [
    [[True, False], [True, True]],
    [[True, False]],
    [True, False],
    False,
    [[0.0, 1.0], [0, -2]],
]


def test_basic():
    a_np = np.array([1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0])
    a_num = num.array(a_np)
    assert num.array_equal(np.where(a_np), num.where(a_num))


@pytest.mark.parametrize("cond", CONDITIONS, ids=str)
def test_condition(cond):
    a_np = np.array(cond)
    x_np = np.array([[1, 2], [3, 4]])
    y_np = np.array([[9, 8], [7, 6]])
    a_num = num.array(a_np)
    x_num = num.array(x_np)
    y_num = num.array(y_np)
    assert np.array_equal(
        np.where(a_np, x_np, y_np), num.where(a_num, x_num, y_num)
    )


@pytest.mark.parametrize(
    "shape_a",
    ((1,), (3,), (1, 3), (3, 3), (2, 3, 3)),
    ids=lambda shape_a: f"(shape_a={shape_a})",
)
def test_broadcast(shape_a):
    a_num = mk_seq_array(num, shape_a)
    a_np = mk_seq_array(np, shape_a)
    cond_num = a_num > 5
    cond_np = a_np > 5

    shape_x = (3, 3)
    x_num = mk_seq_array(num, shape_x)
    x_np = mk_seq_array(np, shape_x)
    shape_y = (1, 3)
    y_num = mk_seq_array(num, shape_y) * 10
    y_np = mk_seq_array(np, shape_y) * 10

    assert np.array_equal(
        np.where(cond_np, x_np, y_np), num.where(cond_num, x_num, y_num)
    )


@pytest.mark.xfail
def test_condition_none():
    # In Numpy, pass and returns [1, 2]
    # In cuPyNumeric, raises AttributeError:
    # 'NoneType' object has no attribute '_maybe_convert'
    x = 0
    y_np = np.array([1, 2])
    y_num = num.array(y_np)
    assert np.array_equal(np.where(None, x, y_np), num.where(None, x, y_num))


@pytest.mark.xfail
@pytest.mark.parametrize(
    "values",
    ((None, None), (None, 1), (1, None)),
    ids=lambda values: f"(values={values})",
)
def test_x_y_none(values):
    # For x=None and y=None,
    # In Numpy, pass and returns [None, None]
    # In cuPyNumeric, pass and returns (array([0]),)
    # For x=None and y=1
    # In Numpy, pass and returns [None, 1]
    # In cuPyNumeric, raises ValueError: both 'x' and 'y' parameters
    # must be specified together for where
    cond = [True, False]
    a_np = np.array(cond)
    a_num = num.array(a_np)
    x, y = values
    assert np.array_equal(np.where(a_np, x, y), num.where(a_num, x, y))


def test_x_y_type():
    x_np = np.arange(4, dtype=np.int32)
    y_np = np.arange(4, dtype=np.float32) * 2.2
    x_num = num.array(x_np)
    y_num = num.array(y_np)

    res_np = np.where(x_np > 2.0, x_np, y_np)
    res_num = num.where(x_num > 2.0, x_num, y_num)

    assert np.array_equal(res_np, res_num)
    assert res_np.dtype == res_num.dtype


def test_condition_empty():
    cond_num = num.array([])
    cond_np = np.array([])
    x = 0
    y = 1
    assert np.array_equal(np.where(cond_np, x, y), num.where(cond_num, x, y))


class TestWhereErrors:
    @pytest.mark.parametrize(
        "shape_y",
        ((0,), (2,), (1, 2), (4, 1)),
        ids=lambda shape_y: f"(shape_y={shape_y})",
    )
    def test_x_y_bad_shape(self, shape_y):
        shape_a = (3, 3)
        a = mk_seq_array(num, shape_a)
        cond = a > 5
        x = 1
        y = mk_seq_array(num, shape_y)

        msg = "shape mismatch"
        with pytest.raises(ValueError, match=msg):
            num.where(cond, x, y)


INPUT = [
    [1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0, 1],
    [[1, 54, 4], [4, 0, 45], [5, 58, 0], [9, 0, 4], [0, 0, 0], [5, 0, 1]],
    [
        [[1, 54, 4], [4, 0, 45]],
        [[5, 58, 0], [9, 0, 4]],
        [[0, 0, 0], [5, 0, 1]],
    ],
    [[[1 + 2j, 54, 4], [4, 0 + 1j, 45]], [[5, 58, 0], [9, 0, 4]]],
    [[True, False], [True, True], [True, False]],
]


@pytest.mark.parametrize("input", INPUT, ids=str)
def test_argwhere(input):
    a_np = np.array(input)
    a_num = num.array(a_np)
    assert np.array_equal(np.argwhere(a_np), num.argwhere(a_num))


@pytest.mark.xfail
def test_argwhere_none():
    # In Numpy, it pass and returns []
    # In cuPyNumeric, it raises AttributeError:
    # 'NoneType' object has no attribute '_thunk'
    assert np.array_equal(np.argwhere(None), num.argwhere(None))


def test_argwhere_empty():
    a_np = np.array([])
    a_num = num.array(a_np)
    assert np.array_equal(np.argwhere(a_np), num.argwhere(a_num))


def test_argwhere_scalar():
    assert np.array_equal(np.argwhere(1), num.argwhere(1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
