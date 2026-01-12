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
from utils.utils import AxisError

import cupynumeric as num

DIM = 5
SIZES = [
    (0,),
    (1),
    (DIM),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]


def test_none_array_compare() -> None:
    res_num = num.squeeze(None)
    res_np = np.squeeze(None)
    assert res_num is None
    assert res_np.item() is None


def test_invalid_axis() -> None:
    size = (1, 2, 1)
    a_np = np.random.randint(low=-10, high=10, size=size)
    msg = r"one"
    with pytest.raises(ValueError, match=msg):
        np.squeeze(a_np, axis=1)

    a_num = num.array(a_np)
    with pytest.raises(ValueError, match=msg):
        num.squeeze(a_num, axis=1)


def test_axis_out_bound() -> None:
    size = (1, 2, 1)
    a_np = np.random.randint(low=-10, high=10, size=size)
    msg = r"axis 3 is out of bounds for array of dimension 3"
    with pytest.raises(AxisError, match=msg):
        np.squeeze(a_np, axis=3)

    a_num = num.array(a_np)
    with pytest.raises(ValueError, match=msg):
        num.squeeze(a_num, axis=3)


@pytest.mark.parametrize("axes", (-1, -3))
def test_num_axis_negative(axes):
    size = (1, 2, 1)
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.squeeze(a, axis=axes)
    res_num = num.squeeze(b, axis=axes)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("axes", (-1, -3))
def test_array_axis_negative(axes):
    size = (1, 2, 1)
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.squeeze(axis=axes)
    res_num = b.squeeze(axis=axes)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_num_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.squeeze(a)
    res_num = num.squeeze(b)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = a.squeeze()
    res_num = b.squeeze()
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size", (s for s in SIZES if isinstance(s, tuple) if 1 in s), ids=str
)
def test_num_axis(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)

    for k, axis in enumerate(a.shape):
        if axis == 1:
            res_np = np.squeeze(a, axis=k)
            res_num = num.squeeze(b, axis=k)
            assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "size", (s for s in SIZES if isinstance(s, tuple) if 1 in s), ids=str
)
def test_array_axis(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)

    for k, axis in enumerate(a.shape):
        if axis == 1:
            res_np = a.squeeze(axis=k)
            res_num = b.squeeze(axis=k)
            assert np.array_equal(res_num, res_np)


class TestInvalid:
    @pytest.mark.parametrize(
        "axis,shape",
        [
            (0, (1, 3, 4)),  # axis=0, squeeze first dimension
            (1, (3, 1, 4)),  # axis=1, squeeze middle dimension
        ],
    )
    def test_squeeze_int_axis_coverage(self, axis: int, shape: tuple) -> None:
        np_arr = np.ones(shape)
        num_arr = num.ones(shape)

        # axis changes to tuple in array.squeeze(), so call the thunk method directly
        thunk_result = num_arr._thunk.squeeze(axis=axis)
        np_result = np_arr.squeeze(axis=axis)

        assert thunk_result.shape == np_result.shape

    def test_squeeze_invalid_axis_type(self) -> None:
        num_arr = num.ones((1, 3, 4))

        # Pass an invalid axis type (e.g., a string) to trigger the else branch
        with pytest.raises(TypeError):
            # Directly call the thunk method to bypass public API validation
            num_arr._thunk.squeeze("invalid_axis_type")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
