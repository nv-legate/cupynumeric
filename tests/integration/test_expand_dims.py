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
    1,
    DIM,
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
]


@pytest.mark.xfail
def test_none_array_compare():
    res_num = num.expand_dims(
        None, 0
    )  # TypeError: cuPyNumeric does not support dtype=object
    res_np = np.expand_dims(None, 0)  # return array([None], dtype=object)
    assert np.array_equal(res_num, res_np, equal_nan=True)


def test_invalid_axis_none():
    size = (1, 2, 1)
    a = num.random.randint(low=-10, high=10, size=size)
    with pytest.raises(TypeError):
        num.expand_dims(a, axis=None)


@pytest.mark.parametrize("axis", (-3, 3))
def test_invalid_axis_range(axis):
    a = num.array([4, 6])
    with pytest.raises(AxisError):
        num.expand_dims(a, axis=axis)


@pytest.mark.parametrize("axes", (-1, -2))
def test_axis_negative(axes):
    size = (1, 2)
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.expand_dims(a, axis=axes)
    res_num = num.expand_dims(b, axis=axes)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_basic(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)
    res_np = np.expand_dims(a, axis=0)
    res_num = num.expand_dims(b, axis=0)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_axis(size):
    a = np.random.randint(low=-10, high=10, size=size)
    b = num.array(a)

    for k in range(len((a.shape))):
        res_np = np.expand_dims(a, axis=k)
        res_num = num.expand_dims(b, axis=k)
        assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
