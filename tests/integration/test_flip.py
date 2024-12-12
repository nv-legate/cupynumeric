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
from itertools import product

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM
from utils.utils import AxisError

import cupynumeric as num

a = num.random.random((10, 10, 10))
AXES_1d = [-2, 0, 1, 2]
AXES_2d = [-1, 0, 1]


# product minus the "diagonal"
def ul_prod(iterable):
    for a, b in product(iterable, repeat=2):
        if a == b:
            continue
        yield (a, b)


class TestFlipErrors:
    """
    this class is to test negative cases
    flip(m, axis=None)
    """

    def test_axis_float(self):
        axis = 2.5
        msg = r"'float' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            num.flip(a, axis=axis)

    def test_axis_outofbound(self):
        axis = 12
        msg = r"out of bounds"
        with pytest.raises(AxisError, match=msg):
            num.flip(a, axis=axis)

    def test_axis_outofbound_negative(self):
        axis = -12
        msg = r"out of bounds"
        with pytest.raises(AxisError, match=msg):
            num.flip(a, axis=axis)

    def test_repeated_axis(self):
        axis = (2, 2)
        msg = r"repeated axis"
        with pytest.raises(ValueError, match=msg):
            num.flip(a, axis=axis)

    def test_axis_outofbound_tuple(self):
        axis = (1, 5)
        msg = r"out of bounds"
        with pytest.raises(AxisError, match=msg):
            num.flip(a, axis=axis)


class TestFlip:
    """
    These are positive cases compared with numpy
    """

    def test_empty_array(self):
        anp = []
        b = num.flip(anp)
        bnp = np.flip(anp)
        assert num.array_equal(b, bnp)

    def test_basic(self):
        anp = a.__array__()
        b = num.flip(a)
        bnp = np.flip(anp)
        assert num.array_equal(b, bnp)

    @pytest.mark.parametrize("axis", AXES_1d)
    def test_axis_1d(self, axis):
        anp = a.__array__()
        b = num.flip(a, axis=axis)
        bnp = np.flip(anp, axis=axis)
        assert num.array_equal(b, bnp)

    @pytest.mark.parametrize("axis", ul_prod(AXES_2d), ids=str)
    def test_axis_2d(self, axis):
        anp = a.__array__()
        b = num.flip(a, axis=axis)
        bnp = np.flip(anp, axis=axis)
        assert num.array_equal(b, bnp)


class TestFlipud:
    def test_empty_array(self):
        anp = []
        b = num.flipud(anp)
        bnp = np.flipud(anp)
        assert num.array_equal(b, bnp)

    def test_basic(self):
        anp = a.__array__()
        b = num.flipud(a)
        bnp = np.flipud(anp)
        assert num.array_equal(b, bnp)

    def test_wrong_dim(self):
        anp = 4
        msg = r"Input must be >= 1-d"
        with pytest.raises(ValueError, match=msg):
            num.flipud(anp)


class TestFliplr:
    def test_empty_array(self):
        arr = num.random.random((1, 0, 1))
        anp = arr.__array__()
        b = num.fliplr(anp)
        bnp = np.fliplr(anp)
        assert num.array_equal(b, bnp)

    def test_basic(self):
        anp = a.__array__()
        b = num.fliplr(a)
        bnp = np.fliplr(anp)
        assert num.array_equal(b, bnp)

    def test_wrong_dim(self):
        anp = []
        msg = r"Input must be >= 2-d."
        with pytest.raises(ValueError, match=msg):
            num.fliplr(anp)


FLIP_FUNCS = ("flip", "fliplr", "flipud")


@pytest.mark.parametrize("func_name", FLIP_FUNCS)
@pytest.mark.parametrize(
    "ndim",
    (
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_max_dims(func_name, ndim):
    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    shape = (5,) * ndim
    a_np = np.random.random(shape)
    a_num = num.array(a_np)

    out_np = func_np(a_np)
    out_num = func_num(a_num)

    assert np.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
