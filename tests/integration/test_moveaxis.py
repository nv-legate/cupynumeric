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
from utils.generators import mk_0to1_array
from utils.utils import TWO_MAX_DIM_RANGE, AxisError

import cupynumeric as num

AXES = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), ([0, 1], [1, 0]))


@pytest.mark.parametrize("ndim", TWO_MAX_DIM_RANGE)
@pytest.mark.parametrize("axes", AXES)
def test_moveaxis(ndim, axes):
    source, destination = axes
    a_np = mk_0to1_array(np, (3,) * ndim)
    a_num = mk_0to1_array(num, (3,) * ndim)
    res_np = np.moveaxis(a_np, source, destination)
    res_num = num.moveaxis(a_num, source, destination)
    assert np.array_equal(res_np, res_num)
    # Check that the returned array is a view (for deferred case inly)
    from cupynumeric.settings import settings

    if settings.force_thunk() != "eager":
        res_num[:] = 0
        assert a_num.sum() == 0


def test_moveaxis_with_empty_axis():
    a_np = np.ones((3, 4, 5))
    a_num = num.ones((3, 4, 5))

    axes = ([], [])
    source, destination = axes

    res_np = np.moveaxis(a_np, source, destination)
    res_num = num.moveaxis(a_num, source, destination)
    assert np.array_equal(res_np, res_num)


EMPTY_ARRAYS = ([], [[]], [[], []])


@pytest.mark.parametrize("a", EMPTY_ARRAYS)
def test_moveaxis_with_empty_array(a):
    axes = (0, -1)
    source, destination = axes

    res_np = np.moveaxis(a, source, destination)
    res_num = num.moveaxis(a, source, destination)
    assert np.array_equal(res_np, res_num)


class TestMoveAxisErrors:
    def setup_method(self):
        self.x = num.ones((3, 4, 5))

    def test_repeated_axis(self):
        msg = "repeated axis"
        with pytest.raises(ValueError, match=msg):
            num.moveaxis(self.x, [0, 0], [1, 0])

        with pytest.raises(ValueError, match=msg):
            num.moveaxis(self.x, [0, 1], [0, -3])

    def test_axis_out_of_bound(self):
        msg = "out of bound"
        with pytest.raises(AxisError, match=msg):
            num.moveaxis(self.x, [0, 3], [0, 1])

        with pytest.raises(AxisError, match=msg):
            num.moveaxis(self.x, [0, 1], [0, -4])

        with pytest.raises(AxisError, match=msg):
            num.moveaxis(self.x, 4, 0)

        with pytest.raises(AxisError, match=msg):
            num.moveaxis(self.x, 0, -4)

    def test_axis_with_different_length(self):
        msg = "arguments must have the same number of elements"
        with pytest.raises(ValueError, match=msg):
            num.moveaxis(self.x, [0], [1, 0])

    def test_axis_float(self):
        msg = "integer argument expected, got float"
        with pytest.raises(TypeError, match=msg):
            num.moveaxis(self.x, [0.0, 1], [1, 0])

        with pytest.raises(TypeError, match=msg):
            num.moveaxis(self.x, [0, 1], [1, 0.0])

    def test_axis_none(self):
        msg = "'NoneType' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            num.moveaxis(self.x, None, 0)

        with pytest.raises(TypeError, match=msg):
            num.moveaxis(self.x, 0, None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
