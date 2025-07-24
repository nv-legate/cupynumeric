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

from itertools import permutations

import numpy as np
import pytest
from utils.generators import mk_seq_array
from utils.utils import TWO_MAX_DIM_RANGE

import cupynumeric as num


def test_2d():
    a = np.arange(8).reshape((2, 4))
    a_num = num.array(a)
    res = np.trace(a)
    res_num = num.trace(a_num)
    assert np.array_equal(res, res_num)

    res = np.trace(a, dtype=float)
    res_num = num.trace(a_num, dtype=float)
    assert np.array_equal(res, res_num)


def test_3d():
    a = np.arange(8).reshape((2, 2, 2))
    a_num = num.array(a)
    res = np.trace(a)
    res_num = num.trace(a_num)
    assert np.array_equal(res, res_num)

    res = np.trace(a, offset=1)
    res_num = num.trace(a_num, offset=1)
    assert np.array_equal(res, res_num)

    res = np.trace(a, offset=1, axis1=1, axis2=2)
    res_num = num.trace(a_num, offset=1, axis1=1, axis2=2)
    assert np.array_equal(res, res_num)

    out = np.array([1, 2], dtype=float)
    out_num = num.array(out)
    np.trace(a, out=out)
    num.trace(a_num, out=out_num)
    assert np.array_equal(out, out_num)

    np.trace(a, dtype=int, out=out)
    num.trace(a_num, dtype=int, out=out_num)
    assert np.array_equal(out, out_num)


def test_4d():
    a = np.arange(24).reshape((2, 2, 2, 3))
    a_num = num.array(a)
    res = np.trace(a)
    res_num = num.trace(a_num)
    assert np.array_equal(res, res_num)


@pytest.mark.parametrize("ndim", TWO_MAX_DIM_RANGE)
def test_ndim(ndim):
    a_shape = tuple(np.random.randint(1, 9) for i in range(ndim))
    np_array = mk_seq_array(np, a_shape)
    num_array = mk_seq_array(num, a_shape)

    # test trace
    for axes in permutations(range(ndim), 2):
        diag_size = min(a_shape[axes[0]], a_shape[axes[1]]) - 1
        for offset in range(-diag_size + 1, diag_size):
            assert np.array_equal(
                np.trace(np_array, offset, axes[0], axes[1]),
                num.trace(num_array, offset, axes[0], axes[1]),
            )


@pytest.mark.parametrize(
    "offset",
    (
        pytest.param(-3, marks=pytest.mark.xfail),
        pytest.param(-2, marks=pytest.mark.xfail),
        -1,
        0,
        1,
        2,
        pytest.param(3, marks=pytest.mark.xfail),
    ),
    ids=lambda offset: f"(offset={offset})",
)
def test_offset(offset):
    # For -3, -2, 3
    # In Numpy, pass and return 0
    # In cuPyNumeric, it raises ValueError:
    # 'offset' for diag or diagonal must be in range
    a = np.arange(24).reshape((2, 3, 4))
    a_num = num.array(a)
    res = np.trace(a, offset=offset)
    res_num = num.trace(a_num, offset=offset)
    assert np.array_equal(res, res_num)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "axes", ((-2, -1), (-2, 0), (1, -3)), ids=lambda axes: f"(axes={axes})"
)
def test_negative_axes(axes):
    # For all 3 cases,
    # In Numpy, pass
    # In cuPyNumeric, it raises ValueError:
    # axes must be the same size as ndim for transpose
    axis1, axis2 = axes
    a = np.arange(24).reshape((2, 3, 4))
    a_num = num.array(a)
    res = np.trace(a, axis1=axis1, axis2=axis2)
    res_num = num.trace(a_num, axis1=axis1, axis2=axis2)
    assert np.array_equal(res, res_num)


class TestTraceErrors:
    def setup_method(self):
        self.a_np = np.arange(24).reshape((2, 3, 4))
        self.a_num = num.array(self.a_np)

    @pytest.mark.parametrize(
        "array", (1, [], [1]), ids=lambda array: f"(array={array})"
    )
    def test_invalid_arrays(self, array):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.trace(array)
        with pytest.raises(expected_exc):
            num.trace(array)

    @pytest.mark.parametrize(
        "axes",
        (
            (None, 0),
            (0, None),
            pytest.param((None, None), marks=pytest.mark.xfail),
        ),
        ids=lambda axes: f"(axes={axes})",
    )
    def test_axes_none(self, axes):
        # For (None, None)
        # In Numpy, it raises TypeError
        # In cuPyNumeric, it pass
        expected_exc = TypeError
        axis1, axis2 = axes
        with pytest.raises(expected_exc):
            np.trace(self.a_np, axis1=axis1, axis2=axis2)
        with pytest.raises(expected_exc):
            num.trace(self.a_num, axis1=axis1, axis2=axis2)

    @pytest.mark.parametrize(
        "axes", ((-4, 1), (0, 3)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_out_of_bound(self, axes):
        expected_exc = ValueError
        axis1, axis2 = axes
        with pytest.raises(expected_exc):
            np.trace(self.a_np, axis1=axis1, axis2=axis2)
        with pytest.raises(expected_exc):
            num.trace(self.a_num, axis1=axis1, axis2=axis2)

    @pytest.mark.parametrize(
        "axes", ((-3, 0), (1, 1)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_duplicate(self, axes):
        expected_exc = ValueError
        axis1, axis2 = axes
        with pytest.raises(expected_exc):
            np.trace(self.a_np, axis1=axis1, axis2=axis2)
        with pytest.raises(expected_exc):
            num.trace(self.a_num, axis1=axis1, axis2=axis2)

    @pytest.mark.parametrize(
        "out_shape",
        ((0,), (1, 4)),
        ids=lambda out_shape: f"(out_shape={out_shape})",
    )
    def test_out_invalid_shape(self, out_shape):
        expected_exc = ValueError
        out_np = np.zeros(out_shape)
        out_num = num.array(out_np)
        with pytest.raises(expected_exc):
            np.trace(self.a_np, out=out_np)
        with pytest.raises(expected_exc):
            num.trace(self.a_num, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
