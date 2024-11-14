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
from utils.utils import AxisError

import cupynumeric as num

ARG_FUNCS = ("argmax", "argmin")


class TestArgReduceErrors:
    """
    this class is to test negative cases
    argmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    argmin(a, axis=None, out=None, *, keepdims=np._NoValue)
    """

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_empty_ary(self, func_name):
        # empty array should not be accepted
        func = getattr(num, func_name)

        msg = r"an empty sequence"
        with pytest.raises(ValueError, match=msg):
            func([])

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_axis_float(self, func_name):
        ndim = 3
        shape = (5,) * ndim
        in_num = np.random.random(shape)
        func = getattr(num, func_name)

        msg = r"axis must be an integer"
        with pytest.raises(ValueError, match=msg):
            func(in_num, axis=ndim - 0.5)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_axis_outofbound(self, func_name):
        ndim = 4
        shape = (5,) * ndim
        in_num = np.random.random(shape)
        func = getattr(num, func_name)

        msg = r"out of bounds"
        with pytest.raises(AxisError, match=msg):
            func(in_num, axis=ndim + 1)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_axis_negative(self, func_name):
        ndim = 2
        shape = (5,) * ndim
        in_num = np.random.random(shape)
        func = getattr(num, func_name)

        msg = r"out of bounds"
        with pytest.raises(AxisError, match=msg):
            func(in_num, axis=-(ndim + 1))

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_out_float(self, func_name):
        shape = (5,) * 3
        func = getattr(num, func_name)

        in_num = np.random.random(shape)
        res_out = np.random.random(size=(1, 5, 5))

        msg = r"output array must have int64 dtype"
        with pytest.raises(ValueError, match=msg):
            func(in_num, out=res_out)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    def test_out_shape_mismatch(self, func_name):
        ndim = 3
        shape = (5,) * ndim
        func = getattr(num, func_name)

        in_num = np.random.random(shape)
        res_out = np.random.randint(1, 10, size=shape)

        msg = r"the output shapes do not match"
        with pytest.raises(ValueError, match=msg):
            func(in_num, out=res_out)


class TestArgMaxAndArgMin:
    """
    These are positive cases compared with numpy
    """

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_argmax_and_argmin_basic(self, func_name, ndim, keepdims):
        shape = (5,) * ndim
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        assert np.array_equal(
            func_np(in_np, keepdims=keepdims),
            func_num(in_num, keepdims=keepdims),
        )

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_argmax_and_argmin_axis(self, func_name, ndim, keepdims):
        shape = (5,) * ndim
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        axis_list = list(range(in_num.ndim))
        axis_list.append(-(ndim - 1))

        for axis in axis_list:
            out_np = func_np(in_np, axis=axis, keepdims=keepdims)
            out_num = func_num(in_num, axis=axis, keepdims=keepdims)
            assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_argmax_and_argmin_out_0dim(self, func_name, keepdims):
        shape = (5,) * 0
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        res_np = np.random.randint(1, 10, size=())
        res_num = num.random.randint(1, 10, size=())

        func_np(in_np, out=res_np, keepdims=keepdims)
        func_num(in_num, out=res_num, keepdims=keepdims)

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_argmax_and_argmin_out_1dim(self, func_name, keepdims):
        shape = (5,) * 1
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        res_np = np.random.randint(1, 10, size=(1,) if keepdims else ())
        res_num = num.random.randint(1, 10, size=(1,) if keepdims else ())

        func_np(in_np, axis=0, out=res_np, keepdims=keepdims)
        func_num(in_num, axis=0, out=res_num, keepdims=keepdims)

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_argmax_and_argmin_out(self, func_name, ndim, keepdims):
        shape = (5,) * ndim
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        for axis in range(in_num.ndim):
            shape_list = list(shape)
            shape_list[axis] = 1
            shape_true = tuple(shape_list)

            res_np = np.random.randint(
                1, 10, size=shape_true if keepdims else (5,) * (ndim - 1)
            )
            res_num = num.random.randint(
                1, 10, size=shape_true if keepdims else (5,) * (ndim - 1)
            )

            func_np(in_np, axis=axis, out=res_np, keepdims=keepdims)
            func_num(in_num, axis=axis, out=res_num, keepdims=keepdims)

            assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
