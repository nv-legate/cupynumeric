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
from utils.utils import MAX_DIM_RANGE, TWO_MAX_DIM_RANGE

import cupynumeric as num

FUNCS = ("amax", "amin")


@pytest.mark.parametrize("initial", (None, -2, 0, 0.5, 2))
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
@pytest.mark.parametrize("func_name", FUNCS)
def test_basic(func_name, ndim, keepdims, initial):
    shape = (5,) * ndim
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)
    kw = {} if initial is None else dict(initial=initial)

    res_np = func_np(in_np, keepdims=keepdims, **kw)
    res_num = func_num(in_num, keepdims=keepdims, **kw)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "src_dt",
    (
        np.int32,
        np.float64,
        pytest.param(np.complex128, marks=pytest.mark.xfail),
    ),
)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("func_name", FUNCS)
def test_src_dt(func_name, keepdims, src_dt):
    # For src_dt=np.complex128,
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    ndim = 3
    shape = (5,) * ndim
    in_np = np.random.randint(-5, 5, size=shape).astype(src_dt)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    assert np.array_equal(
        func_np(in_np, keepdims=keepdims),
        func_num(in_num, keepdims=keepdims),
    )


@pytest.mark.parametrize("initial", (None, -2, 0, 0.5, 2))
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
@pytest.mark.parametrize("func_name", FUNCS)
def test_axis(func_name, ndim, keepdims, initial):
    shape = (5,) * ndim
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)
    kw = {} if initial is None else dict(initial=initial)

    axis_list = list(range(in_num.ndim))
    axis_list.append(-ndim)

    for axis in axis_list:
        res_np = func_np(in_np, axis=axis, keepdims=keepdims, **kw)
        res_num = func_num(in_num, axis=axis, keepdims=keepdims, **kw)
        assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
@pytest.mark.parametrize("axes", ((-3, -1), (-1, 0), (-2, 2), (0, 2)))
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("func_name", FUNCS)
def test_axis_tuple(func_name, keepdims, axes):
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    shape = (3, 4, 5)
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    res_np = func_np(in_np, axis=axes, keepdims=keepdims)
    res_num = func_num(in_num, axis=axes, keepdims=keepdims)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("func_name", FUNCS)
def test_out_dim0(func_name, keepdims):
    shape = (5,) * 0
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    res_np = np.empty(())
    res_num = num.empty(())

    func_np(in_np, out=res_np, keepdims=keepdims)
    func_num(in_num, out=res_num, keepdims=keepdims)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("func_name", FUNCS)
def test_out_dim1(func_name, keepdims):
    shape = (5,) * 1
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    res_shape = (1,) if keepdims else ()
    res_np = np.empty(res_shape)
    res_num = num.empty(res_shape)

    func_np(in_np, axis=0, out=res_np, keepdims=keepdims)
    func_num(in_num, axis=0, out=res_num, keepdims=keepdims)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("initial", (None, -2, 0, 0.5, 2))
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("ndim", TWO_MAX_DIM_RANGE)
@pytest.mark.parametrize("func_name", FUNCS)
def test_out(func_name, ndim, keepdims, initial):
    shape = (5,) * ndim
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)
    kw = {} if initial is None else dict(initial=initial)

    for axis in range(in_num.ndim):
        shape_list = list(shape)
        shape_list[axis] = 1
        shape_true = tuple(shape_list)

        res_shape = shape_true if keepdims else (5,) * (ndim - 1)
        res_np = np.empty(res_shape)
        res_num = num.empty(res_shape)

        func_np(in_np, axis=axis, out=res_np, keepdims=keepdims, **kw)
        func_num(in_num, axis=axis, out=res_num, keepdims=keepdims, **kw)
        assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "out_dt",
    (
        np.int32,
        np.float64,
        pytest.param(np.complex128, marks=pytest.mark.xfail),
    ),
)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("func_name", FUNCS)
def test_out_with_dtype(func_name, keepdims, out_dt):
    # For out_dt=np.complex128
    # In Numpy, it pass
    # In cuPyNumeric, it raises KeyError
    ndim = 3
    shape = (5,) * ndim
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    for axis in range(in_num.ndim):
        shape_list = list(shape)
        shape_list[axis] = 1
        shape_true = tuple(shape_list)

        res_shape = shape_true if keepdims else (5,) * (ndim - 1)
        res_np = np.empty(res_shape, dtype=out_dt)
        res_num = num.empty(res_shape, dtype=out_dt)

        func_np(in_np, axis=axis, out=res_np, keepdims=keepdims)
        func_num(in_num, axis=axis, out=res_num, keepdims=keepdims)
        assert np.array_equal(res_np, res_num)


@pytest.mark.xfail
@pytest.mark.parametrize("func_name", FUNCS)
def test_where(func_name):
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    shape = (3, 4, 5)
    in_np = np.random.randint(-5, 5, size=shape)
    in_num = num.array(in_np)
    where_np = in_np > 0.5
    where_num = num.array(where_np)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    val = 0
    assert np.array_equal(
        func_np(in_np, initial=val, where=where_np),
        func_num(in_num, initial=val, where=where_num),
    )


class TestAmaxAminErrors:
    def setup_method(self):
        size = (3, 4, 5)
        self.arr_np = np.random.randint(-5, 5, size=size)
        self.arr_num = num.array(self.arr_np)

    @pytest.mark.parametrize("func_name", FUNCS)
    def test_empty_array(self, func_name):
        expected_exc = ValueError
        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        with pytest.raises(expected_exc):
            func_np([])
        with pytest.raises(expected_exc):
            func_num([])

    @pytest.mark.parametrize(
        "axis", (-4, 3), ids=lambda axis: f"(axis={axis})"
    )
    @pytest.mark.parametrize("func_name", FUNCS)
    def test_axis_out_of_bound(self, func_name, axis):
        expected_exc = ValueError
        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        with pytest.raises(expected_exc):
            func_np(self.arr_np, axis=axis)
        with pytest.raises(expected_exc):
            func_num(self.arr_num, axis=axis)

    @pytest.mark.parametrize(
        "axis_out_shape",
        (
            (None, (1,)),
            (1, (3, 4)),
        ),
        ids=lambda axis_out_shape: f"(axis_out_shape={axis_out_shape})",
    )
    @pytest.mark.parametrize("func_name", FUNCS)
    def test_out_invalid_shape(self, func_name, axis_out_shape):
        axis, out_shape = axis_out_shape
        expected_exc = ValueError
        out_np = np.empty(out_shape)
        out_num = num.empty(out_shape)
        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        with pytest.raises(expected_exc):
            func_np(self.arr_np, axis=axis, out=out_np)
        with pytest.raises(expected_exc):
            func_num(self.arr_num, axis=axis, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
