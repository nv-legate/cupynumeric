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

from itertools import combinations_with_replacement

import numpy as np
import pytest

import cupynumeric as num

SCALARS_INF = (np.inf, -np.inf, np.nan, 0)
ARRAYS_INF = ([np.inf, -np.inf, np.nan, 0],)


@pytest.mark.parametrize("x", SCALARS_INF + ARRAYS_INF)
@pytest.mark.parametrize("func_name", ("isposinf", "isneginf"))
def test_inf_basic(func_name, x):
    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    assert np.array_equal(func_np(x), func_num(x))


@pytest.mark.parametrize("out_dt", (bool, int, float))
@pytest.mark.parametrize("x", (np.inf,) + ARRAYS_INF)
@pytest.mark.parametrize("func_name", ("isposinf", "isneginf"))
def test_inf_out(func_name, x, out_dt):
    res_shape = (4,)
    res_np = np.empty(res_shape, dtype=out_dt)
    res_num = num.empty(res_shape, dtype=out_dt)

    func_np = getattr(np, func_name)
    func_num = getattr(num, func_name)

    func_np(x, out=res_np)
    func_num(x, out=res_num)
    assert np.array_equal(res_np, res_num)


class TestInfErrors:
    @pytest.mark.parametrize("func_name", ("isposinf", "isneginf"))
    def test_out_invalid_shape(self, func_name):
        expected_exc = ValueError
        x = [np.inf, -np.inf, np.nan, 0]
        res_shape = (3,)
        res_np = np.empty(res_shape)
        res_num = num.empty(res_shape)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        with pytest.raises(expected_exc):
            func_np(x, out=res_np)
        with pytest.raises(expected_exc):
            func_num(x, out=res_num)


SCALARS = (pytest.param("a string", marks=pytest.mark.xfail), False)
ARRAYS = (
    [1.0, 2.0, 3.0],
    [1.0 + 0j, 2.0 + 0j, 3.0 + 0j],
    [1.0 + 1j, 2.0 + 0j, 3.0 + 1j],
)


@pytest.mark.parametrize("x", SCALARS + ARRAYS)
def test_isreal(x):
    # for x is 'a string', np.isreal is False, num.isreal is Array(True)
    assert np.array_equal(np.isreal(x), num.isreal(x))


@pytest.mark.parametrize("x", SCALARS + ARRAYS)
def test_iscomplex(x):
    assert np.array_equal(np.iscomplex(x), num.iscomplex(x))


@pytest.mark.parametrize("x", SCALARS)
def test_isrealobj_scalar(x):
    assert np.array_equal(np.isrealobj(x), num.isrealobj(x))


@pytest.mark.parametrize("x", SCALARS)
def test_iscomplexobj_scalar(x):
    assert np.array_equal(np.iscomplexobj(x), num.iscomplexobj(x))


@pytest.mark.parametrize("x", ARRAYS)
def test_isrealobj_array(x):
    in_np = np.array(x)
    in_num = num.array(x)

    assert np.array_equal(np.isrealobj(in_np), num.isrealobj(in_num))
    assert np.array_equal(np.isrealobj(in_np), num.isrealobj(in_np))


@pytest.mark.parametrize("x", ARRAYS)
def test_iscomplexobj_array(x):
    in_np = np.array(x)
    in_num = num.array(x)

    assert np.array_equal(np.iscomplexobj(in_np), num.iscomplexobj(in_num))
    assert np.array_equal(np.iscomplexobj(in_np), num.iscomplexobj(in_np))


@pytest.mark.parametrize("x", (1.0, True, [1, 2, 3]))
def test_isscalar(x):
    assert np.isscalar(x) is num.isscalar(x)


def test_isscalar_array():
    in_np = np.array([1, 2, 3])
    in_num = num.array([1, 2, 3])
    assert np.isscalar(in_np) is num.isscalar(in_num) is False

    # NumPy's scalar reduction returns a Python scalar
    assert num.isscalar(np.sum(in_np)) is True
    # but cuPyNumeric's scalar reduction returns a 0-D array that behaves like
    # a deferred scalar
    assert num.isscalar(num.sum(in_np)) is False


SCALAR_PAIRS = (
    (0, -1e-8),
    (1e10, 1.00001e10),
    (1 + 1j, 1 + 1.00001j),
    (0, -0.000001),
    (1e10, 1.0001e10),
    (1 + 1j, 1 + 1.0001j),
)


@pytest.mark.parametrize(("a", "b"), SCALAR_PAIRS)
def test_isclose_scalars(a, b):
    out_np = np.isclose(a, b)
    out_num = num.isclose(a, b)
    assert np.array_equal(out_np, out_num)


def test_isclose_arrays():
    in1_np = np.random.rand(10)
    in2_np = in1_np + np.random.uniform(low=5e-09, high=2e-08, size=10)
    in1_num = num.array(in1_np)
    in2_num = num.array(in2_np)

    out_np = np.isclose(in1_np, in2_np)
    out_num = num.isclose(in1_num, in2_num)
    assert np.array_equal(out_np, out_num)


SHAPES_BROADCASTING = (
    (1, 3),
    (2, 3),
    (1, 2, 3),
    (2, 2, 3),
    (1,),
    (1, 1),
    (1, 2, 1),
    (2, 2, 1),
)


@pytest.mark.parametrize(
    "shape_b", SHAPES_BROADCASTING, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_isclose_broadcast(shape_b):
    len_in_arr = 20
    in1_np = np.random.rand(len_in_arr)
    in2_np = in1_np + np.random.uniform(low=5e-09, high=2e-08, size=len_in_arr)

    shape_a = (3,)
    size_a = np.prod(shape_a)
    a_np = np.array([in1_np[i % len_in_arr] for i in range(size_a)]).reshape(
        shape_a
    )

    size_b = np.prod(shape_b)
    b_np = np.array([in2_np[i % len_in_arr] for i in range(size_b)]).reshape(
        shape_b
    )

    a_num = num.array(a_np)
    b_num = num.array(b_np)
    out_np = np.isclose(a_np, b_np)
    out_num = num.isclose(a_num, b_num)
    assert np.array_equal(out_np, out_num)


EMPTY_ARRAY_PAIRS = (([], []), ([], [[]]), ([[]], [[]]))


@pytest.mark.parametrize(("a", "b"), EMPTY_ARRAY_PAIRS)
def test_isclose_empty_arrays(a, b):
    out_np = np.isclose(a, b)
    out_num = num.isclose(a, b)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize(("rtol", "atol"), ((1e-04, 1e-06), (1e-06, 1e-09)))
def test_isclose_arrays_rtol_atol(rtol, atol):
    in1_np = np.random.rand(10)
    in2_np = in1_np + np.random.uniform(low=5e-09, high=2e-08, size=10)
    in1_num = num.array(in1_np)
    in2_num = num.array(in2_np)

    out_np = np.isclose(in1_np, in2_np, rtol=rtol, atol=atol)
    out_num = num.isclose(in1_num, in2_num, rtol=rtol, atol=atol)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize(
    "equal_nan", (False, pytest.param(True, marks=pytest.mark.xfail))
)
def test_isclose_euqal_nan(equal_nan):
    # If equal_nan is True,
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    values = [np.inf, -np.inf, np.nan, 0.0, -0.0]
    pairs = tuple(combinations_with_replacement(values, 2))
    in1_np = np.array([x for x, _ in pairs])
    in2_np = np.array([y for _, y in pairs])
    in1_num = num.array(in1_np)
    in2_num = num.array(in2_np)

    out_np = np.isclose(in1_np, in2_np, equal_nan=equal_nan)
    out_num = num.isclose(in1_num, in2_num, equal_nan=equal_nan)
    assert np.array_equal(out_np, out_num)


class TestIsCloseErrors:
    def test_arrays_invalid_shape(self):
        expected_exc = ValueError
        a_np = np.random.rand(6)
        b_np = a_np + np.random.uniform(low=5e-09, high=2e-08, size=6)
        in1_np = a_np.reshape((2, 3))
        in2_np = b_np.reshape((3, 2))
        in1_num = num.array(in1_np)
        in2_num = num.array(in2_np)

        with pytest.raises(expected_exc):
            np.isclose(in1_np, in2_np)
        with pytest.raises(expected_exc):
            num.isclose(in1_num, in2_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
