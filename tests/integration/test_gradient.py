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
from math import prod

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM

import cupynumeric as cn


def test_gradient_with_scalar_dx():
    f_np = np.arange(1000, dtype=float)
    f_cn = cn.array(f_np)
    res_np = np.gradient(f_np)
    res_cn = cn.gradient(f_cn)
    assert np.allclose(res_np, res_cn)


def test_gradient_1d():
    a_np = np.array(np.random.random(size=1000), dtype=float)
    f_np = np.sort(a_np)
    f_cn = cn.array(f_np)
    res_np = np.gradient(f_np)
    res_cn = cn.gradient(f_cn)
    assert np.allclose(res_np, res_cn)


@pytest.mark.parametrize(
    "ndim",
    (
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
@pytest.mark.parametrize("edge_order", [1, 2])
def test_nd_arrays(ndim, edge_order):
    shape = (5,) * ndim
    size = prod(shape)
    arr_np = np.random.randint(-100, 100, size, dtype=int)
    in_np = np.sort(arr_np).reshape(shape).astype(float)
    in_cn = cn.array(in_np)

    for a in range(0, ndim):
        res_np = np.gradient(in_np, axis=a, edge_order=edge_order)
        res_cn = np.gradient(in_cn, axis=a, edge_order=edge_order)
        assert np.allclose(res_np, res_cn)


@pytest.mark.parametrize(
    "ndim",
    (
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
@pytest.mark.parametrize("varargs", [0.5, 1, 2, 0.3, 0])
def test_scalar_varargs(ndim, varargs):
    shape = (5,) * ndim
    size = prod(shape)
    arr_np = np.random.randint(-100, 100, size, dtype=int)
    in_np = np.sort(arr_np).reshape(shape).astype(float)
    in_cn = cn.array(in_np)
    res_np = np.gradient(in_np, varargs)
    res_cn = cn.gradient(in_cn, varargs)
    assert np.allclose(res_np, res_cn, equal_nan=True)


@pytest.mark.parametrize(
    "ndim",
    (
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_array_1d_varargs(ndim):
    shape = (5,) * ndim
    size = prod(shape)
    arr_np = np.random.randint(-100, 100, size, dtype=int)
    in_np = np.sort(arr_np).reshape(shape).astype(float)
    in_cn = cn.array(in_np)
    varargs = list(i * 0.5 for i in range(ndim))
    res_np = np.gradient(in_np, *varargs)
    res_cn = cn.gradient(in_cn, *varargs)
    assert np.allclose(res_np, res_cn)


@pytest.mark.parametrize(
    "ndim",
    (
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_list_of_axes(ndim):
    shape = (5,) * ndim
    size = prod(shape)
    arr_np = np.random.randint(-100, 100, size, dtype=int)
    in_np = np.sort(arr_np).reshape(shape).astype(float)
    in_cn = cn.array(in_np)
    axes = tuple(i for i in range(ndim))
    res_np = np.gradient(in_np, axis=axes)
    res_cn = cn.gradient(in_cn, axis=axes)
    assert np.allclose(res_np, res_cn)


def test_varargs_coordinates():
    # Test gradient with varargs to specify coordinates
    x_np = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x_cn = cn.array(x_np)
    y_coordinates_np = np.array(
        [0.0, 1.0, 2.0]
    )  # Coordinates along the first dimension
    y_coordinates_cn = cn.array(y_coordinates_np)
    x_coordinates_np = np.array(
        [10.0, 20.0, 30.0]
    )  # Coordinates along the second dimension
    x_coordinates_cn = cn.array(x_coordinates_np)
    res_np = np.gradient(x_np, y_coordinates_np, x_coordinates_np)
    res_cn = np.gradient(x_cn, y_coordinates_cn, x_coordinates_cn)
    assert np.allclose(res_np, res_cn)


def test_mixed_varargs():
    # Test gradient with varargs to specify coordinates
    x_np = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x_cn = cn.array(x_np)
    y_coordinates_np = np.array(
        [0.0, 1.0, 2.0]
    )  # Coordinates along the first dimension
    y_coordinates_cn = cn.array(y_coordinates_np)
    res_np = np.gradient(x_np, y_coordinates_np, 0.5)
    res_cn = np.gradient(x_cn, y_coordinates_cn, 0.5)
    assert np.allclose(res_np, res_cn)


@pytest.mark.parametrize(
    "in_arr",
    [
        [],
        [1],
    ],
)
def test_corner_cases(in_arr):
    in_cn = cn.array(in_arr)
    with pytest.raises(ValueError):
        cn.gradient(in_cn)  # too small


@pytest.mark.parametrize("axis", [2, -4])
def test_invalid_axis(axis):
    # Test gradient with invalid axis
    x = cn.array([[1.0, 2.0, 4.0], [3.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.raises(ValueError):
        cn.gradient(x, axis=axis)  # Invalid axis for 2D array


@pytest.mark.parametrize("edge_order", [3, -1])
def test_invalid_edge_order(edge_order):
    # Test gradient with invalid edge_order
    x = cn.array([1.0, 2.0, 4.0, 7.0])
    with pytest.raises(ValueError):
        cn.gradient(x, edge_order=edge_order)  # Invalid edge_order


def test_invalid_varargs():
    # Test gradient with varargs
    x = cn.array([[1.0, 2.0, 4.0], [3.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.raises(TypeError):
        cn.gradient(x, 0.5, 1.0, 2.0)  # Too many varargs provided


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
