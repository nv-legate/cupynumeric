#!/usr/bin/env python
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

import cupynumeric as num

SCALARS_TRUE_DEFAULT = ((0, -1e-8), (1e10, 1.00001e10), (1 + 1j, 1 + 1.00001j))

SCALARS_TRUE_INF = ((np.inf, np.inf), (-np.inf, -np.inf))


@pytest.mark.parametrize(("a", "b"), SCALARS_TRUE_DEFAULT + SCALARS_TRUE_INF)
def test_scalar_true(
    a: int | float | complex, b: int | float | complex
) -> None:
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)
    assert res_np is bool(res_num) is True

    res_np_swapped = np.allclose(b, a)
    res_num_swapped = num.allclose(b, a)
    assert res_np_swapped is bool(res_num_swapped) is True


SCALARS_FALSE_DEFAULT = (
    (0, -0.000001),
    (1e10, 1.0001e10),
    (1 + 1j, 1 + 1.0001j),
)

SCALARS_FALSE_INF = ((np.inf, -np.inf),)


@pytest.mark.parametrize(("a", "b"), SCALARS_FALSE_DEFAULT + SCALARS_FALSE_INF)
def test_scalar_false(
    a: int | float | complex, b: int | float | complex
) -> None:
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)
    assert res_np is bool(res_num) is False

    res_np_swapped = np.allclose(b, a)
    res_num_swapped = num.allclose(b, a)
    assert res_np_swapped is bool(res_num_swapped) is False


SHAPES = ((1,), (6,), (1, 1), (2, 3), (2, 3, 4))


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"(shape={shape})")
def test_array_true(shape: tuple[int, ...]) -> None:
    len_scalars = len(SCALARS_TRUE_DEFAULT)
    size = np.prod(shape)
    array = [SCALARS_TRUE_DEFAULT[i % len_scalars] for i in range(size)]
    a_np = np.array([x[0] for x in array]).reshape(shape)
    b_np = np.array([x[1] for x in array]).reshape(shape)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)
    assert res_np is bool(res_num) is True

    res_np_swapped = np.allclose(b_np, a_np)
    res_num_swapped = num.allclose(b_num, a_num)
    assert res_np_swapped is bool(res_num_swapped) is True


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"(shape={shape})")
def test_array_true_inf(shape: tuple[int, ...]) -> None:
    tup_scalars_true = SCALARS_TRUE_INF + SCALARS_TRUE_DEFAULT
    len_scalars = len(tup_scalars_true)
    size = np.prod(shape)
    array = [tup_scalars_true[i % len_scalars] for i in range(size)]
    a_np = np.array([x[0] for x in array]).reshape(shape).astype(float)
    b_np = np.array([x[1] for x in array]).reshape(shape).astype(float)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)
    assert res_np is bool(res_num) is True

    res_np_swapped = np.allclose(b_np, a_np)
    res_num_swapped = num.allclose(b_num, a_num)
    assert res_np_swapped is bool(res_num_swapped) is True


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"(shape={shape})")
def test_array_false(shape: tuple[int, ...]) -> None:
    len_scalars = len(SCALARS_TRUE_DEFAULT)
    size = np.prod(shape)
    array = [SCALARS_TRUE_DEFAULT[i % len_scalars] for i in range(size)]
    array[-1] = SCALARS_FALSE_DEFAULT[0]
    a_np = np.array([x[0] for x in array]).reshape(shape)
    b_np = np.array([x[1] for x in array]).reshape(shape)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)
    assert res_np is bool(res_num) is False

    res_np_swapped = np.allclose(b_np, a_np)
    res_num_swapped = num.allclose(b_num, a_num)
    assert res_np_swapped is bool(res_num_swapped) is False


SHAPES_BROADCASTING1 = ((1, 3), (2, 3), (1, 2, 3), (2, 2, 3))


@pytest.mark.parametrize(
    "shape_b", SHAPES_BROADCASTING1, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_broadcast_true1(shape_b: tuple[int, ...]) -> None:
    len_scalars = len(SCALARS_TRUE_DEFAULT)

    shape_a = (3,)
    size_a = np.prod(shape_a)
    array_a = [SCALARS_TRUE_DEFAULT[i % len_scalars] for i in range(size_a)]
    a_np = np.array([x[0] for x in array_a]).reshape(shape_a)

    size_b = np.prod(shape_b)
    array_b = [array_a[i % size_a] for i in range(size_b)]
    b_np = np.array([x[1] for x in array_b]).reshape(shape_b)

    a_num = num.array(a_np)
    b_num = num.array(b_np)
    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is bool(res_num) is True


@pytest.mark.parametrize(
    "shape_b", SHAPES_BROADCASTING1, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_broadcast_false1(shape_b: tuple[int, ...]) -> None:
    len_scalars = len(SCALARS_FALSE_DEFAULT)

    shape_a = (3,)
    size_a = np.prod(shape_a)
    array_a = [SCALARS_FALSE_DEFAULT[i % len_scalars] for i in range(size_a)]
    a_np = np.array([x[0] for x in array_a]).reshape(shape_a)

    size_b = np.prod(shape_b)
    array_b = [array_a[i % size_a] for i in range(size_b)]
    b_np = np.array([x[1] for x in array_b]).reshape(shape_b)

    a_num = num.array(a_np)
    b_num = num.array(b_np)
    print("a_num: ", a_num)
    print("b_num: ", b_num)
    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is bool(res_num) is False


SHAPES_BROADCASTING2 = ((1,), (1, 1), (1, 2, 1), (2, 2, 1))


@pytest.mark.parametrize(
    "shape_b", SHAPES_BROADCASTING2, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_broadcast_true2(shape_b: tuple[int, ...]) -> None:
    shape_a = (3,)
    size_a = np.prod(shape_a)
    a_np = np.array(
        [SCALARS_TRUE_DEFAULT[0][0] for _ in range(size_a)]
    ).reshape(shape_a)

    size_b = np.prod(shape_b)
    b_np = np.array(
        [SCALARS_TRUE_DEFAULT[0][1] for _ in range(size_b)]
    ).reshape(shape_b)

    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is bool(res_num) is True


@pytest.mark.parametrize(
    "shape_b", SHAPES_BROADCASTING2, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_broadcast_false2(shape_b: tuple[int, ...]) -> None:
    shape_a = (3,)
    size_a = np.prod(shape_a)
    a_np = np.array(
        [SCALARS_FALSE_DEFAULT[0][0] for _ in range(size_a)]
    ).reshape(shape_a)

    size_b = np.prod(shape_b)
    b_np = np.array(
        [SCALARS_FALSE_DEFAULT[0][1] for _ in range(size_b)]
    ).reshape(shape_b)

    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is bool(res_num) is False


@pytest.mark.parametrize(
    "equal_nan", (False, pytest.param(True, marks=pytest.mark.xfail))
)
@pytest.mark.parametrize(
    "arr",
    ([np.nan], [1, 2, np.nan], [[1, 2], [3, np.nan]]),
    ids=lambda arr: f"(arr={arr})",
)
def test_equal_nan_basic(
    arr: list[float] | list[list[float]], equal_nan: bool
) -> None:
    # If equal_nan is True,
    # In Numpy, it pass
    # In cuPyNumeric, it raises NotImplementedError
    res_np = np.allclose(arr, arr, equal_nan=equal_nan)
    res_num = num.allclose(arr, arr, equal_nan=equal_nan)
    assert res_np == res_num


EMPTY_ARRAY_PAIRS = (([], []), ([], [[]]), ([[]], [[]]))


@pytest.mark.parametrize(("a", "b"), EMPTY_ARRAY_PAIRS)
def test_empty_array(a: list[list[float]], b: list[list[float]]) -> None:
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)

    assert res_np is bool(res_num) is True


SCALAR_BROADCASTING = ((1e10, [1.00001e10]), (1e10, [[1.00001e10]]))


@pytest.mark.xfail
@pytest.mark.parametrize(("a", "b"), SCALAR_BROADCASTING)
def test_scalar_broadcasting(
    a: float | list[float], b: float | list[float]
) -> None:
    # for all cases,
    # In Numpy, it pass
    # In cuPyNumeric, it raises AttributeError:
    # 'Store' object has no attribute '_broadcast'
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)

    assert res_np is bool(res_num) is True


@pytest.mark.parametrize(("a", "b"), SCALARS_FALSE_DEFAULT)
def test_scalar_rtol_atol_true(
    a: int | float | complex, b: int | float | complex
) -> None:
    rtol = 1e-04
    atol = 1e-06

    res_np = np.allclose(a, b, rtol=rtol, atol=atol)
    res_num = num.allclose(a, b, rtol=rtol, atol=atol)

    assert res_np is bool(res_num) is True


@pytest.mark.parametrize(("a", "b"), SCALARS_TRUE_DEFAULT)
def test_scalar_rtol_atol_false(
    a: int | float | complex, b: int | float | complex
) -> None:
    rtol = 1e-06
    atol = 1e-09

    res_np = np.allclose(a, b, rtol=rtol, atol=atol)
    res_num = num.allclose(a, b, rtol=rtol, atol=atol)

    assert res_np is bool(res_num) is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
