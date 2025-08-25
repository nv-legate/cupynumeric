# Copyright 2025 NVIDIA Corporation
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
from typing import Any
from utils.comparisons import allclose
from utils.generators import mk_seq_array

import cupynumeric as num


@pytest.mark.parametrize("endpoint", (True, False, None))
@pytest.mark.parametrize("number", (0, 1, 10))
@pytest.mark.parametrize("base", (10.0, 2.0, np.e, 1000))
@pytest.mark.parametrize(
    "values", ((0, 3), (1.0, 2.0), (-1, 1), (0.5, 2.5), (0, 0), (-1, -1))
)
def test_scalar_basic(values, base, number, endpoint):
    start, stop = values
    result_np = np.logspace(
        start, stop, num=number, endpoint=endpoint, base=base
    )
    result_num = num.logspace(
        start, stop, num=number, endpoint=endpoint, base=base
    )
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("endpoint", (True, False, None))
@pytest.mark.parametrize("number", (0, 1, 10))
@pytest.mark.parametrize("base", (10.0, 2.0, np.e))
def test_arrays_basic(number, endpoint, base):
    shape = (2, 2, 3)
    np_start = mk_seq_array(np, shape)
    num_start = mk_seq_array(num, shape)
    np_stop = mk_seq_array(np, shape) + 1
    num_stop = mk_seq_array(num, shape) + 1
    result_np = np.logspace(
        np_start, np_stop, num=number, endpoint=endpoint, base=base
    )
    result_num = num.logspace(
        num_start, num_stop, num=number, endpoint=endpoint, base=base
    )
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("shape", ((0,), (3,), (2, 1)))
def test_array_with_scalar(shape):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    scalar = 1.0

    result_np = np.logspace(np_arr, scalar)
    result_num = num.logspace(num_arr, scalar)
    assert allclose(result_np, result_num)

    result_np = np.logspace(scalar, np_arr)
    result_num = num.logspace(scalar, num_arr)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("endpoint", (True, False))
@pytest.mark.parametrize("shape", ((0,), (2, 1)))
def test_empty_array(shape, endpoint):
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    result_np = np.logspace(np_arr, [], endpoint=endpoint)
    result_num = num.logspace(num_arr, [], endpoint=endpoint)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("base", (-10, -1000, -10000))
def test_negative_base(base):
    start, stop = 0, 2
    result_np = np.logspace(start, stop, base=base)
    result_num = num.logspace(start, stop, base=base)
    assert np.allclose(result_np, result_num, equal_nan=True)


@pytest.mark.parametrize(
    "args, kwargs",
    [
        # 10**310 is larger than float64 max
        ((300, 310), {"num": 3}),
        # 10**10000 is way beyond float64 max
        ((0, 10000), {"num": 3}),
        # 2**1100 is inf
        ((0, 1100), {"num": 3, "base": 2}),
        # 2**1024 is inf
        ((1024, 1025), {"num": 2, "base": 2}),
        # e**800 is inf
        ((0, 800), {"num": 3, "base": np.e}),
        # 1.5**5000 is inf
        ((0, 5000), {"num": 3, "base": 1.5}),
    ],
)
def test_logspace_overflow(args, kwargs):
    result_np = np.logspace(*args, **kwargs)
    result_num = num.logspace(*args, **kwargs)
    assert np.allclose(result_np, result_num, equal_nan=True)


@pytest.mark.parametrize(
    "dtype",
    (None, np.float32, np.float64, np.int64, np.int32, np.int16, np.int8),
)
def test_dtype(dtype):
    start, stop = 0, 2
    result_np = np.logspace(start, stop, dtype=dtype)
    result_num = num.logspace(start, stop, dtype=dtype)
    assert result_np.dtype == result_num.dtype
    assert allclose(result_np, result_num)


def test_single_point():
    result_np = np.logspace(2, 2, 1)
    result_num = num.logspace(2, 2, 1)
    assert allclose(result_np, result_num)


def test_zero_points():
    result_np = np.logspace(0, 1, 0)
    result_num = num.logspace(0, 1, 0)
    assert allclose(result_np, result_num)


def test_num_negative():
    start, stop = 0, 2
    message = "Number of samples, -1, must be non-negative."
    with pytest.raises(ValueError, match=message):
        np.logspace(start, stop, num=-1)

    with pytest.raises(ValueError, match=message):
        num.logspace(start, stop, num=-1)


@pytest.mark.parametrize("start, stop", [(None, 2), (0, None), (None, None)])
def test_none_cases(start, stop):
    # Note: both fail at linespace but different error messages
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\) for -: '[^']+' and '[^']+'",
    ):
        np.logspace(start, stop)

    with pytest.raises((AttributeError, TypeError)):
        num.logspace(start, stop)


def test_complex_broadcasting_multidim_base():
    start = [0, 1]
    stop = [2, 3]
    base = [[2, 10], [5, 100]]

    result_np = np.logspace(start, stop, base=base)
    result_num = num.logspace(start, stop, base=base)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("axis", [0, -1])
def test_array_base_with_axis(axis):
    start = [0, 1]
    stop = [2, 3]
    base = [2, 10]

    result_np = np.logspace(start, stop, base=base, axis=axis)
    result_num = num.logspace(start, stop, base=base, axis=axis)
    assert allclose(result_np, result_num)


def test_scalar_wrapped_in_array_base():
    start, stop = 0, 2
    # Scalar wrapped in 0-d array
    base_scalar = np.array(10.0)

    result_np = np.logspace(start, stop, base=base_scalar)
    result_num = num.logspace(start, stop, base=base_scalar)
    assert allclose(result_np, result_num)


def test_base_1d_array():
    start, stop = 0, 2
    base_1d = mk_seq_array(np, (10000,))

    result_np = np.logspace(start, stop, base=base_1d)
    result_num = num.logspace(start, stop, base=base_1d)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize(
    "start, stop, base",
    [
        # 3d broadcasting
        ([[[0]], [[1]]], [2, 3], [[2], [10]]),
        # edge case single element arrays
        ([[0]], [2], [[10]]),
        # mixed scalar/array complex
        (0, [[1, 2], [3, 4]], [2, 10]),
    ],
)
def test_logspace_broadcast_cases(start, stop, base):
    result_np = np.logspace(start, stop, base=base)
    result_num = num.logspace(start, stop, base=base)
    assert allclose(result_np, result_num)


def test_large_dimensional_broadcast():
    start = np.ones((1, 2, 1, 3))
    stop = np.ones((2, 1, 4, 1))
    base = np.ones((1, 1, 4, 3))

    result_np = np.logspace(start, stop, base=base)
    result_num = num.logspace(start, stop, base=base)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_axis_with_complex_broadcasting(axis):
    start = [[0, 1]]
    stop = [[2], [3]]
    base = [2, 10]

    result_np = np.logspace(start, stop, base=base, axis=axis)
    result_num = num.logspace(start, stop, base=base, axis=axis)
    assert allclose(result_np, result_num)


def test_logspace_broadcast_shape_mismatch():
    start = mk_seq_array(np, (2,))
    stop = mk_seq_array(np, (3,))
    base = mk_seq_array(np, (4,))

    start_num = num.array(start)
    stop_num = num.array(stop)
    base_num = num.array(base)

    message = (
        r"shape mismatch: objects cannot be broadcast to a single shape\.  "
        r"Mismatch is between arg 0 with shape \(2,\) and arg 1 with shape "
        r"\(3,\)\."
    )

    with pytest.raises(ValueError, match=message):
        np.logspace(start, stop, base=base)
    with pytest.raises(ValueError, match=message):
        num.logspace(start_num, stop_num, base=base_num)


@pytest.mark.parametrize("num_points", (0, 1, 5))
def test_logspace_base_one(num_points):
    start = 0
    stop = 10
    result_np = np.logspace(start, stop, num=num_points, base=1)
    result_num = num.logspace(start, stop, num=num_points, base=1)
    assert allclose(result_np, result_num)


def test_logspace_base_near_zero():
    start = 0
    stop = 5
    base = 0.1
    result_np = np.logspace(start, stop, base=base)
    result_num = num.logspace(start, stop, base=base)
    assert allclose(result_np, result_num)


@pytest.mark.parametrize("start_dtype", (np.int32, np.float32, np.float64))
@pytest.mark.parametrize("stop_dtype", (np.int32, np.float32, np.float64))
@pytest.mark.parametrize("base_dtype", (np.int32, np.float32, np.float64))
def test_mixed_dtypes(start_dtype, stop_dtype, base_dtype):
    start = np.array(0, dtype=start_dtype)
    stop = np.array(2, dtype=stop_dtype)
    base = np.array(10, dtype=base_dtype)

    start_num = num.array(start)
    stop_num = num.array(stop)
    base_num = num.array(base)

    result_np = np.logspace(start, stop, base=base)
    result_num = num.logspace(start_num, stop_num, base=base_num)
    assert np.allclose(result_np, result_num)


@pytest.mark.parametrize("start", [np.inf, -np.inf])
def test_logspace_start_inf(start):
    stop = 2
    result_np = np.logspace(start, stop, num=3)
    result_num = num.logspace(start, stop, num=3)
    assert np.allclose(result_np, result_num, equal_nan=True)


def test_logspace_start_nan():
    start = np.nan
    stop = 2
    result_np = np.logspace(start, stop, num=3)
    result_num = num.logspace(start, stop, num=3)
    assert np.allclose(result_np, result_num, equal_nan=True)


def test_logspace_very_large_num():
    start = 0
    stop = 10
    num_points = 100000
    result_np = np.logspace(start, stop, num=num_points)
    result_num = num.logspace(start, stop, num=num_points)
    assert allclose(result_np, result_num)


def test_input_immutability():
    original_start = np.array([0.0, 1.0])
    start_copy_np = original_start.copy()
    start_copy_num = original_start.copy()

    original_stop = np.array([2.0, 3.0])
    stop_copy_np = original_stop.copy()
    stop_copy_num = original_stop.copy()

    original_base = np.array([10.0, 2.0])
    base_copy_np = original_base.copy()
    base_copy_num = original_base.copy()

    np.logspace(start_copy_np, stop_copy_np, base=base_copy_np)
    num.logspace(start_copy_num, stop_copy_num, base=base_copy_num)

    # Assert that the input arrays remain unchanged
    assert np.array_equal(original_start, start_copy_np)
    assert np.array_equal(original_start, start_copy_num)
    assert np.array_equal(original_stop, stop_copy_np)
    assert np.array_equal(original_stop, stop_copy_num)
    assert np.array_equal(original_base, base_copy_np)
    assert np.array_equal(original_base, base_copy_num)


@pytest.mark.parametrize("number", [None, 0.5, "1"])
def test_invalid_number(number: Any) -> None:
    start = 0
    stop = 1
    np_expected_error = "cannot be interpreted as an integer"
    num_expected_error = "must be an integer"
    with pytest.raises(TypeError, match=np_expected_error):
        np.logspace(start, stop, num=number)
    with pytest.raises(TypeError, match=num_expected_error):
        num.logspace(start, stop, num=number)


def test_invalid_axis() -> None:
    start = 0
    stop = 10
    number = 3
    axis = 1
    expected_error = "axis 1 is out of bounds for array of dimension 1"
    with pytest.raises(np.exceptions.AxisError, match=expected_error):
        np.logspace(start, stop, num=number, axis=axis)
    with pytest.raises(np.exceptions.AxisError, match=expected_error):
        num.logspace(start, stop, num=number, axis=axis)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
