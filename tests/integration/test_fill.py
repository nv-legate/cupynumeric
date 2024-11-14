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

import re

import numpy as np
import pytest

import cupynumeric as num

INF_VALUES = [-np.inf, np.inf]
FLOAT_FILL_VALUES = (-2.4e120, -1.3, 8.9e-130, 0.0, 5.7e-150, 0.6, 3.7e160)
FLOAT_BIG_VALUES = (-2.4e120, 3.7e160)


def test_fill_empty_array():
    a_np = np.array([])
    a_num = num.array(a_np)
    a_np.fill(1)
    a_num.fill(1)
    assert np.array_equal(a_np, a_num)


def test_fill_float_with_none():
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    a_np.fill(None)
    a_num.fill(None)
    assert np.array_equal(a_np, a_num, equal_nan=True)


def test_fill_float_with_nan():
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    a_np.fill(np.nan)
    a_num.fill(np.nan)
    assert np.array_equal(a_np, a_num, equal_nan=True)


def test_fill_int_with_none():
    a_np = np.full((2, 3), 1)
    a_num = num.array(a_np)
    # numpy fill with -9223372036854775808,
    # while cupynumeric raises TypeError
    #
    # Update (wonchan): Numpy 1.23.3 no longer fills
    # the array with -9223372036854775808 on 'array.fill(None)'
    # but raises the same exception as cuPyNumeric
    try:
        int(None)
    except TypeError as e:
        msg = re.escape(str(e))
    with pytest.raises(TypeError, match=msg):
        a_num.fill(None)


def test_fill_int_with_nan():
    a_np = np.full((2, 3), 1)
    a_num = num.array(a_np)
    # numpy fill with -9223372036854775808,
    # while cupynumeric raises ValueError
    msg = r"cannot convert float NaN to integer"
    with pytest.raises(ValueError, match=msg):
        a_num.fill(np.nan)


@pytest.mark.parametrize("value", INF_VALUES)
def test_fill_inf_to_int(value: float) -> None:
    a_np = np.full((2, 3), 1)
    a_num = num.array(a_np)
    # numpy fill with -9223372036854775808,
    # while cupynumeric raises OverflowError
    msg = r"cannot convert float infinity to integer"
    with pytest.raises(OverflowError, match=msg):
        a_num.fill(value)


@pytest.mark.parametrize("value", INF_VALUES)
def test_fill_inf_to_float(value: float) -> None:
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    a_np.fill(value)
    a_num.fill(value)
    assert np.array_equal(a_np, a_num)


@pytest.mark.parametrize("value", FLOAT_FILL_VALUES)
def test_fill_float_to_float(value: float) -> None:
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    a_np.fill(value)
    a_num.fill(value)
    assert np.array_equal(a_np, a_num)


@pytest.mark.parametrize("value", FLOAT_BIG_VALUES)
def test_fill_float_to_int(value: float) -> None:
    a_np = np.full((2, 3), 1)
    a_num = num.array(a_np)
    # numpy fill with -9223372036854775808,
    # while cupynumeric raises OverflowError
    msg = r"Python int too large to convert to C long"
    with pytest.raises(OverflowError, match=msg):
        a_num.fill(value)


def test_fill_int_to_float() -> None:
    a_np = np.full((2, 3), 0.5)
    a_num = num.array(a_np)
    a_np.fill(5)
    a_num.fill(5)
    assert np.array_equal(a_np, a_num)


@pytest.mark.xfail
def test_fill_string() -> None:
    a_list = ["hello", "hi"]
    a_np = np.array(a_list)
    a_num = num.array(a_np)
    a_np.fill("ok")
    a_num.fill("ok")
    assert np.array_equal(a_np, a_num)


def test_fill_string_to_int() -> None:
    a_np = np.full((2, 3), 3)
    a_num = num.array(a_np)
    msg = r"invalid literal for int()"
    with pytest.raises(ValueError, match=msg):
        a_num.fill("OK")


def test_fill_string_to_float() -> None:
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    msg = r"could not convert string to float"
    with pytest.raises(ValueError, match=msg):
        a_num.fill("OK")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
