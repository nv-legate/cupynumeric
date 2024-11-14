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

import cupynumeric as num

# cupynumeric.searchsorted(a: ndarray, v: Union[int, float, ndarray],
# side: Literal['left', 'right'] = 'left',
# sorter: Optional[ndarray] = None) → Union[int, ndarray]

# ndarray.searchsorted(v, side='left', sorter=None)


SIDES = ["left", "right"]

STANDARD_CASES = [
    (156, np.uint8),
    (123, np.uint16),
    (241, np.uint32),
    (1, np.uint64),
    (21, np.int8),
    (5, np.int16),
    (34, np.int32),
    (11, np.int64),
    (31, np.float32),
    (11, np.float64),
    (422, np.double),
    (220, np.double),
    (244, np.complex64),
    (24, np.complex128),
    (220, np.complex128),
    (0, np.uint32),
]

DTYPE_CASES = [
    (3, np.uint64, np.float32),
    (51, np.uint32, np.complex64),
    (23, np.uint32, np.float64),
    (51, np.complex64, np.float64),
    (21, np.complex64, np.int32),
    (22, np.complex128, np.float32),
]


class TestSearchSortedErrors(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        expected_exc = AttributeError
        with pytest.raises(expected_exc):
            np.searchsorted(None, 10)
            # Numpy raises ValueError:
            # object of too small depth for desired array
        with pytest.raises(expected_exc):
            num.searchsorted(None, 10)
            # cuNemeric raises AttributeError: 'NoneType' object
            # has no attribute 'searchsorted'

    @pytest.mark.xfail
    def test_val_none(self):
        arr = [2, 3, 10, 9]
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.searchsorted(arr, None)
            # Numpy raises TypeError: '<' not supported between
            # instances of 'NoneType' and 'NoneType'
        with pytest.raises(expected_exc):
            num.searchsorted(arr, None)
            # cuPyNumeric raises AssertionError
            #       if self.deferred is None:
            #           if self.parent is None:
            #    >          assert self.runtime.is_supported_dtype
            #                    (self.array.dtype)
            #    E               AssertionError
            # cupynumeric/cupynumeric/eager.py:to_deferred_array()

    @pytest.mark.xfail
    def test_side_invalid(self):
        arr = [2, 3, 10, 9]
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.searchsorted(arr, 10, "hi")
            # Numpy raises ValueError: search side must be 'left' or 'right'
            # (got 'hi')
        with pytest.raises(expected_exc):
            num.searchsorted(arr, 10, "hi")
            # cuPyNumeric passed, and result is the same as that of 'right'.

    def test_ndim_mismatch(self):
        a = np.random.random((5, 5, 5))
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.searchsorted(a, 5)
        with pytest.raises(expected_exc):
            np.searchsorted(a, 5)

    @pytest.mark.xfail
    def test_sorter_ndim_mismatch(self):
        a = np.random.randint(-100, 100, size=100)
        v = np.random.randint(-100, 100, size=10)
        a_argsorted = np.random.random((5, 5, 5))
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.searchsorted(a, v, sorter=a_argsorted)
        with pytest.raises(expected_exc):
            # Numpy raises TypeError
            np.searchsorted(a, v, sorter=a_argsorted)

    def test_sorter_shape_mismatch(self):
        a = np.random.randint(-100, 100, size=100)
        v = np.random.randint(-100, 100, size=10)
        a_argsorted = np.random.randint(-100, 100, size=10)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.searchsorted(a, v, sorter=a_argsorted)
        with pytest.raises(expected_exc):
            np.searchsorted(a, v, sorter=a_argsorted)

    @pytest.mark.xfail
    def test_sorter_dtype_mismatch(self):
        a = np.random.randint(-100, 100, size=100)
        v = np.random.randint(-100, 100, size=10)
        a_argsorted = np.random.random(size=100)
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.searchsorted(a, v, sorter=a_argsorted)
        with pytest.raises(expected_exc):
            # Numpy raises TypeError
            np.searchsorted(a, v, sorter=a_argsorted)


def generate_random(volume, datatype):
    a_np = None
    if np.issubdtype(datatype, np.integer):
        a_np = np.array(
            np.random.randint(
                np.iinfo(datatype).min,
                np.iinfo(datatype).max,
                size=volume,
                dtype=datatype,
            ),
            dtype=datatype,
        )
    elif np.issubdtype(datatype, np.floating):
        a_np = np.array(np.random.random(size=volume), dtype=datatype)
    elif np.issubdtype(datatype, np.complexfloating):
        a_np = np.array(
            np.random.random(size=volume) + np.random.random(size=volume) * 1j,
            dtype=datatype,
        )
    else:
        assert False
    return a_np


def check_api(a, dtype2=None, v=None, side="left"):
    a_argsorted = np.argsort(a)
    if v is None:
        if dtype2 is not None:
            v = generate_random(10, dtype2)
        else:
            v = generate_random(10, a.dtype)

    a_num = num.array(a)
    v_num = num.array(v)

    a_num_argsorted = num.array(a_argsorted)

    res_np = a.searchsorted(v, side=side, sorter=a_argsorted)
    res_num = a_num.searchsorted(v_num, side=side, sorter=a_num_argsorted)
    assert num.array_equal(res_np, res_num)

    res_np = np.searchsorted(a, v, side=side, sorter=a_argsorted)
    res_num = num.searchsorted(a_num, v_num, side=side, sorter=a_num_argsorted)
    assert num.array_equal(res_np, res_num)


@pytest.mark.parametrize("side", SIDES)
def test_empty_v(side):
    check_api(np.arange(25), None, np.arange(0), side)
    check_api(np.array([]), side=side)
    check_api(np.arange(0), None, np.arange(0), side=side)


@pytest.mark.parametrize("volume, dtype1, dtype2", DTYPE_CASES, ids=str)
@pytest.mark.parametrize("side", SIDES)
def test_dtype_conversions(volume, dtype1, dtype2, side):
    check_api(generate_random(volume, dtype1), dtype2, side=side)


@pytest.mark.parametrize("volume, dtype", STANDARD_CASES, ids=str)
@pytest.mark.parametrize("side", SIDES)
def test_standard_cases(volume, dtype, side):
    check_api(generate_random(volume, dtype), side=side)


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("side", SIDES)
def test_ndim(ndim, side):
    a = np.random.randint(-100, 100, size=100)
    v = np.random.randint(-100, 100, size=2**ndim).reshape(
        tuple(2 for i in range(ndim))
    )
    check_api(a, None, v, side)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
