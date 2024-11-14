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

DTYPE_ALL = [
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
    np.float16,
    np.float32,
    np.float64,
    bool,
    np.complex64,
    np.complex128,
]
VALUES = [0, 1, 2, 100]
NEGATIVE_TYPE = [None, np.inf, -np.inf, -3.5, 3.5, 5j, 10 + 20j, -100 - 100j]
NEGATIVE_VALUE = [-1, -2]


@pytest.mark.parametrize("val", VALUES)
def test_value(val):
    res_np = np.identity(val)
    res_num = num.identity(val)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("neg_type", NEGATIVE_TYPE)
def test_value_negative_type(neg_type):
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        np.identity(neg_type)
    with pytest.raises(expected_exc):
        num.identity(neg_type)


@pytest.mark.parametrize("neg_val", NEGATIVE_VALUE)
def test_value_negative_value(neg_val):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.identity(neg_val)
    with pytest.raises(expected_exc):
        num.identity(neg_val)


@pytest.mark.parametrize("dtype", DTYPE_ALL)
def test_dtype(dtype):
    res_np = np.identity(5, dtype=dtype)
    res_num = num.identity(5, dtype=dtype)
    assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
