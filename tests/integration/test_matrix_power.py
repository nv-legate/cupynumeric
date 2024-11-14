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
from utils.comparisons import allclose
from utils.generators import mk_0to1_array

import cupynumeric as num

# TODO: add negative exponents here, once they become supported
EXPONENTS = (0, 1, 2, 3, 5)


@pytest.mark.parametrize(
    "dtype",
    (
        np.float64,
        np.complex128,
        pytest.param(np.int32, marks=pytest.mark.xfail),
    ),
)
@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM - 2))
@pytest.mark.parametrize("exp", EXPONENTS)
def test_matrix_power(ndim, exp, dtype):
    # If dtype=np.int32 and exp greater than 1,
    # In Numpy, pass
    # In cuPyNumeric, raises TypeError: Unsupported type: int32
    shape = (3,) * ndim + (2, 2)
    a_np = mk_0to1_array(np, shape, dtype=dtype)
    a_num = mk_0to1_array(num, shape, dtype=dtype)
    res_np = np.linalg.matrix_power(a_np, exp)
    res_num = num.linalg.matrix_power(a_num, exp)
    assert allclose(res_np, res_num)


@pytest.mark.parametrize(
    "exp",
    (
        0,
        1,
        pytest.param(2, marks=pytest.mark.xfail),
        pytest.param(3, marks=pytest.mark.xfail),
    ),
)
def test_matrix_power_empty_matrix(exp):
    # If exp =2 or 3,
    # In Numpy, pass and returns empty array
    # In cuPyNumeric, raise AssertionError in _contract
    shape = (0, 0)
    a_np = mk_0to1_array(np, shape)
    a_num = mk_0to1_array(num, shape)
    res_np = np.linalg.matrix_power(a_np, exp)
    res_num = num.linalg.matrix_power(a_num, exp)
    assert np.array_equal(res_np, res_num)


class TestMatrixPowerErrors:
    @pytest.mark.parametrize("ndim", (0, 1), ids=lambda ndim: f"(ndim={ndim})")
    def test_matrix_ndim_smaller_than_two(self, ndim):
        shape = (3,) * ndim
        a_num = mk_0to1_array(num, shape)
        a_np = mk_0to1_array(np, shape)
        expected_exc = num.linalg.LinAlgError
        with pytest.raises(expected_exc):
            num.linalg.matrix_power(a_num, 1)
        with pytest.raises(expected_exc):
            np.linalg.matrix_power(a_np, 1)

    @pytest.mark.parametrize(
        "shape", ((2, 1), (2, 2, 1)), ids=lambda shape: f"(shape={shape})"
    )
    def test_matrix_not_square(self, shape):
        a_num = mk_0to1_array(num, shape)
        a_np = mk_0to1_array(np, shape)
        expected_exc = num.linalg.LinAlgError
        with pytest.raises(expected_exc):
            num.linalg.matrix_power(a_num, 1)
        with pytest.raises(expected_exc):
            np.linalg.matrix_power(a_np, 1)

    @pytest.mark.parametrize(
        "n", (-1.0, 1.0, [1], None), ids=lambda n: f"(n={n})"
    )
    def test_n_not_int(self, n):
        shape = (2, 2)
        a_num = mk_0to1_array(num, shape)
        a_np = mk_0to1_array(np, shape)
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            num.linalg.matrix_power(a_num, n)
        with pytest.raises(expected_exc):
            np.linalg.matrix_power(a_np, n)

    def test_n_negative_int(self):
        shape = (2, 2)
        n = -1
        a_num = mk_0to1_array(num, shape)
        with pytest.raises(NotImplementedError):
            num.linalg.matrix_power(a_num, n)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
