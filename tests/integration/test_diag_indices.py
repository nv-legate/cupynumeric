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


@pytest.mark.parametrize("n", [10, -10.5, -1])
def test_diag_indices_default_ndim(n):
    a_np = np.diag_indices(n)
    a_num = num.diag_indices(n)
    assert np.array_equal(a_np, a_num)


@pytest.mark.parametrize("ndim", MAX_DIM_RANGE)
def test_diag_indices_basic(ndim):
    a_np = np.diag_indices(10, ndim)
    a_num = num.diag_indices(10, ndim)
    assert np.array_equal(a_np, a_num)


@pytest.mark.parametrize("n", [0, 0.0, 1, 10.5])
@pytest.mark.parametrize("ndim", [-4, 0, 1])
def test_diag_indices(n, ndim):
    a_np = np.diag_indices(n, ndim)
    a_num = num.diag_indices(n, ndim)
    assert np.array_equal(a_np, a_num)


class TestDiagIndicesErrors:
    @pytest.mark.xfail
    @pytest.mark.parametrize("n", [-10.5, -1])
    def test_negative_n_DIVERGENCE(self, n):
        # np.diag_indices(-10.5) returns empty 2-D array, dtype=float64
        # np.diag_indices(-1) returns empty 2-D array, dtype=int32
        # num.diag_indices(-10.5) raises ValueError
        # num.diag_indices(-1) raises ValueError
        a_np = np.diag_indices(n)
        a_num = num.diag_indices(n)
        assert np.array_equal(a_np, a_num)

    def test_none_n(self):
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            num.diag_indices(None)

    @pytest.mark.parametrize("ndim", [-1.5, 0.0, 1.5])
    def test_float_ndim(self, ndim):
        msg = "can't multiply sequence by non-int of type 'float'"
        with pytest.raises(TypeError, match=msg):
            num.diag_indices(10, ndim)

    def test_none_ndim(self):
        msg = "can't multiply sequence by non-int of type 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            num.diag_indices(10, None)


@pytest.mark.parametrize("size", [(5,), (0,)], ids=str)
@pytest.mark.parametrize("ndim", TWO_MAX_DIM_RANGE)
def test_diag_indices_from_basic(size, ndim):
    shape = size * ndim
    a = np.ones(shape, dtype=int)
    a_num = num.array(a)
    a_np = np.diag_indices_from(a)
    a_num = num.diag_indices_from(a_num)
    assert np.array_equal(a_np, a_num)


class TestDiagIndicesFromErrors:
    @pytest.mark.parametrize("size", [(5,), (0,)], ids=str)
    def test_1d(self, size):
        a = np.ones(size, dtype=int)
        msg = "input array must be at least 2-d"
        with pytest.raises(ValueError, match=msg):
            num.diag_indices_from(a)

    def test_0d(self):
        a = np.array(3)
        msg = "input array must be at least 2-d"
        with pytest.raises(ValueError, match=msg):
            num.diag_indices_from(a)

    @pytest.mark.parametrize("size", [(5, 2), (5, 0, 0)], ids=str)
    def test_unequal_length(self, size):
        a = np.ones(size, dtype=int)
        msg = "All dimensions of input must be of equal length"
        with pytest.raises(ValueError, match=msg):
            num.diag_indices_from(a)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
