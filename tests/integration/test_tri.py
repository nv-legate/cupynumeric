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
from utils.utils import check_module_function

import cupynumeric as num

KS = (0, -1, 1, -2, 2)
N = 100


@pytest.mark.parametrize("n", (0, 1, N), ids=lambda n: f"(n={n})")
def test_tri_n(n):
    print_msg = f"np & cupynumeric.tri({n})"
    check_module_function("tri", [n], {}, print_msg)


@pytest.mark.parametrize("k", KS + (-N, N), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize("m", (1, 10, N), ids=lambda m: f"(M={m})")
@pytest.mark.parametrize("n", (1, N), ids=lambda n: f"(n={n})")
def test_tri_full(n, m, k):
    print_msg = f"np & cupynumeric.tri({n}, k={k}, M={m})"
    check_module_function("tri", [n], {"k": k, "M": m}, print_msg)


@pytest.mark.parametrize("m", (0, None), ids=lambda m: f"(M={m})")
def test_tri_m(m):
    print_msg = f"np & cupynumeric.tri({N}, M={m})"
    check_module_function("tri", [N], {"M": m}, print_msg)


DTYPES = (int, float, bool, pytest.param(None, marks=pytest.mark.xfail))


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_tri_dtype(dtype):
    # cuPyNumeric: returns an array with dtype=int
    # Numpy: returns an array with dtype=float
    print_msg = f"np & cupynumeric.tri({N}, dtype={dtype})"
    check_module_function("tri", [N], {"dtype": dtype}, print_msg)


@pytest.mark.xfail
@pytest.mark.parametrize("k", (-10.5, 0.0, 10.5), ids=lambda k: f"(k={k})")
def test_tri_float_k(k):
    # cuPyNumeric: struct.error: required argument is not an integer
    # Numpy: pass
    print_msg = f"np & cupynumeric.tri({N}, k={k})"
    check_module_function("tri", [N], {"k": k}, print_msg)


class TestTriErrors:
    def test_negative_n(self):
        with pytest.raises(ValueError):
            num.tri(-100)

    @pytest.mark.xfail
    def test_negative_n_DIVERGENCE(self):
        # np.tri(-100) returns empty array
        # num.tri(-100) raises ValueError
        n = -100
        np_res = np.tri(n)
        num_res = num.tri(n)
        assert np.array_equal(np_res, num_res)

    @pytest.mark.parametrize("n", (-10.5, 0.0, 10.5))
    def test_float_n(self, n):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.tri(n)

    @pytest.mark.xfail
    @pytest.mark.parametrize("n", (-10.5, 0.0, 10.5))
    def test_float_n_DIVERGENCE(self, n):
        # np.tri(-10.5) returns empty array
        # np.tri(0.0) returns empty array
        # np.tri(10.5) returns array
        # num.tri(-10.5) raises TypeError
        # num.tri(0.0) raises TypeError
        # num.tri(10.5) raises TypeError
        np_res = np.tri(n)
        num_res = num.tri(n)
        assert np.array_equal(np_res, num_res)

    def test_negative_m(self):
        with pytest.raises(ValueError):
            num.tri(N, M=-10)

    @pytest.mark.xfail
    def test_negative_m_DIVERGENCE(self):
        # np.tri(100, M=-10) returns empty array
        # num.tri(100, M=-10) raises ValueError
        m = -10
        np_res = np.tri(N, M=m)
        num_res = num.tri(N, M=m)
        assert np.array_equal(np_res, num_res)

    @pytest.mark.parametrize("m", (-10.5, 0.0, 10.5))
    def test_float_m(self, m):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.tri(N, M=m)

    @pytest.mark.xfail
    @pytest.mark.parametrize("m", (-10.5, 0.0, 10.5))
    def test_float_m_DIVERGENCE(self, m):
        # np.tri(100, M=-10.5) returns empty array
        # np.tri(100, M=0.0) returns empty array
        # np.tri(100, M=10.5) returns array
        # num.tri(100, M=-10.5) raises TypeError
        # num.tri(100, M=0.0) raises TypeError
        # num.tri(100, M=10.5) raises TypeError
        np_res = np.tri(N, M=m)
        num_res = num.tri(N, M=m)
        assert np.array_equal(np_res, num_res)

    def test_n_none(self):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.tri(None)

    @pytest.mark.xfail
    def test_k_none(self):
        # In cuPyNumeric, it raises struct.error,
        # msg is required argument is not an integer
        # In Numpy, it raises TypeError,
        # msg is bad operand type for unary -: 'NoneType'
        with pytest.raises(TypeError):
            num.tri(N, k=None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
