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
FUNCTIONS_INDICES = ("tril_indices", "triu_indices")
FUNCTIONS_INDICES_FROM = ("tril_indices_from", "triu_indices_from")
N = 100


def _test_from(func, shape, k):
    num_f = getattr(num, func)
    np_f = getattr(np, func)
    a = num.ones(shape, dtype=int)
    an = np.ones(shape, dtype=int)

    b = num_f(a, k=k)
    bn = np_f(an, k=k)
    assert num.array_equal(b, bn)


@pytest.mark.parametrize("n", (0, 1, 100), ids=lambda n: f"(n={n})")
@pytest.mark.parametrize("func", FUNCTIONS_INDICES)
def test_trilu_indices_default(func, n):
    print_msg = f"np & cupynumeric.{func}({n})"
    check_module_function(func, [n], {}, print_msg)


@pytest.mark.parametrize("k", KS + (-N, N), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize("m", (1, 10, N), ids=lambda m: f"(m={m})")
@pytest.mark.parametrize("n", (1, N), ids=lambda n: f"(n={n})")
@pytest.mark.parametrize("func", FUNCTIONS_INDICES)
def test_trilu_indices_full(func, n, m, k):
    print_msg = f"np & cupynumeric.{func}({n}, k={k}, m={m})"
    check_module_function(func, [n], {"k": k, "m": m}, print_msg)


@pytest.mark.parametrize("m", (0, None), ids=lambda m: f"(m={m})")
@pytest.mark.parametrize("func", FUNCTIONS_INDICES)
def test_trilu_indices_m(func, m):
    print_msg = f"np & cupynumeric.{func}({N}, m={m})"
    check_module_function(func, [N], {"m": m}, print_msg)


@pytest.mark.xfail
@pytest.mark.parametrize("k", (-10.5, 0.0, 10.5), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize("func", FUNCTIONS_INDICES)
def test_trilu_indices_float_k(func, k):
    # cuPyNumeric: struct.error: required argument is not an integer
    # Numpy: pass
    print_msg = f"np & cupynumeric.{func}({N}, k={k})"
    check_module_function(func, [N], {"k": k}, print_msg)


class TestTriluIndicesErrors:
    def test_negative_n(self):
        with pytest.raises(ValueError):
            num.tril_indices(-100)

    @pytest.mark.xfail
    def test_negative_n_DIVERGENCE(self):
        # np.tril_indices(-100) returns empty array, dtype=int64
        # num.tril_indices(-100) raises ValueError
        n = -100
        np_res = np.tril_indices(n)
        num_res = num.tril_indices(n)
        assert np.array_equal(np_res, num_res)

    @pytest.mark.parametrize("n", (-10.5, 0.0, 10.5))
    def test_float_n(self, n):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.tril_indices(n)

    @pytest.mark.xfail
    @pytest.mark.parametrize("n", (-10.5, 0.0, 10.5))
    def test_float_n_DIVERGENCE(self, n):
        # np.tril_indices(-10.5) returns empty array, dtype=int64
        # np.tril_indices(0.0) returns empty array, dtype=int64
        # np.tril_indices(10.5) returns array, dtype=int64
        # num.tril_indices(-10.5) raises TypeError
        # num.tril_indices(0.0) raises TypeError
        # num.tril_indices(10.5) raises TypeError
        np_res = np.tril_indices(n)
        num_res = num.tril_indices(n)
        assert np.array_equal(np_res, num_res)

    def test_negative_m(self):
        with pytest.raises(ValueError):
            num.tril_indices(N, m=-10)

    @pytest.mark.xfail
    def test_negative_m_DIVERGENCE(self):
        # np.tril_indices(100, m=-10) returns empty array, dtype=int64
        # num.tril_indices(100, m=-10) raises ValueError
        m = -10
        np_res = np.tril_indices(N, m=m)
        num_res = num.tril_indices(N, m=m)
        assert np.array_equal(np_res, num_res)

    @pytest.mark.parametrize("m", (-10.5, 0.0, 10.5))
    def test_float_m(self, m):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.tril_indices(N, m=m)

    @pytest.mark.xfail
    @pytest.mark.parametrize("m", (-10.5, 0.0, 10.5))
    def test_float_m_DIVERGENCE(self, m):
        # np.tril_indices(100, m=-10.5) returns empty array, dtype=int64
        # np.tril_indices(100, m=0.0) returns empty array, dtype=int64
        # np.tril_indices(100, m=10.5) returns array, dtype=int64
        # num.tril_indices(100, m=-10.5) raises TypeError
        # num.tril_indices(100, m=0.0) raises TypeError
        # num.tril_indices(100, m=10.5) raises TypeError
        np_res = np.tril_indices(N, m=m)
        num_res = num.tril_indices(N, m=m)
        assert np.array_equal(np_res, num_res)

    def test_n_none(self):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.tril_indices(None)

    @pytest.mark.xfail
    def test_k_none(self):
        # In cuPyNumeric, it raises struct.error,
        # msg is required argument is not an integer
        # In Numpy, it raises TypeError,
        # msg is bad operand type for unary -: 'NoneType'
        with pytest.raises(TypeError):
            num.tril_indices(N, k=None)


ARRAY_SHAPE = ((1, 1), (1, N), (10, N), (N, N), (N, 10), (N, 1))


@pytest.mark.parametrize("k", KS + (-N, N), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize(
    "shape", ARRAY_SHAPE, ids=lambda shape: f"(shape={shape})"
)
@pytest.mark.parametrize("func", FUNCTIONS_INDICES_FROM)
def test_trilu_indices_from(func, shape, k):
    _test_from(func, shape, k)


@pytest.mark.parametrize(
    "shape", ((10, 0), (0, 10), (0, 0)), ids=lambda shape: f"(shape={shape})"
)
@pytest.mark.parametrize("func", FUNCTIONS_INDICES_FROM)
def test_trilu_indices_from_empty_array(func, shape):
    k = 0
    _test_from(func, shape, k)


@pytest.mark.xfail
@pytest.mark.parametrize("k", (-10.5, 0.0, 10.5), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize("func", FUNCTIONS_INDICES_FROM)
def test_trilu_indices_from_float_k(func, k):
    # cuPyNumeric: struct.error: required argument is not an integer
    # Numpy: pass
    shape = (10, 10)
    _test_from(func, shape, k)


class TestTriluIndicesFromErrors:
    @pytest.mark.parametrize("size", ((5,), (0,)), ids=str)
    @pytest.mark.parametrize(
        "dimension", (1, 3), ids=lambda dimension: f"(dim={dimension})"
    )
    @pytest.mark.parametrize("func", FUNCTIONS_INDICES_FROM)
    def test_arr_non_2d(self, func, dimension, size):
        expected_exc = ValueError
        shape = size * dimension
        a = num.ones(shape, dtype=int)
        with pytest.raises(expected_exc):
            getattr(np, func)(a)
        with pytest.raises(expected_exc):
            getattr(num, func)(a)

    @pytest.mark.parametrize("func", FUNCTIONS_INDICES_FROM)
    def test_arr_0d(self, func):
        expected_exc = ValueError
        a = np.array(3)
        with pytest.raises(expected_exc):
            getattr(np, func)(a)
        with pytest.raises(expected_exc):
            getattr(num, func)(a)

    @pytest.mark.parametrize("func", FUNCTIONS_INDICES_FROM)
    def test_arr_none(self, func):
        expected_exc = AttributeError
        with pytest.raises(expected_exc):
            getattr(np, func)(None)
        with pytest.raises(expected_exc):
            getattr(num, func)(None)

    @pytest.mark.xfail
    def test_k_none(self):
        # In cuPyNumeric, it raises struct.error,
        # msg is required argument is not an integer
        # In Numpy, it raises TypeError,
        # msg is bad operand type for unary -: 'NoneType'
        a = num.ones((3, 3))
        with pytest.raises(TypeError):
            num.tril_indices_from(a, k=None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
