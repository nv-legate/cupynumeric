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
from typing import Any
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


DTYPES = (int, float, bool, None)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_tri_dtype(dtype):
    print_msg = f"np & cupynumeric.tri({N}, dtype={dtype})"
    check_module_function("tri", [N], {"dtype": dtype}, print_msg)


@pytest.mark.parametrize("k", (-10.5, 0.0, 10.5), ids=lambda k: f"(k={k})")
def test_tri_float_k(k: float) -> None:
    print_msg = f"np & cupynumeric.tri({N}, k={k})"
    check_module_function("tri", [N], {"k": k}, print_msg)


@pytest.mark.parametrize("n", (-100, -10.5, 0.0, 10.5))
def test_float_n_DIVERGENCE(n: int | float) -> None:
    np_res = np.tri(n)
    num_res = num.tri(n)
    assert np.array_equal(np_res, num_res)


@pytest.mark.parametrize("m", (-100, -10.5, 0.0, 10.5))
def test_m_DIVERGENCE(m: int | float) -> None:
    np_res = np.tri(N, M=m)
    num_res = num.tri(N, M=m)
    assert np.array_equal(np_res, num_res)


class TestTriErrors:
    def test_n_none(self) -> None:
        msg = "N parameter must be an integer."
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

    @pytest.mark.parametrize(
        "like_value", [np.array([1, 2, 3]), "not_none", 123, [], {}, True]
    )
    def test_like_parameter_not_supported(self, like_value: Any) -> None:
        with pytest.raises(
            ValueError, match="like parameter is currently not supported"
        ):
            num.tri(N, like=like_value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
