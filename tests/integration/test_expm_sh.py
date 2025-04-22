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
import scipy as sp
from utils.comparisons import allclose

import cupynumeric as num

SIZES = (4, 10)

RTOL = {
    np.dtype(np.float32): 1e-1,
    np.dtype(np.complex64): 1e-1,
    np.dtype(np.float64): 1e-5,
    np.dtype(np.complex128): 1e-5,
}

ATOL = {
    np.dtype(np.float32): 1e-3,
    np.dtype(np.complex64): 1e-3,
    np.dtype(np.float64): 1e-8,
    np.dtype(np.complex128): 1e-8,
}


def make_skew_hermitian(
    n: int, min_v: float = 0.0, max_v: float = 100.0
) -> np.ndarray:
    num_off_d = int(n * (n - 1) / 2)

    np.random.seed(1729)

    r_array = np.array(
        [np.random.uniform(min_v, max_v) for k in range(num_off_d)],
        dtype=np.dtype("float64"),
    )

    i_array = np.array(
        [np.random.uniform(min_v, max_v) for k in range(num_off_d)],
        dtype=np.dtype("float64"),
    )

    d_array = np.array(
        [np.random.uniform(min_v, max_v) for k in range(n)],
        dtype=np.dtype("float64"),
    )

    mat = np.zeros((n, n), dtype=np.dtype("complex64"))

    arr_index = 0
    for col in range(1, n):
        for row in range(0, col):
            mat[row, col] = r_array[arr_index] + i_array[arr_index] * 1.0j
            mat[col, row] = -np.conjugate(mat[row, col])

            arr_index = arr_index + 1

        c_1 = col - 1
        mat[c_1, c_1] = d_array[c_1] * 1.0j

    mat[n - 1][n - 1] = d_array[n - 1] * 1.0j

    return mat


def check_skew_hermitian(A: np.ndarray) -> bool:
    assert A.ndim == 2
    n = A.shape[0]
    assert n == A.shape[1]
    num_half_off_d = int(n * (n - 1) / 2)

    arr_off_d = np.array(
        [A[i, j] + np.conjugate(A[j, i]) for i in range(n) for j in range(i)],
        dtype=np.dtype("complex64"),
    )

    check_arr = np.zeros((num_half_off_d,), dtype=np.dtype("complex64"))
    assert arr_off_d.size == num_half_off_d

    assert allclose(
        arr_off_d, check_arr, atol=ATOL[A.dtype], check_dtype=False
    )

    assert np.all([np.real(A[k, k]) for k in range(n)] == np.zeros(n))
    return True


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("min_v", (0.0,))
@pytest.mark.parametrize("max_v", (10.0,))
def test_expm_rnd_sh_tensor_pade(n: int, min_v: float, max_v: float):
    m = 3
    a = np.zeros(shape=(m, n, n), dtype=np.complex64)
    for idx in np.ndindex(a.shape[:-2]):
        a[idx] = make_skew_hermitian(n, min_v, max_v)

    # more info for debug purposes:
    # (out_num, m, s) = num.linalg.expm_impl(a)
    #
    out_num = num.linalg.expm(a, method="pade")
    out_s = sp.linalg.expm(a)

    rtol = RTOL[out_num.dtype]
    atol = ATOL[out_num.dtype]
    if n > 1024:
        atol *= 20.0

    tol_satisfied = allclose(
        out_num, out_s, rtol=rtol, atol=atol, check_dtype=False
    )

    # scipy result may not be reliable,
    # hence check which exp L2 norm is
    # closer to unity:
    #
    if tol_satisfied is False:
        for i in range(m):
            # check diff in ||exp(A)||_2:
            #
            norm_exp_s = np.linalg.norm(out_s[i], ord=2)
            norm_exp_num = np.linalg.norm(out_num[i], ord=2)
            #
            # conversion to string shows more decimals...
            #
            print("external ||exp(A)|| = %s\n" % (str(norm_exp_s)))
            print("Cupynumeric ||exp(A)|| = %s\n" % (str(norm_exp_num)))

            assert np.abs(1.0 - norm_exp_num) <= np.abs(1.0 - norm_exp_s)

    assert True


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("min_v", (0.0,))
@pytest.mark.parametrize("max_v", (10.0,))
def test_expm_rnd_sh_tensor_taylor(n: int, min_v: float, max_v: float):
    m = 3
    a = np.zeros(shape=(m, n, n), dtype=np.complex64)
    for idx in np.ndindex(a.shape[:-2]):
        a[idx] = make_skew_hermitian(n, min_v, max_v)

    # more info for debug purposes:
    # (out_num, m, s) = num.linalg.expm_impl(a)
    #
    out_num = num.linalg.expm(a, method="taylor")
    out_s = sp.linalg.expm(a)

    rtol = RTOL[out_num.dtype]
    atol = ATOL[out_num.dtype]
    if n > 1024:
        atol *= 20.0

    tol_satisfied = allclose(
        out_num, out_s, rtol=rtol, atol=atol, check_dtype=False
    )

    # scipy result may not be reliable,
    # hence check which exp L2 norm is
    # closer to unity:
    #
    if tol_satisfied is False:
        for i in range(m):
            # check diff in ||exp(A)||_2:
            #
            norm_exp_s = np.linalg.norm(out_s[i], ord=2)
            norm_exp_num = np.linalg.norm(out_num[i], ord=2)
            #
            # conversion to string shows more decimals...
            #
            print("external ||exp(A)|| = %s\n" % (str(norm_exp_s)))
            print("Cupynumeric ||exp(A)|| = %s\n" % (str(norm_exp_num)))

            assert np.abs(1.0 - norm_exp_num) <= np.abs(1.0 - norm_exp_s)

    assert True


@pytest.mark.parametrize("nsize", ((1,), (2, 3), (), (1, 1)))
def test_invalid_array(nsize: tuple[int, ...]) -> None:
    array_np = np.random.randint(0, 10, size=nsize)
    array_num = num.array(array_np)
    with pytest.raises(ValueError, match="Invalid input shape for expm"):
        num.linalg.expm(array_num)


@pytest.mark.parametrize(
    "method",
    ["invalid", "exp", "PADE", "TAYLOR", "implicit", "", "none", "auto"],
)
def test_expm_invalid_methods(method: str) -> None:
    arr = np.array([[1, 2], [2, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match=f"Method {method} not supported"):
        num.linalg.expm(arr, method=method)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
