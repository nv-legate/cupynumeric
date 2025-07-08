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

# import cupy as cp
# import cupyx.scipy.linalg as cpxl
import pytest
import scipy as sp
from utils.comparisons import allclose

import cupynumeric as num

SIZES = (4, 10, 50)

RTOL = {
    np.dtype(np.float32): 1e-1,
    np.dtype(np.complex64): 1e-1,
    np.dtype(np.float64): 1e-5,
    np.dtype(np.complex128): 1e-5,
}

ATOL = {
    np.dtype(np.float32): 1e-3,
    np.dtype(np.complex64): 1e-3,
    np.dtype(np.float64): 1e-6,
    np.dtype(np.complex128): 1e-6,
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
@pytest.mark.parametrize("min_v", (0.0,))  # 10.0)
@pytest.mark.parametrize("max_v", (2.0,))  # 100.0)
def test_expm_rnd_skew_h(n, min_v, max_v):
    a = make_skew_hermitian(n, min_v, max_v)
    check_skew_hermitian(a)

    # more info for debug purposes:
    # (out_num, m, s) = num.linalg.expm_impl(a)
    #
    out_num = num.linalg.expm(a)
    out_s = sp.linalg.expm(a)

    # cupy experiments:
    # (keep this code for possible future use)
    #
    # a_cp = cp.asarray(a)
    # out_cp = cpxl.expm(a_cp)
    # out_s = cp.asnumpy(out_cp)

    rtol = RTOL[out_num.dtype]
    atol = ATOL[out_num.dtype]
    if n > 1024:
        atol *= 20.0

    print("\nexternal solver: %s\n" % (str(out_s)))
    print("CuPyNumeric: %s\n" % (str(out_num)))

    tol_satisfied = allclose(
        out_num, out_s, rtol=rtol, atol=atol, check_dtype=False
    )

    if tol_satisfied == False:
        # check diff in ||exp(A)||_2:
        #
        norm_exp_s = np.linalg.norm(out_s, ord=2)
        norm_exp_num = np.linalg.norm(out_num, ord=2)
        #
        # conversion to string shows more decimals...
        #
        print("external ||exp(A)|| = %s\n" % (str(norm_exp_s)))
        print("Cupynumeric ||exp(A)|| = %s\n" % (str(norm_exp_num)))
        assert np.abs(1.0 - norm_exp_num) <= np.abs(1.0 - norm_exp_s)

        (_, R) = np.linalg.qr(a)
        min_abs_diag = np.min([np.abs(R[k, k]) for k in range(a.shape[0])])
        if min_abs_diag.item() < atol:
            print("source matrix close to singular!")
            assert False

        return

    assert True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
