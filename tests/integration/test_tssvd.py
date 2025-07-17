# Copyright 2025 NVIDIA Corporation
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


MN_SIZES = ((25, 5), )

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


# make random real full column rank mxn, m>n matrix:
#
def make_random_matrix(
        m: int, n: int, scale: float = 10.0,
        dtype_=np.dtype("float64") ) -> np.ndarray:
    np.random.seed(6174)
    
    mat = scale * np.random.rand(m, n)

    mat = mat.astype(dtype_)

    # strictly diagonally dominant:
    #
    for i in range(n):
        mat[i, i] = 1.0 + np.sum(np.abs(mat[i,:]))

    return mat


def check_decreasing(a: np.ndarray) -> bool:
    return bool(np.all([a[j] > a[j+1] for j in range(0, a.size-1)]))


@pytest.mark.parametrize("shapes", MN_SIZES)
@pytest.mark.parametrize("scale", (10.0,))
def test_tssvd_ordered(shapes: (int, int), scale: float):
    m = shapes[0]
    n = shapes[1]

    a = make_random_matrix(m, n, scale, np.dtype("float64"))

    u, svals, vt = num.linalg.tssvd(a)

    assert check_decreasing(svals) == True


@pytest.mark.parametrize("shapes", MN_SIZES)
@pytest.mark.parametrize("scale", (10.0,))
def test_tssvd_vs_svd(shapes: (int, int), scale: float):
    m = shapes[0]
    n = shapes[1]

    a = make_random_matrix(m, n, scale, np.dtype("float64"))

    u, svals, vt = num.linalg.tssvd(a)

    S = num.diag(svals)

    up, sp, vtp = num.linalg.svd(a, full_matrices=False)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]

    assert allclose(
        sp, svals, rtol=rtol, atol=atol, check_dtype=False
    )

    a1 = num.matmul(u, S)
    ap = num.matmul(a1, vt)
    
    assert allclose(
        ap, a, rtol=rtol, atol=atol, check_dtype=False
    )

    print("svals:\n%s\n"%(str(svals)))
    print("U:\n%s\n%s\n"%(str(u), str(up)))
    print("V.T:\n%s\n%s\n"%(str(vt), str(vtp)))

    # for _distinct_ singular values, the orthonormal
    # matrices U,V should be "equal", in absolute value;
    # otherwise columns may be permuted, so comparison
    # would fail;
    #
    if np.unique(svals).size == svals.size:
        # compare absolute values, as changing the sign of a
        # column in an orthonormal matrix results in an
        # orthonormal matrix; hence, column signs may differ,
        # but one could still have a correct decomposition:
        #
        assert allclose(
            np.abs(up), np.abs(u), rtol=rtol, atol=atol, check_dtype=False
        )

        assert allclose(
            np.abs(vtp), np.abs(vt), rtol=rtol, atol=atol, check_dtype=False
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
