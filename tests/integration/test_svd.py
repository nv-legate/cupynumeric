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

import re
import numpy as np
import pytest
from utils.comparisons import allclose

import cupynumeric as num

SIZES = (8, 9, 255)

RTOL = {
    np.dtype(np.int32): 1e-1,
    np.dtype(np.int64): 1e-1,
    np.dtype(np.float32): 1e-1,
    np.dtype(np.complex64): 1e-1,
    np.dtype(np.float64): 1e-5,
    np.dtype(np.complex128): 1e-5,
}

ATOL = {
    np.dtype(np.int32): 1e-3,
    np.dtype(np.int64): 1e-3,
    np.dtype(np.float32): 1e-3,
    np.dtype(np.complex64): 1e-3,
    np.dtype(np.float64): 1e-8,
    np.dtype(np.complex128): 1e-8,
}


def assert_result(a, u, s, vh):
    # (u * s) @ vh
    m = a.shape[0]
    n = a.shape[1]
    k = min(m, n)

    if k < m:
        u = u[:, :k]

    a2 = num.matmul(u * s, vh)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]
    assert allclose(a, a2, rtol=rtol, atol=atol, check_dtype=False)


@pytest.mark.parametrize("m", SIZES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("full_matrices", (False, True))
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_svd(m, n, full_matrices, a_dtype):
    if m < n:
        pytest.skip()

    if np.issubdtype(a_dtype, np.complexfloating):
        a = np.random.rand(m, n) + np.random.rand(m, n) * 1j
    else:
        a = np.random.rand(m, n)

    a = a.astype(a_dtype)

    u, s, vh = num.linalg.svd(a, full_matrices)

    assert_result(a, u, s, vh)


def test_svd_corner_cases():
    a = num.random.rand(1, 1)

    u, s, vh = num.linalg.svd(a)

    assert_result(a, u, s, vh)


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_svd_dtype_int(dtype):
    a_array = [[1, 4, 5], [2, 3, 1], [9, 5, 2]]
    a = num.array(a_array).astype(dtype)

    u, s, vh = num.linalg.svd(a)

    assert_result(a, u, s, vh)


class TestSvdErrors:
    def setup_method(self):
        self.n = 3
        self.a = num.random.rand(self.n, self.n).astype(np.float64)
        self.b = num.random.rand(self.n).astype(np.float64)

    def test_a_bad_dim(self):
        a = num.random.rand(self.n).astype(np.float64)
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.svd(a)

        a = 10
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.svd(a)

    def test_a_dim_greater_than_two(self):
        a = num.random.rand(self.n, self.n, self.n).astype(np.float64)
        with pytest.raises(NotImplementedError):
            num.linalg.svd(a)

    def test_a_bad_dtype_float16(self):
        a = self.a.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.svd(a)
    
    def test_linalgerror_1d(self) -> None:
        arr_np = np.array([1, 2, 3])
        arr_num = num.array(arr_np)
        msg = (
            r"1-dimensional array given. "
            "Array must be at least two-dimensional"
        )
        with pytest.raises(np.linalg.LinAlgError, match=re.escape(msg)):
            np.linalg.svd(arr_np)
        with pytest.raises(num.linalg.LinAlgError, match=re.escape(msg)):
            num.linalg.svd(arr_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
