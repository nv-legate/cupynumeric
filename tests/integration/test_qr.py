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


@pytest.mark.parametrize("m", SIZES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_qr(m, n, a_dtype):
    a = np.random.rand(m, n).astype(a_dtype)

    q, r = num.linalg.qr(a)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]
    assert allclose(
        a, num.matmul(q, r), rtol=rtol, atol=atol, check_dtype=False
    )


def test_qr_corner_cases():
    a = num.random.rand(1, 1)

    q, r = num.linalg.qr(a)
    assert allclose(a, num.matmul(q, r))


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_qr_dtype_int(dtype):
    a_array = [[1, 4, 5], [2, 3, 1], [9, 5, 2]]
    a = num.array(a_array).astype(dtype)

    q, r = num.linalg.qr(a)

    rtol = RTOL[q.dtype]
    atol = ATOL[q.dtype]
    assert allclose(
        a, num.matmul(q, r), rtol=rtol, atol=atol, check_dtype=False
    )


class TestQrErrors:
    def setup_method(self):
        self.n = 3
        self.a = num.random.rand(self.n, self.n).astype(np.float64)
        self.b = num.random.rand(self.n).astype(np.float64)

    def test_a_bad_dim(self):
        a = num.random.rand(self.n).astype(np.float64)
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.qr(a)

        a = 10
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.qr(a)

    def test_a_dim_greater_than_two(self):
        a = num.random.rand(self.n, self.n, self.n).astype(np.float64)
        with pytest.raises(NotImplementedError):
            num.linalg.qr(a)

    def test_a_bad_dtype_float16(self):
        a = self.a.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.qr(a)
            
    def test_qr_linalgerror_1d(self) -> None:
        arr_np = np.array([1, 2, 3])
        arr_num = num.array(arr_np)
        msg = (
            r"1-dimensional array given. "
            "Array must be at least two-dimensional"
        )
        with pytest.raises(np.linalg.LinAlgError, match=re.escape(msg)):
            np.linalg.qr(arr_np)
        with pytest.raises(num.linalg.LinAlgError, match=re.escape(msg)):
            num.linalg.qr(arr_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
