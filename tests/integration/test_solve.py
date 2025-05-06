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


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_solve_1d(n, a_dtype, b_dtype):
    a = np.random.rand(n, n).astype(a_dtype) + np.eye(n) * n
    b = np.random.rand(n).astype(b_dtype)

    out = num.linalg.solve(a, b)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    if n > 1024:
        atol *= 20.0

    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_solve_2d(n, a_dtype, b_dtype):
    a = np.random.rand(n, n).astype(a_dtype) + np.eye(n) * n
    b = np.random.rand(n, n + 2).astype(b_dtype)

    out = num.linalg.solve(a, b)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    if n > 1024:
        atol *= 20.0

    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )


def test_solve_corner_cases():
    a = num.random.rand(1, 1)
    b = num.random.rand(1)

    out = num.linalg.solve(a, b)
    assert allclose(b, num.matmul(a, out))

    b = num.random.rand(1, 1)
    out = num.linalg.solve(a, b)
    assert allclose(b, num.matmul(a, out))


def test_solve_b_is_empty():
    a = num.random.rand(1, 1)
    b = num.atleast_2d([])

    out = num.linalg.solve(a, b)
    assert np.array_equal(b, out)


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_solve_dtype_int(dtype):
    a = [[1, 4, 5], [2, 3, 1], [9, 5, 2]]
    b = [1, 2, 3]
    a_num = num.array(a).astype(dtype)
    b_num = num.array(b).astype(dtype)
    out = num.linalg.solve(a_num, b_num)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b_num, num.matmul(a_num, out), rtol=rtol, atol=atol, check_dtype=False
    )


def test_solve_with_output():
    n = 8
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    output = num.zeros((n,)).astype(np.float32)

    out = num.linalg.solve(a, b, out=output)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )
    assert allclose(
        b, num.matmul(a, output), rtol=rtol, atol=atol, check_dtype=False
    )


class TestSolveErrors:
    def setup_method(self):
        self.n = 3
        self.a = num.random.rand(self.n, self.n).astype(np.float64)
        self.b = num.random.rand(self.n).astype(np.float64)

    def test_a_bad_dim(self):
        a = num.random.rand(self.n).astype(np.float64)
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)

        a = 10
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)

    def test_b_bad_dim(self):
        b = 10
        msg = "Array must be at least one-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(self.a, b)

    def test_a_dim_greater_than_two(self):
        a = num.random.rand(self.n, self.n, self.n).astype(np.float64)
        b = num.random.rand(self.n, self.n).astype(np.float64)
        with pytest.raises(NotImplementedError):
            num.linalg.solve(a, b)

    def test_b_dim_greater_than_two(self):
        a = num.random.rand(self.n, self.n).astype(np.float64)
        b = num.random.rand(self.n, self.n, self.n).astype(np.float64)
        with pytest.raises(NotImplementedError):
            num.linalg.solve(a, b)

    def test_a_bad_dtype_float16(self):
        a = self.a.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve(a, self.b)

    def test_b_bad_dtype_float16(self):
        b = self.b.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve(self.a, b)

    def test_a_last_2_dims_not_square(self):
        a = num.random.rand(self.n, self.n + 1).astype(np.float64)
        msg = "Last 2 dimensions of the array must be square"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)

    def test_a_b_mismatched_shape(self):
        b = num.random.rand(self.n + 1).astype(np.float64)
        with pytest.raises(ValueError):
            num.linalg.solve(self.a, b)

        b = num.random.rand(self.n + 1, self.n).astype(np.float64)
        with pytest.raises(ValueError):
            num.linalg.solve(self.a, b)

    def test_output_mismatched_shape(self):
        output = num.zeros((self.n + 1,)).astype(np.float64)
        msg = "Output shape mismatch"
        with pytest.raises(ValueError, match=msg):
            num.linalg.solve(self.a, self.b, out=output)

    def test_output_mismatched_dtype(self):
        output = num.zeros((self.n,)).astype(np.float32)
        msg = "Output type mismatch"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve(self.a, self.b, out=output)

    def test_a_singular_matrix(self):
        a = num.zeros((self.n, self.n)).astype(np.float64)
        msg = "Singular matrix"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
