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
from utils.comparisons import allclose

import cupynumeric as num
from cupynumeric.runtime import runtime

SIZES = (5, 8, 127)

SIZES_BATCHED = [
    (1, 2, 2),
    (27, 5, 5),
    (3, 2, 127, 127),
    (3, 2, 5, 5),
    (1, 2, 5, 5),
    (4, 1, 5, 5),
    (2, 1, 0, 0),
    (4, 4, 127, 127),
]

RTOL = {
    np.dtype(np.float32): 1e-4,
    np.dtype(np.complex64): 1e-4,
    np.dtype(np.float64): 1e-8,
    np.dtype(np.complex128): 1e-8,
}

ATOL = {
    np.dtype(np.float32): 1e-5,
    np.dtype(np.complex64): 1e-5,
    np.dtype(np.float64): 1e-10,
    np.dtype(np.complex128): 1e-10,
}

MULTI_GPU = runtime.num_gpus > 1


def make_positive_definite(a):
    """Create a positive definite matrix from a square matrix."""
    # A.T @ A + n*I is positive definite
    n = a.shape[-1]
    result = a @ a.T.conj() + np.eye(n, dtype=a.dtype) * n * 2
    return result


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
def test_cho_solve_1d(
    n: int, a_dtype: np.dtype, b_dtype: np.dtype, lower: bool
):
    """Test cho_solve with 1D RHS."""
    # Create positive definite matrix
    a_base = np.random.rand(n, n).astype(a_dtype)
    if np.issubdtype(a_dtype, np.complexfloating):
        a_base = a_base + 1j * np.random.rand(n, n).astype(a_dtype)
    a = make_positive_definite(a_base)
    b = np.random.rand(n).astype(b_dtype)
    if np.issubdtype(b_dtype, np.complexfloating):
        b = b + 1j * np.random.rand(n).astype(b_dtype)

    # Compute Cholesky factorization using cho_factor
    c_and_lower = num.linalg.cho_factor(a, lower=lower)

    # Solve using cho_solve
    x = num.linalg.cho_solve(c_and_lower, b)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]
    if n > 100:
        atol *= 10.0

    # Verify: A @ x = b
    assert allclose(
        b, num.matmul(a, x), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
def test_cho_solve_2d(
    n: int, a_dtype: np.dtype, b_dtype: np.dtype, lower: bool
):
    """Test cho_solve with 2D RHS."""
    nrhs = n // 2 + 1
    # Create positive definite matrix
    a_base = np.random.rand(n, n).astype(a_dtype)
    if np.issubdtype(a_dtype, np.complexfloating):
        a_base = a_base + 1j * np.random.rand(n, n).astype(a_dtype)
    a = make_positive_definite(a_base)
    b = np.random.rand(n, nrhs).astype(b_dtype)
    if np.issubdtype(b_dtype, np.complexfloating):
        b = b + 1j * np.random.rand(n, nrhs).astype(b_dtype)

    # Compute Cholesky factorization using cho_factor
    c_and_lower = num.linalg.cho_factor(a, lower=lower)

    # Solve using cho_solve
    x = num.linalg.cho_solve(c_and_lower, b)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]
    if n > 100:
        atol *= 10.0

    # Verify: A @ x = b
    assert allclose(
        b, num.matmul(a, x), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize("size", SIZES_BATCHED)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
def test_cho_solve_batched(
    size: tuple, a_dtype: np.dtype, b_dtype: np.dtype, lower: bool
):
    """Test cho_solve with batched matrices."""
    nrhs = size[0] + 1
    size_b = tuple(size[:-1]) + (nrhs,)

    # Create batched positive definite matrices
    a = np.random.rand(np.prod(size)).reshape(size).astype(a_dtype)
    if np.issubdtype(a_dtype, np.complexfloating):
        a = a + 1j * np.random.rand(np.prod(size)).reshape(size).astype(
            a_dtype
        )

    n = size[-1]
    if n > 0:
        a_flat = a.reshape(-1, n, n)
        for i in range(a_flat.shape[0]):
            a_flat[i] = make_positive_definite(a_flat[i])
        a = a_flat.reshape(size)

    b = np.random.rand(np.prod(size_b)).reshape(size_b).astype(b_dtype)
    if np.issubdtype(b_dtype, np.complexfloating):
        b = b + 1j * np.random.rand(np.prod(size_b)).reshape(size_b).astype(
            b_dtype
        )

    # Compute Cholesky factorization using cho_factor
    c_and_lower = num.linalg.cho_factor(a, lower=lower)

    # Solve using cho_solve
    x = num.linalg.cho_solve(c_and_lower, b)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]

    # Verify: A @ x = b for each batch
    assert allclose(b, a @ x, rtol=rtol, atol=atol, check_dtype=False)


def test_cho_solve_corner_cases():
    """Test cho_solve with 1x1 matrices."""
    a = num.array([[5.0]])
    c_and_lower = num.linalg.cho_factor(a, lower=True)
    b = num.array([10.0])

    x = num.linalg.cho_solve(c_and_lower, b)
    assert allclose(x, num.array([2.0]))

    b = num.array([[10.0]])
    x = num.linalg.cho_solve(c_and_lower, b)
    assert allclose(x, num.array([[2.0]]))


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_cho_solve_dtype_int(dtype: np.dtype):
    """Test cho_solve with integer input."""
    a = [[4, 2], [2, 5]]
    a_num = num.array(a).astype(np.float64)
    # Make it positive definite
    a_num = num.matmul(a_num, a_num.T) + num.eye(2, dtype=dtype) * 4
    a_num = a_num.astype(dtype)

    b = [8, 10]
    b_num = num.array(b).astype(dtype)

    c_and_lower = num.linalg.cho_factor(a_num, lower=True)
    x = num.linalg.cho_solve(c_and_lower, b_num)

    rtol = RTOL[x.dtype]
    atol = ATOL[x.dtype]
    assert allclose(
        b_num, num.matmul(a_num, x), rtol=rtol, atol=atol, check_dtype=False
    )


def test_cho_solve_overwrite_b():
    """Test cho_solve with overwrite_b parameter."""
    n = 8
    a_base = np.random.rand(n, n).astype(np.float64)
    a = make_positive_definite(a_base)
    b = np.random.rand(n).astype(np.float64)
    b_copy = b.copy()

    c_and_lower = num.linalg.cho_factor(a, lower=True)

    # With overwrite_b=False (default), b should not be modified
    x = num.linalg.cho_solve(c_and_lower, b, overwrite_b=False)
    assert allclose(b, b_copy)

    # With overwrite_b=True, result should still be correct
    x2 = num.linalg.cho_solve(c_and_lower, b, overwrite_b=True)
    assert allclose(x, x2)


def test_cho_solve_check_finite():
    """Test cho_solve with check_finite parameter."""
    n = 5
    a_base = np.random.rand(n, n).astype(np.float64)
    a = make_positive_definite(a_base)
    b = np.random.rand(n).astype(np.float64)

    c_and_lower = num.linalg.cho_factor(a, lower=True)

    # Normal case should work
    x = num.linalg.cho_solve(c_and_lower, b, check_finite=True)
    assert allclose(a @ x, b)

    # With NaN in c
    c, lower = c_and_lower
    c_nan = c.copy()
    c_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        num.linalg.cho_solve((c_nan, lower), b, check_finite=True)

    # With inf in b
    b_inf = b.copy()
    b_inf[0] = np.inf
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        num.linalg.cho_solve(c_and_lower, b_inf, check_finite=True)

    # With check_finite=False, should not raise (though result will be invalid)
    num.linalg.cho_solve((c_nan, lower), b, check_finite=False)
    num.linalg.cho_solve(c_and_lower, b_inf, check_finite=False)


class TestChoSolveErrors:
    """Test error cases for cho_solve."""

    def setup_method(self):
        self.n = 5
        a_base = num.random.rand(self.n, self.n).astype(np.float64)
        a = a_base @ a_base.T + num.eye(self.n) * self.n * 2
        self.c, self.lower = num.linalg.cho_factor(a, lower=True)
        self.b = num.random.rand(self.n).astype(np.float64)

    def test_c_bad_dim(self):
        """Test with non-2D matrix c."""
        c = num.random.rand(self.n).astype(np.float64)
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.cho_solve((c, self.lower), self.b)

    def test_b_bad_dim(self):
        """Test with scalar b."""
        b = 10
        msg = "Array must be at least one-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.cho_solve((self.c, self.lower), b)

    def test_c_bad_dtype_float16(self):
        """Test with unsupported float16 dtype for c."""
        c = self.c.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.cho_solve((c, self.lower), self.b)

    def test_b_bad_dtype_float16(self):
        """Test with unsupported float16 dtype for b."""
        b = self.b.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.cho_solve((self.c, self.lower), b)

    def test_c_last_2_dims_not_square(self):
        """Test with non-square matrix c."""
        c = num.random.rand(self.n, self.n + 1).astype(np.float64)
        msg = "Last 2 dimensions of the array must be square"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.cho_solve((c, self.lower), self.b)

    def test_c_b_mismatched_shape(self):
        """Test with mismatched shapes between c and b."""
        b = num.random.rand(self.n + 1).astype(np.float64)
        with pytest.raises(ValueError):
            num.linalg.cho_solve((self.c, self.lower), b)

        b = num.random.rand(self.n + 1, self.n).astype(np.float64)
        with pytest.raises(ValueError):
            num.linalg.cho_solve((self.c, self.lower), b)

    def test_c_dim_3_b_not_dim_3(self):
        """Test with batched c but non-batched b."""
        c = num.random.rand(10, self.n, self.n).astype(np.float64)
        b = num.random.rand(10, self.n).astype(np.float64)
        msg = (
            "Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"
        )
        import re

        with pytest.raises(ValueError, match=re.escape(msg)):
            num.linalg.cho_solve((c, self.lower), b)

    def test_c_dim_3_dimension_mismatch(self):
        """Test with mismatched batch dimensions."""
        n = 5
        c = num.random.rand(37, n, n).astype(np.float64)
        b = num.random.rand(37, n + 1, 12).astype(np.float64)
        msg = (
            "Input operand 1 has a mismatch in its dimension "
            f"{b.ndim - 2}, with signature (...,m,m),(...,m,n)->(...,m,n)"
            f" (size {n + 1} is different from {n})"
        )
        import re

        with pytest.raises(ValueError, match=re.escape(msg)):
            num.linalg.cho_solve((c, self.lower), b)


def test_cho_solve_empty():
    """Test cho_solve with empty arrays."""
    a = num.empty((0, 0))
    c_and_lower = num.linalg.cho_factor(a, lower=True)
    b = num.empty((0,))
    x = num.linalg.cho_solve(c_and_lower, b)
    assert x.shape == (0,)

    b = num.empty((0, 5))
    x = num.linalg.cho_solve(c_and_lower, b)
    assert x.shape == (0, 5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
