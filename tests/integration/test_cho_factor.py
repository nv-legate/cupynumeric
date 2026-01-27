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

SIZES = (5, 8, 127)

SIZES_BATCHED = [
    (1, 2, 2),
    (27, 5, 5),
    (3, 2, 5, 5),
    (1, 2, 5, 5),
    (4, 1, 5, 5),
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


def make_positive_definite(a):
    """Create a positive definite matrix from a square matrix."""
    # A.T @ A + n*I is positive definite
    n = a.shape[-1]
    result = a @ a.T.conj() + np.eye(n, dtype=a.dtype) * n * 2
    return result


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
def test_cho_factor_basic(n: int, dtype: np.dtype, lower: bool):
    """Test cho_factor returns correct tuple and can be used with cho_solve."""
    # Create positive definite matrix
    a_base = np.random.rand(n, n).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a_base = a_base + 1j * np.random.rand(n, n).astype(dtype)
    a = make_positive_definite(a_base)

    # Compute Cholesky factorization using cho_factor
    c, lower_flag = num.linalg.cho_factor(a, lower=lower)

    # Verify the return type is a tuple
    assert isinstance(c, num.ndarray)
    assert isinstance(lower_flag, bool)
    assert lower_flag == lower

    # Verify the shape
    assert c.shape == a.shape

    # Test that it works with cho_solve
    b = np.random.rand(n).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        b = b + 1j * np.random.rand(n).astype(dtype)

    x = num.linalg.cho_solve((c, lower_flag), b)

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
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_cho_factor_lower_true(n: int, dtype: np.dtype):
    """Test cho_factor with lower=True returns lower triangular factor."""
    a_base = np.random.rand(n, n).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a_base = a_base + 1j * np.random.rand(n, n).astype(dtype)
    a = make_positive_definite(a_base)

    c, lower_flag = num.linalg.cho_factor(a, lower=True)
    assert lower_flag is True

    # Verify A = L @ L.H
    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]
    if n > 100:
        atol *= 10.0

    # Extract only the lower triangular part (upper triangle contains garbage)
    c_np = num.array(c)
    L = np.tril(c_np)
    reconstructed = L @ L.T.conj()

    assert allclose(a, reconstructed, rtol=rtol, atol=atol, check_dtype=False)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_cho_factor_lower_false(n: int, dtype: np.dtype):
    """Test cho_factor with lower=False returns upper triangular factor."""
    a_base = np.random.rand(n, n).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a_base = a_base + 1j * np.random.rand(n, n).astype(dtype)
    a = make_positive_definite(a_base)

    c, lower_flag = num.linalg.cho_factor(a, lower=False)
    assert lower_flag is False

    # Verify A = U.H @ U
    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]
    if n > 100:
        atol *= 10.0

    # Extract only the upper triangular part (lower triangle contains garbage)
    c_np = num.array(c)
    U = np.triu(c_np)
    reconstructed = U.T.conj() @ U

    assert allclose(a, reconstructed, rtol=rtol, atol=atol, check_dtype=False)


@pytest.mark.parametrize("size", SIZES_BATCHED)
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
def test_cho_factor_batched(size: tuple, dtype: np.dtype, lower: bool):
    """Test cho_factor with batched matrices."""
    # Create batched positive definite matrices
    a = np.random.rand(np.prod(size)).reshape(size).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * np.random.rand(np.prod(size)).reshape(size).astype(dtype)

    n = size[-1]
    if n > 0:
        a_flat = a.reshape(-1, n, n)
        for i in range(a_flat.shape[0]):
            a_flat[i] = make_positive_definite(a_flat[i])

        a_mod_unused_flat = a_flat.copy()
        for i in range(a_mod_unused_flat.shape[0]):
            # Invalidate part of matrix that is not supposed to be touched by the factorization
            a_mod_unused_flat[i, lower ^ 1, lower ^ 0] = 23
        a = a_flat.reshape(size)
        a_mod_unused = a_mod_unused_flat.reshape(size)

    # Compute Cholesky factorization using cho_factor
    c, lower_flag = num.linalg.cho_factor(a_mod_unused, lower=lower)

    assert isinstance(c, num.ndarray)
    assert isinstance(lower_flag, bool)
    assert lower_flag == lower
    assert c.shape == a.shape

    # Test that it works with cho_solve
    nrhs = 3
    size_b = tuple(size[:-1]) + (nrhs,)
    b = np.random.rand(np.prod(size_b)).reshape(size_b).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        b = b + 1j * np.random.rand(np.prod(size_b)).reshape(size_b).astype(
            dtype
        )

    x = num.linalg.cho_solve((c, lower_flag), b)

    rtol = RTOL[a.dtype]
    atol = ATOL[a.dtype]

    # Verify: A @ x = b for each batch
    assert allclose(b, a @ x, rtol=rtol, atol=atol, check_dtype=False)


def test_cho_factor_check_finite():
    """Test cho_factor with check_finite parameter."""
    n = 5
    a_base = np.random.rand(n, n).astype(np.float64)
    a = make_positive_definite(a_base)

    # Normal case should work
    c, lower = num.linalg.cho_factor(a, check_finite=True)
    assert c.shape == (n, n)
    assert isinstance(lower, bool)

    # With NaN in a
    a_nan = a.copy()
    a_nan[1, 0] = np.nan
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        num.linalg.cho_factor(a_nan, check_finite=True)

    # With inf in a
    a_inf = a.copy()
    a_inf[1, 0] = np.inf
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        num.linalg.cho_factor(a_inf, check_finite=True)

    # With check_finite=False, should not raise (though result will be invalid)
    num.linalg.cho_factor(a_nan, check_finite=False)
    num.linalg.cho_factor(a_inf, check_finite=False)


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_cho_factor_dtype_int(dtype: np.dtype):
    """Test cho_factor with integer input (should convert to float)."""
    a = [[4, 2], [2, 5]]
    a_num = num.array(a).astype(np.float64)
    # Make it positive definite
    a_num = num.matmul(a_num, a_num.T) + num.eye(2) * 4
    a_num = a_num.astype(dtype)

    c, lower = num.linalg.cho_factor(a_num, lower=True)

    # Verify it works with cho_solve
    b = [8, 10]
    b_num = num.array(b).astype(dtype)
    x = num.linalg.cho_solve((c, lower), b_num)

    rtol = RTOL[c.dtype]
    atol = ATOL[c.dtype]
    assert allclose(
        b_num, num.matmul(a_num, x), rtol=rtol, atol=atol, check_dtype=False
    )


def test_cho_factor_default_params():
    """Test cho_factor with default parameters (lower=False)."""
    n = 5
    a_base = np.random.rand(n, n).astype(np.float64)
    a = make_positive_definite(a_base)

    # Default should be lower=False
    c, lower = num.linalg.cho_factor(a)
    assert lower is False


def test_cho_factor_1x1():
    """Test cho_factor with 1x1 matrix."""
    a = num.array([[5.0]])
    c, lower = num.linalg.cho_factor(a, lower=True)

    assert c.shape == (1, 1)
    assert lower is True

    # Should work with cho_solve
    b = num.array([10.0])
    x = num.linalg.cho_solve((c, lower), b)
    assert allclose(x, num.array([2.0]))


class TestChoFactorErrors:
    """Test error cases for cho_factor."""

    def test_bad_dim_1d(self):
        """Test with 1D array."""
        a = num.random.rand(5).astype(np.float64)
        msg = "Array must be at least two-dimensional"
        with pytest.raises(ValueError, match=msg):
            num.linalg.cho_factor(a)

    def test_bad_dim_0d(self):
        """Test with scalar."""
        a = 5.0
        msg = "Array must be at least two-dimensional"
        with pytest.raises(ValueError, match=msg):
            num.linalg.cho_factor(a)

    def test_not_square(self):
        """Test with non-square matrix."""
        a = num.random.rand(5, 6).astype(np.float64)
        msg = "Last 2 dimensions of the array must be square"
        with pytest.raises(ValueError, match=msg):
            num.linalg.cho_factor(a)

    def test_not_positive_definite(self):
        """Test with non-positive-definite matrix."""
        # This matrix is not positive definite
        a = num.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(num.linalg.LinAlgError):
            num.linalg.cho_factor(a)


def test_cho_factor_integration_with_cho_solve():
    """Integration test: cho_factor result should work seamlessly with cho_solve."""
    n = 10
    a_base = np.random.rand(n, n).astype(np.float64)
    a = make_positive_definite(a_base)

    # Test both lower and upper
    for lower in [True, False]:
        c_and_lower = num.linalg.cho_factor(a, lower=lower)

        # Test with 1D RHS
        b_1d = np.random.rand(n).astype(np.float64)
        x_1d = num.linalg.cho_solve(c_and_lower, b_1d)
        assert allclose(a @ x_1d, b_1d)

        # Test with 2D RHS
        b_2d = np.random.rand(n, 3).astype(np.float64)
        x_2d = num.linalg.cho_solve(c_and_lower, b_2d)
        assert allclose(a @ x_2d, b_2d)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
