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
# SIZES = (2,)

SIZES_BATCHED = [
    (27, 5, 5),
    (1, 5, 5),
    (3, 2, 127, 127),
    (3, 2, 5, 5),
    (1, 2, 5, 5),
    (4, 1, 5, 5),
    (2, 1, 0, 0),
    (4, 4, 127, 127),
]

# SIZES_BATCHED = [(3, 2, 2)]


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


def make_triangular(a, lower=True):
    """Create a triangular matrix from a square matrix."""
    if lower:
        return np.tril(a)
    else:
        return np.triu(a)


def apply_trans(a, trans):
    """Apply transpose operation to matrix."""
    if trans in (0, "N", "n"):
        return a
    elif trans in (1, "T", "t"):
        return a.swapaxes(-2, -1)
    elif trans in (2, "C", "c"):
        return a.conj().swapaxes(-2, -1)
    else:
        raise ValueError(f"Invalid trans: {trans}")


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
@pytest.mark.parametrize("trans", (0, 1, 2))
def test_solve_triangular_1d(
    n: int, a_dtype: np.dtype, b_dtype: np.dtype, lower: bool, trans: int
):
    """Test solve_triangular with 1D RHS."""
    # Create matrix with strong diagonal dominance
    a = np.random.rand(n, n).astype(a_dtype) + np.eye(n, dtype=a_dtype) * n
    b = np.random.rand(n).astype(b_dtype)

    a = num.asarray(a)
    b = num.asarray(b)

    out = num.linalg.solve_triangular(a, b, trans=trans, lower=lower)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    if n > 100:
        atol *= 10.0

    # Verify: A @ x = b  (or A^T @ x = b, or A^H @ x = b)
    # Extract triangular portion for verification
    a_tri = make_triangular(a, lower)
    a_op = apply_trans(a_tri, trans)

    assert allclose(
        b, num.matmul(a_op, out), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
@pytest.mark.parametrize("trans", (0, 1, 2))
def test_solve_triangular_2d(
    n: int, a_dtype: np.dtype, b_dtype: np.dtype, lower: bool, trans: int
):
    """Test solve_triangular with 2D RHS."""
    nrhs = n // 2 + 1
    # Create matrix with strong diagonal dominance
    a = np.random.rand(n, n).astype(a_dtype) + np.eye(n, dtype=a_dtype) * n
    b = np.random.rand(n, nrhs).astype(b_dtype)

    out = num.linalg.solve_triangular(a, b, trans=trans, lower=lower)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    if n > 100:
        atol *= 10.0

    # Verify: A @ x = b  (or A^T @ x = b, or A^H @ x = b)
    # Extract triangular portion for verification
    a_tri = make_triangular(a, lower)
    a_op = apply_trans(a_tri, trans)
    assert allclose(
        b, num.matmul(a_op, out), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize("size", SIZES_BATCHED)
@pytest.mark.parametrize("nrhs", (1, 13))
@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "b_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
@pytest.mark.parametrize("trans", (0, 1, 2))
def test_solve_triangular_batched(
    size: tuple,
    nrhs: int,
    a_dtype: np.dtype,
    b_dtype: np.dtype,
    lower: bool,
    trans: int,
):
    """Test solve_triangular with batched matrices."""
    size_b = tuple(size[:-1]) + (nrhs,)

    # Create batched matrices with strong diagonal
    a = np.random.rand(np.prod(size)).reshape(size).astype(a_dtype)
    n = size[-1]
    if n > 0:
        a_flat = a.reshape(-1, n, n)
        for i in range(a_flat.shape[0]):
            a_flat[i] = a_flat[i] + np.eye(n, dtype=a_dtype) * n
        a = a_flat.reshape(size)

    b = np.random.rand(np.prod(size_b)).reshape(size_b).astype(b_dtype)

    out = num.linalg.solve_triangular(a, b, trans=trans, lower=lower)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]

    # Verify: A @ x = b for each batch
    # Extract triangular portion for verification
    a_tri = a.copy()
    if n > 0:
        a_tri_flat = a_tri.reshape(-1, n, n)
        for i in range(a_tri_flat.shape[0]):
            a_tri_flat[i] = make_triangular(a_tri_flat[i], lower)
        a_tri = a_tri_flat.reshape(size)
    a_op = apply_trans(a_tri, trans)
    assert allclose(b, a_op @ out, rtol=rtol, atol=atol, check_dtype=False)


@pytest.mark.parametrize("trans", ("N", "T", "C"))
def test_solve_triangular_trans_strings(trans: str):
    """Test solve_triangular with string trans parameter."""
    n = 5
    a = np.random.rand(n, n).astype(np.float64) + np.eye(n) * n
    b = np.random.rand(n).astype(np.float64)

    out = num.linalg.solve_triangular(a, b, trans=trans, lower=True)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]

    # Extract triangular portion for verification
    a_tri = make_triangular(a, lower=True)
    a_op = apply_trans(a_tri, trans)
    assert allclose(
        b, num.matmul(a_op, out), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize(
    "a_dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("lower", (True, False))
def test_solve_triangular_unit_diagonal(a_dtype: np.dtype, lower: bool):
    """Test solve_triangular with unit diagonal."""
    n = 8
    # Create matrix positive definite (with unit diagonal)
    a = np.random.rand(n, n).astype(a_dtype) * 1.0 / n
    b = np.random.rand(n, 3).astype(a_dtype)

    out = num.linalg.solve_triangular(a, b, lower=lower, unit_diagonal=True)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]

    # Extract triangular portion for verification
    a_tri = make_triangular(a, lower)
    np.fill_diagonal(a_tri, 1.0)
    assert allclose(
        b, num.matmul(a_tri, out), rtol=rtol, atol=atol, check_dtype=False
    )


def test_solve_triangular_corner_cases():
    """Test solve_triangular with 1x1 matrices."""
    a = num.array([[5.0]])
    b = num.array([10.0])

    out = num.linalg.solve_triangular(a, b, lower=True)
    assert allclose(out, num.array([2.0]))

    b = num.array([[10.0]])
    out = num.linalg.solve_triangular(a, b, lower=True)
    assert allclose(out, num.array([[2.0]]))


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_solve_triangular_dtype_int(dtype: np.dtype):
    """Test solve_triangular with integer input."""
    a = [[2, 0, 0], [1, 3, 0], [4, 2, 5]]
    b = [4, 6, 10]
    a_num = num.array(a).astype(dtype)
    b_num = num.array(b).astype(dtype)

    out = num.linalg.solve_triangular(a_num, b_num, lower=True)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b_num, num.matmul(a_num, out), rtol=rtol, atol=atol, check_dtype=False
    )


def test_solve_triangular_overwrite_b():
    """Test solve_triangular with overwrite_b parameter."""
    n = 8
    a = np.random.rand(n, n).astype(np.float64) + np.eye(n) * n
    b = np.random.rand(n).astype(np.float64)
    b_copy = b.copy()

    # With overwrite_b=False (default), b should not be modified
    out = num.linalg.solve_triangular(a, b, lower=True, overwrite_b=False)
    assert allclose(b, b_copy)

    # With overwrite_b=True, result should still be correct
    out2 = num.linalg.solve_triangular(a, b, lower=True, overwrite_b=True)
    assert allclose(out, out2)


class TestSolveTriangularErrors:
    """Test error cases for solve_triangular."""

    def setup_method(self):
        self.n = 5
        self.a = num.tril(num.random.rand(self.n, self.n).astype(np.float64))
        self.a += num.eye(self.n) * self.n
        self.b = num.random.rand(self.n).astype(np.float64)

    def test_a_bad_dim(self):
        """Test with non-2D matrix a."""
        a = num.random.rand(self.n).astype(np.float64)
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve_triangular(a, self.b)

    def test_b_bad_dim(self):
        """Test with scalar b."""
        b = 10
        msg = "Array must be at least one-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve_triangular(self.a, b)

    def test_a_bad_dtype_float16(self):
        """Test with unsupported float16 dtype for a."""
        a = self.a.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve_triangular(a, self.b)

    def test_b_bad_dtype_float16(self):
        """Test with unsupported float16 dtype for b."""
        b = self.b.astype(np.float16)
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve_triangular(self.a, b)

    def test_a_last_2_dims_not_square(self):
        """Test with non-square matrix a."""
        a = num.random.rand(self.n, self.n + 1).astype(np.float64)
        msg = "Last 2 dimensions of the array must be square"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve_triangular(a, self.b)

    def test_a_b_mismatched_shape(self):
        """Test with mismatched shapes between a and b."""
        b = num.random.rand(self.n + 1).astype(np.float64)
        with pytest.raises(ValueError):
            num.linalg.solve_triangular(self.a, b)

        b = num.random.rand(self.n + 1, self.n).astype(np.float64)
        with pytest.raises(ValueError):
            num.linalg.solve_triangular(self.a, b)

    def test_a_dim_3_b_not_dim_3(self):
        """Test with batched a but non-batched b."""
        a = num.random.rand(10, self.n, self.n).astype(np.float64)
        b = num.random.rand(10, self.n).astype(np.float64)
        msg = (
            "Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"
        )
        import re

        with pytest.raises(ValueError, match=re.escape(msg)):
            num.linalg.solve_triangular(a, b)

    def test_a_dim_3_dimension_mismatch(self):
        """Test with mismatched batch dimensions."""
        n = 5
        a = num.random.rand(37, n, n).astype(np.float64)
        b = num.random.rand(37, n + 1, 12).astype(np.float64)
        msg = (
            "Input operand 1 has a mismatch in its dimension 1, with signature"
            " (...,m,m),(...,m,n)->(...,m,n) (size 6 is different from 5)"
        )
        import re

        with pytest.raises(ValueError, match=re.escape(msg)):
            num.linalg.solve_triangular(a, b)

    def test_invalid_trans_value(self):
        """Test with invalid trans parameter - should default to trans=1 like scipy."""
        # Invalid trans values should default to trans=1 (transpose)
        out_invalid = num.linalg.solve_triangular(self.a, self.b, trans=5)
        out_expected = num.linalg.solve_triangular(self.a, self.b, trans=1)
        assert allclose(out_invalid, out_expected)

        out_invalid_str = num.linalg.solve_triangular(
            self.a, self.b, trans="X"
        )
        assert allclose(out_invalid_str, out_expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
