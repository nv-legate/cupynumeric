# Copyright 2026 NVIDIA Corporation
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

from cupynumeric.linalg import LinAlgError

import cupynumeric as num

RTOL = {
    np.float32: 1e-1,
    np.complex64: 1e-1,
    np.float64: 1e-5,
    np.complex128: 1e-5,
}

ATOL = {
    np.float32: 1e-3,
    np.complex64: 1e-3,
    np.float64: 1e-8,
    np.complex128: 1e-8,
}


def create_random_invertible_matrix(size, dtype, batch_shape=None):
    """Constructs a random invertible matrix of shape (size, size)
    or (*batch_shape, size, size) if batch_shape is not None."""
    shape = (size, size) if batch_shape is None else batch_shape + (size, size)
    rand_matrix = np.random.randn(*shape).astype(dtype)
    diag_vals = np.sum(np.abs(rand_matrix), axis=-1)

    if batch_shape is None:
        np.fill_diagonal(rand_matrix, diag_vals)
    else:
        total_batch_size = np.prod(batch_shape)
        matrices = [
            create_random_invertible_matrix(size, dtype)
            for _ in range(total_batch_size)
        ]
        rand_matrix = np.stack(matrices).reshape(batch_shape + (size, size))
    return rand_matrix


@pytest.mark.parametrize("size", (10, 50))
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_inv_basic(size, dtype):
    """Test basic inv functionality"""
    a = create_random_invertible_matrix(size, dtype)
    a_inv_num = num.linalg.inv(a)
    a_inv_np = np.linalg.inv(a)

    assert allclose(
        a_inv_num,
        a_inv_np,
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
        check_dtype=False,
    )


@pytest.mark.parametrize("batch_shape", ((2,), (5, 1), (4, 2)))
@pytest.mark.parametrize("size", (5, 10))
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.xfail(
    reason="Solver implementation for matrices on batch size > 2 returns incorrect results"
)
def test_inv_batch(batch_shape, size, dtype):
    """Test inv functionality with batching"""
    a = create_random_invertible_matrix(size, dtype, batch_shape)
    a_inv_num = num.linalg.inv(a)
    a_inv_np = np.linalg.inv(a)

    assert allclose(
        a_inv_num,
        a_inv_np,
        rtol=RTOL[dtype],
        atol=ATOL[dtype],
        check_dtype=False,
    )


def test_require_two_dims():
    """Test that inv raises an error when there are less than two dimensions"""
    a = num.random.randn(10).astype(np.float64)
    msg = "Array must be at least two-dimensional."
    with pytest.raises(LinAlgError, match=msg):
        num.linalg.inv(a)


@pytest.mark.parametrize("shape", ((5, 10), (1, 5, 10, 20)))
def test_non_square_matrix_error(shape):
    """Test that inv raises an error for non-square matrices"""
    a = num.random.randn(*shape).astype(np.float64)
    msg = "Last 2 dimensions of the array must be square."
    with pytest.raises(LinAlgError, match=msg):
        num.linalg.inv(a)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
