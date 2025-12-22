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
from cupynumeric.linalg import LinAlgError

import cupynumeric as num


@pytest.mark.parametrize("size", (10, 50))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.skipif
def test_pinv_basic(size, dtype):
    """Test basic pinv functionality with default rtol"""
    a = np.random.randn(size, size).astype(dtype)
    a_pinv = num.linalg.pinv(a)

    # Verify: A @ A+ @ A ≈ A
    result = a @ a_pinv @ a
    assert allclose(a, result, rtol=1e-3, atol=1e-3, check_dtype=False)


@pytest.mark.parametrize("size", (50, 100, 150))
@pytest.mark.parametrize("rtol", (1e-5, 1e-3))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_pinv_different_rtol(size, rtol, dtype):
    """Test pinv with different rtol values on medium-sized matrices"""
    a = np.random.randn(size, size).astype(dtype)
    a_pinv = num.linalg.pinv(a, rtol=rtol)
    result = a @ a_pinv @ a
    assert allclose(a, result, rtol=1e-2, atol=1e-2, check_dtype=False)


def test_pinv_tall_matrix():
    """Test pinv with tall rectangular matrices (M > N)"""
    m, n = 100, 50
    a = np.random.randn(m, n).astype(np.float64)

    a_pinv = num.linalg.pinv(a)
    assert a_pinv.shape == (n, m)

    result = a @ a_pinv @ a
    assert allclose(a, result, rtol=1e-3, atol=1e-3, check_dtype=False)


def test_pinv_fat_matrix_not_supported():
    """Test pinv with fat rectangular matrix (M < N)"""
    a = num.random.randn(10, 20).astype(np.float64)
    with pytest.raises(
        NotImplementedError, match=".*pinv is not supported for M < N.*"
    ):
        num.linalg.pinv(a)


def test_1D_matrix_not_supported():
    """Test pinv with fat rectangular matrix (M < N)"""
    a = num.random.randn(10).astype(np.float64)
    with pytest.raises(
        LinAlgError, match=".*Array must be at least two-dimensional.*"
    ):
        num.linalg.pinv(a)


def test_3D_matrix_not_supported():
    """Test pinv with unsupported 3D matrix"""
    a = num.random.randn(10, 10, 10).astype(np.float64)
    with pytest.raises(
        NotImplementedError,
        match=".*cuPyNumeric does not yet support stacked 2d arrays.*",
    ):
        num.linalg.pinv(a)


def test_rcond_not_supported():
    """Test that rcond parameter is not supported"""
    a = num.random.randn(10, 10).astype(np.float64)
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        num.linalg.pinv(a, rcond=1e-5)


def test_hermitian_not_supported():
    """Test that hermitian parameter is not supported"""
    a = num.random.randn(10, 10).astype(np.float64)
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        num.linalg.pinv(a, hermitian=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
