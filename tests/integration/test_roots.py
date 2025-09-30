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


def test_zero_polynomial():
    arr = np.array([0, 0, 0])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_leading_zeros():
    arr = np.array([0, 0, 1, -2, 1])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_trailing_zeros_only_constant():
    arr = np.array([0, 0, 5])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_constant_polynomial():
    arr = np.array([5])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_linear_polynomial():
    arr = np.array([2, -6])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_quadratic_real_roots():
    arr = np.array([1, -3, 2])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_quadratic_complex_roots():
    arr = np.array([1, 0, 1])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_cubic_polynomial():
    arr = np.array([1, -6, 11, -6])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result.real, expected.real)


def test_quartic_polynomial():
    arr = np.array([1, 0, -2, 0, 1])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_repeated_roots():
    arr = np.array([1, -4, 6, -4, 1])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    # we need to increase tolerance since there is a numerical
    # instability for the result of eigvals when different BLAS
    # libraries are used
    assert allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_large_coefficients():
    arr = np.array([1e15, -2e15, 1e15])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_small_coefficients():
    arr = np.array([1e-15, -2e-15, 1e-15])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_complex_coefficients():
    arr = np.array([1 + 1j, -2 - 2j, 1 + 1j])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_nan_coefficients():
    arr = np.array([1, np.nan, 1])
    arr_num = num.array(arr)

    message = "Array must not contain infs or NaNs"
    with pytest.raises(np.linalg.LinAlgError, match=message):
        num.roots(arr_num)
    with pytest.raises(np.linalg.LinAlgError, match=message):
        np.roots(arr)


def test_inf_coefficients():
    arr = np.array([1, np.inf, 1])
    arr_num = num.array(arr)

    message = "Array must not contain infs or NaNs"
    with pytest.raises(np.linalg.LinAlgError, match=message):
        num.roots(arr_num)
    with pytest.raises(np.linalg.LinAlgError, match=message):
        np.roots(arr)


def test_empty_array():
    result = num.roots([])
    expected = np.roots([])
    assert allclose(result, expected)


def test_non_contiguous_array():
    large_p = [1, -3, 2, 0, 0]
    p = num.array(large_p)[:3]
    result = num.roots(p)
    expected = np.roots(np.array(p))
    assert allclose(result, expected)


def test_trailing_zeros():
    arr = np.array([1, -1, 0, 0])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_single_zero_coefficient():
    arr = np.array([0])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_nonzero_constant_polynomial_no_roots():
    # Test with a positive constant
    arr = np.array([7.5])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)

    # Test with a negative constant
    arr = np.array([-3.2])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def test_multidimensional_input_error():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Input must be a rank-1 array"):
        num.roots(arr)
    with pytest.raises(ValueError, match="Input must be a rank-1 array"):
        np.roots(arr)


def test_dtype_conversion():
    arr = np.array([1, -2, 1])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)
    # Result should be floating point even with integer input
    assert np.issubdtype(result.dtype, np.floating) or np.issubdtype(
        result.dtype, np.complexfloating
    )


def test_scalar_input():
    # NumPy has a bug where np.roots(p) throws:
    # "TypeError: dispatcher for __array_function__ did not return an iterable"
    # for any scalar p even though it should handle scalars. cuPyNumeric
    # correctly handles this by using atleast_1d() to convert the scalar to a
    # 1D array [p].
    arr = np.array([5])
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)  # Work around NumPy's scalar bug
    assert allclose(result, expected)


def test_mixed_zeros_and_coefficients():
    arr = np.array([0, 1, 0, -2, 0, 1, 0])  # Leading and trailing zeros
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


def random_symmetric_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def random_nonsymmetric_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n))


def random_complex_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, n))
    imag = rng.standard_normal((n, n))
    return real + 1j * imag


def random_hermitian_matrix(n, seed=None):
    A = random_complex_matrix(n, seed)
    return (A + A.conj().T) / 2


def random_positive_definite_matrix(n, seed=None):
    A = random_symmetric_matrix(n, seed)
    return A @ A.T + np.eye(n)  # Ensure positive definiteness


def defective_matrix():
    return np.array([[2, 1], [0, 2]])


@pytest.mark.parametrize(
    "matrix_fn",
    [
        random_symmetric_matrix,
        random_nonsymmetric_matrix,
        random_complex_matrix,
        random_hermitian_matrix,
        random_positive_definite_matrix,
    ],
)
def test_roots_from_random_matrix(matrix_fn):
    A = matrix_fn(4, seed=42)
    arr = np.poly(A)
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    # complex sort handles both real and complex
    assert allclose(result, expected)


def test_roots_from_defective_matrix():
    A = defective_matrix()
    arr = np.poly(A)
    arr_num = num.array(arr)

    result = num.roots(arr_num)
    expected = np.roots(arr)
    assert allclose(result, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
