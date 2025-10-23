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

import cupynumeric as num
from utils.generators import mk_seq_array
from utils.comparisons import allclose


DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]

DIM = 5
SIZES = [
    (0,),
    (1),
    (DIM),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]


def test_three_keys():
    a = mk_seq_array(np, 100)
    b = mk_seq_array(np, 100)
    c = mk_seq_array(np, 100)

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c))
    result = num.lexsort((a_num, b_num, c_num))
    assert allclose(result, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtypes(dtype):
    a = mk_seq_array(np, 100).astype(dtype)
    b = mk_seq_array(np, 100).astype(dtype)

    a_num = num.array(a)
    b_num = num.array(b)

    result = num.lexsort((a_num, b_num))
    expected = np.lexsort((a, b))
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "dtype_a",
    [
        np.int32,
        np.uint32,
        np.int8,
        np.uint64,
        np.int16,
        np.float32,
        np.float64,
        np.complex64,
    ],
)
@pytest.mark.parametrize(
    "dtype_b",
    [
        np.int64,
        np.int32,
        np.uint64,
        np.float32,
        np.float64,
        np.float16,
        np.complex64,
        np.complex128,
    ],
)
def test_mismatched_dtypes(dtype_a, dtype_b):
    a = mk_seq_array(np, 500).astype(dtype_a)
    b = mk_seq_array(np, 500).astype(dtype_b)

    a_num = num.array(a)
    b_num = num.array(b)

    result = num.lexsort((a_num, b_num))
    expected = np.lexsort((a, b))
    assert allclose(result, expected)


@pytest.mark.parametrize("size", SIZES)
def test_sizes(size):
    a = mk_seq_array(np, size)
    b = mk_seq_array(np, size)

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


@pytest.mark.parametrize("mod_val", [10, 50, 100])
def test_with_duplicates(mod_val):
    size = 1000
    # Create arrays with duplicates using modulo
    a = mk_seq_array(np, size) % mod_val
    b = mk_seq_array(np, size) % (mod_val * 2)

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


@pytest.mark.parametrize("group_size", [5, 20, 100])
def test_with_grouped_duplicates(group_size):
    size = 1000
    # Create arrays with groups of same values
    a = mk_seq_array(np, size) // group_size
    b = (mk_seq_array(np, size) * 3) % (group_size * 2)

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_stability():
    a = num.array([10000000, 10000000, 10000000, 20000000, 20000000, 20000000])
    b = num.array([30000000, 30000000, 30000000, 40000000, 40000000, 40000000])
    result = num.lexsort((a, b))
    expected = np.lexsort((a, b))
    assert allclose(result, expected)


def test_all_equal():
    a = num.array([50000000, 50000000, 50000000, 50000000])
    b = num.array([10000000, 10000000, 10000000, 10000000])
    result = num.lexsort((a, b))
    expected = np.lexsort((a, b))
    assert allclose(result, expected)


def test_key_priority():
    a = num.array([20000000, 10000000, 30000000])  # secondary sort key
    b = num.array([10000000, 20000000, 10000000])  # primary sort key
    result = num.lexsort((a, b))
    expected = np.lexsort((a, b))
    assert allclose(result, expected)

    sorted_indices = result
    assert b[sorted_indices[0]] <= b[sorted_indices[1]]


def test_mismatched_shapes():
    a = mk_seq_array(np, 100)
    b = mk_seq_array(np, 50)

    a_num = num.array(a)
    b_num = num.array(b)

    with pytest.raises(ValueError):
        num.lexsort((a_num, b_num))

    with pytest.raises(ValueError):
        np.lexsort((a, b))


def test_negative_values():
    a = num.array([-10000000, -20000000, 0, 10000000, 20000000])
    b = num.array([50000000, -30000000, 0, -10000000, 40000000])

    result = num.lexsort((a, b))
    expected = np.lexsort((a, b))
    assert allclose(result, expected)


def test_none_input():
    with pytest.raises(TypeError):
        np.lexsort(None)

    with pytest.raises(TypeError):
        num.lexsort(None)


@pytest.mark.parametrize("axis", [0, -1])
def test_axis_1d(axis):
    a = mk_seq_array(np, 1000)
    b = mk_seq_array(np, 1000)
    c = mk_seq_array(np, 1000)

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c), axis=axis)
    result = num.lexsort((a_num, b_num, c_num), axis=axis)
    assert allclose(result, expected)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_axis_2d(axis):
    shape = (10, 100)
    a = mk_seq_array(np, shape)
    b = mk_seq_array(np, shape)
    c = mk_seq_array(np, shape)

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c), axis=axis)
    result = num.lexsort((a_num, b_num, c_num), axis=axis)
    assert allclose(result, expected)


def test_bool_arrays():
    a = np.array([True, False, True, False])
    b = np.array([False, True, False, True])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "ar1",
    [
        np.array([np.nan, 1000000, 2000000, 3000000]),
        np.array([np.nan, 1000000, np.nan, 3000000]),
        np.array([np.inf, -np.inf, 1000000, 2000000]),
        np.array([np.nan, np.inf, -np.inf, 1000000]),
    ],
)
@pytest.mark.parametrize(
    "ar2",
    [
        np.array([1000000, 2000000, 3000000, 4000000]),
        np.array([1000000, np.nan, 3000000, 4000000]),
        np.array([3000000, 4000000, np.inf, -np.inf]),
        np.array([np.inf, -np.inf, np.nan, 3000000]),
    ],
)
def test_special_float_values(ar1, ar2):
    ar1_num = num.array(ar1)
    ar2_num = num.array(ar2)

    expected = np.lexsort((ar1, ar2))
    result = num.lexsort((ar1_num, ar2_num))
    assert allclose(result, expected)


def test_complex_numbers():
    a = np.array(
        [
            1000000 + 2000000j,
            3000000 + 4000000j,
            1000000 + 2000000j,
            5000000 + 6000000j,
        ]
    )
    b = np.array(
        [
            7000000 + 8000000j,
            9000000 + 10000000j,
            11000000 + 12000000j,
            13000000 + 14000000j,
        ]
    )

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_single_key():
    a = mk_seq_array(np, 1000)
    a_num = num.array(a)

    expected = np.lexsort((a,))
    result = num.lexsort((a_num,))
    assert allclose(result, expected)


def test_empty_keys_tuple():
    with pytest.raises((TypeError)):
        np.lexsort(())

    with pytest.raises((TypeError)):
        num.lexsort(())


def test_0d_array():
    a = np.array(42)
    a_num = num.array(a)
    expected = np.lexsort((a,))
    result = num.lexsort((a_num,))
    assert allclose(result, expected)


def test_multiple_0d_arrays():
    a = np.array(42)
    b = np.array(17)
    c = np.array(99)

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c))
    result = num.lexsort((a_num, b_num, c_num))
    assert allclose(result, expected)


def test_large_number_of_keys():
    arrays_np = [mk_seq_array(np, 100) for _ in range(10)]
    arrays_num = [num.array(arr) for arr in arrays_np]

    expected = np.lexsort(tuple(arrays_np))
    result = num.lexsort(tuple(arrays_num))
    assert allclose(result, expected)


def test_mixed_sign_bitwidth_keys():
    # int64 and uint64
    a_i64 = np.array([-1, -2, 3, 4], dtype=np.int64)
    b_u64 = np.array([10, 20, 30, 40], dtype=np.uint64)

    a_num = num.array(a_i64)
    b_num = num.array(b_u64)

    expected = np.lexsort((a_i64, b_u64))  # a_i64 secondary, b_u64 primary
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)

    # int8 and uint32
    a_i8 = np.array([-1, 10, -5, 5], dtype=np.int8)
    b_u32 = np.array([1000, 10, 1000, 20], dtype=np.uint32)

    a_num = num.array(a_i8)
    b_num = num.array(b_u32)

    expected = np.lexsort((a_i8, b_u32))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_all_special_float_values_keys():
    # Primary key is all NaN, secondary determines order
    a = np.array([1000000, 2000000, 3000000])  # secondary
    b = np.array([np.nan, np.nan, np.nan])  # primary

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)

    # Primary key is all Inf, secondary determines order
    a = np.array([1000000, 2000000, 3000000])  # secondary
    b = np.array([np.inf, np.inf, np.inf])  # primary

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_mixed_real_complex_keys():
    # Test case 1: Primary key is complex, secondary is real
    a_complex = np.array([1 + 1j, 2 + 2j, 1 + 1j, 3 + 3j])  # primary key
    b_real = np.array([4, 3, 5, 2])  # secondary key

    a_num = num.array(a_complex)
    b_num = num.array(b_real)

    # In both lexsort functions, first argument is primary key, second is secondary
    expected = np.lexsort((a_complex, b_real))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)

    # Test case 2: Primary key is real, secondary is complex
    a_real = np.array([4, 3, 5, 2])  # primary key
    b_complex = np.array([1 + 1j, 2 + 2j, 1 + 1j, 3 + 3j])  # secondary key

    a_num = num.array(a_real)
    b_num = num.array(b_complex)

    expected = np.lexsort((a_real, b_complex))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_mismatched_rank():
    a = mk_seq_array(np, (5, 5))  # 2D array
    b = mk_seq_array(np, 25)  # 1D array, same total elements

    a_num = num.array(a)
    b_num = num.array(b)

    # lexsort requires the same *shape*, not just the same size.
    with pytest.raises(ValueError):
        num.lexsort((a_num, b_num))

    with pytest.raises(ValueError):
        np.lexsort((a, b))


def test_invalid_axis():
    shape = (10, 100)
    a = mk_seq_array(np, shape)
    b = mk_seq_array(np, shape)
    a_num = num.array(a)
    b_num = num.array(b)

    # Test with an axis out of bounds for a 2D array
    with pytest.raises(np.exceptions.AxisError):
        np.lexsort((a, b), axis=2)

    with pytest.raises(np.exceptions.AxisError):
        num.lexsort((a_num, b_num), axis=2)

    # Test with another out-of-bounds axis
    with pytest.raises(np.exceptions.AxisError):
        np.lexsort((a, b), axis=-3)

    with pytest.raises(np.exceptions.AxisError):
        num.lexsort((a_num, b_num), axis=-3)


def test_single_element_different_dtypes():
    """Test lexsort with single element arrays of different dtypes"""
    a = np.array([42], dtype=np.int32)
    b = np.array([3.14], dtype=np.float64)
    c = np.array([1 + 2j], dtype=np.complex128)

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c))
    result = num.lexsort((a_num, b_num, c_num))
    assert allclose(result, expected)


def test_3d_arrays():
    """Test lexsort with 3D arrays"""
    shape = (2, 3, 4)
    a = mk_seq_array(np, shape)
    b = mk_seq_array(np, shape)
    c = mk_seq_array(np, shape)

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c), axis=1)
    result = num.lexsort((a_num, b_num, c_num), axis=1)
    assert allclose(result, expected)


def test_very_small_arrays():
    """Test lexsort with very small arrays (1-3 elements)"""
    for size in [1, 2, 3]:
        a = mk_seq_array(np, size)
        b = mk_seq_array(np, size)

        a_num = num.array(a)
        b_num = num.array(b)

        expected = np.lexsort((a, b))
        result = num.lexsort((a_num, b_num))
        assert allclose(result, expected)


def test_very_long_arrays():
    """Test lexsort with very long arrays (stress test)"""
    size = 50000  # Reduced from 10 million to 50k elements
    a = mk_seq_array(np, size)
    b = mk_seq_array(np, size)

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_axis_edge_cases():
    """Test lexsort with various axis edge cases"""
    shape = (5, 10, 15)
    a = mk_seq_array(np, shape)
    b = mk_seq_array(np, shape)

    a_num = num.array(a)
    b_num = num.array(b)

    # Test all valid axes
    for axis in [0, 1, 2, -1, -2, -3]:
        expected = np.lexsort((a, b), axis=axis)
        result = num.lexsort((a_num, b_num), axis=axis)
        assert allclose(result, expected)


def test_empty_arrays():
    """Test lexsort with empty arrays"""
    a = np.array([])
    b = np.array([])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_single_key_empty_array():
    """Test lexsort with single key and empty array"""
    a = np.array([])
    a_num = num.array(a)

    expected = np.lexsort((a,))
    result = num.lexsort((a_num,))
    assert allclose(result, expected)


def test_identical_keys():
    """Test lexsort when all keys are identical"""
    a = np.array([1, 1, 1, 1])
    b = np.array([2, 2, 2, 2])
    c = np.array([3, 3, 3, 3])

    a_num = num.array(a)
    b_num = num.array(b)
    c_num = num.array(c)

    expected = np.lexsort((a, b, c))
    result = num.lexsort((a_num, b_num, c_num))
    assert allclose(result, expected)


def test_negative_infinity_sorting():
    """Test lexsort with negative infinity values"""
    a = np.array([1, 2, 3, 4])
    b = np.array([-np.inf, -np.inf, 1, 2])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_mixed_inf_nan():
    """Test lexsort with mixed infinity and NaN values"""
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([np.inf, -np.inf, np.nan, 1, 2])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_complex_with_nan():
    """Test lexsort with complex numbers containing NaN"""
    a = np.array([1 + 1j, 2 + 2j, np.nan + 1j, 4 + 4j])
    b = np.array([1, 2, 3, 4])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_very_small_values():
    """Test lexsort with very small floating point values"""
    a = np.array([1e-10, 1e-20, 1e-30, 1e-40])
    b = np.array([1, 2, 3, 4])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


def test_very_large_values():
    """Test lexsort with very large floating point values"""
    a = np.array([1e10, 1e20, 1e30, 1e40])
    b = np.array([1, 2, 3, 4])

    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.lexsort((a, b))
    result = num.lexsort((a_num, b_num))
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "a, check_tuple_diff",
    [
        # 1D array - should return scalar 0
        ([3, 1, 2], False),
        # 2D array - each row is treated as a key
        ([[3, 1, 2], [1, 2, 3]], True),
        # 2D array with multiple rows
        ([[3, 1, 2, 4], [1, 2, 3, 1], [2, 1, 1, 3]], False),
        # 3D array - sorting happens along last axis
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], False),
    ],
)
def test_single_ndarray(a, check_tuple_diff):
    """Test lexsort with a single ndarray (not wrapped in tuple) of various dimensions"""
    a_np = np.array(a)
    a_num = num.array(a_np)

    expected = np.lexsort(a_np)
    result = num.lexsort(a_num)
    assert allclose(result, expected)
    assert result.shape == expected.shape

    # For 2D arrays, verify that single array behaves differently from tuple
    if check_tuple_diff:
        expected_tuple = np.lexsort((a_np,))
        result_tuple = num.lexsort((a_num,))
        assert allclose(result_tuple, expected_tuple)
        assert not allclose(result, expected_tuple)


def test_single_ndarray_2d_with_axis():
    """Test lexsort with a single 2D ndarray and different axis values"""
    a = np.array([[3, 1, 2], [1, 2, 3]])
    a_num = num.array(a)

    # Test with axis=-1 (default)
    expected = np.lexsort(a, axis=-1)
    result = num.lexsort(a_num, axis=-1)
    assert allclose(result, expected)

    # Test with axis=0
    expected_axis0 = np.lexsort(a, axis=0)
    result_axis0 = num.lexsort(a_num, axis=0)
    assert allclose(result_axis0, expected_axis0)


def test_single_ndarray_dtypes():
    """Test lexsort with single ndarray of different dtypes"""
    for dtype in [np.int32, np.float64, np.complex128]:
        a = mk_seq_array(np, (3, 4)).astype(dtype)
        a_num = num.array(a)

        expected = np.lexsort(a)
        result = num.lexsort(a_num)
        assert allclose(result, expected), f"Failed for dtype {dtype}"


@pytest.mark.parametrize(
    "a, b",
    [
        # Tuple of 1D arrays
        ([3, 1, 2], [1, 2, 3]),
        # Tuple of 2D arrays
        ([[3, 1, 2], [1, 2, 3]], [[1, 2, 3], [3, 1, 2]]),
        # Single 2D array (b is None to indicate single array case)
        ([[3, 1, 2], [1, 2, 3]], None),
    ],
)
def test_axis_none(a, b):
    """Test that axis=None raises TypeError for all input types"""
    a_np = np.array(a)
    a_num = num.array(a_np)

    if b is None:
        # Single array case (not wrapped in tuple)
        np_keys = a_np
        num_keys = a_num
    else:
        # Tuple of arrays case
        b_np = np.array(b)
        b_num = num.array(b_np)
        np_keys = (a_np, b_np)
        num_keys = (a_num, b_num)

    # NumPy raises TypeError for axis=None
    with pytest.raises(
        TypeError,
        match="'NoneType' object cannot be interpreted as an integer",
    ):
        np.lexsort(np_keys, axis=None)

    # cuPyNumeric should match NumPy's behavior
    with pytest.raises(
        TypeError,
        match="'NoneType' object cannot be interpreted as an integer",
    ):
        num.lexsort(num_keys, axis=None)


def test_string_arrays_fallback():
    """Test that lexsort falls back to NumPy for unsupported dtypes like strings"""
    # Example from NumPy documentation
    surnames = ("Hertz", "Galilei", "Hertz")
    first_names = ("Heinrich", "Galileo", "Gustav")

    # NumPy result
    expected = np.lexsort((first_names, surnames))

    # cuPyNumeric should fall back to NumPy with a warning
    with pytest.warns(
        RuntimeWarning,
        match="cuPyNumeric does not support.*falling back to NumPy",
    ):
        result = num.lexsort((first_names, surnames))

    # Results should match
    assert allclose(result, expected)
    # Result should still be a cuPyNumeric ndarray
    assert isinstance(result, num.ndarray)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
