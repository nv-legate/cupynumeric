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

import cupynumeric as num
import numpy as np
import pytest
from utils.comparisons import allclose
from utils.generators import mk_seq_array


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1,), (1,)),
        ((3,), (3, 4)),
        ((3, 4), (2, 3)),
        ((3, 4), (2, 1, 4)),
        ((3, 5, 6), (2, 3)),
        ((3, 5, 6), (2, 3, 4)),
    ],
)
def test_basic(a_shape, b_shape):
    a = mk_seq_array(np, a_shape)
    b = mk_seq_array(np, b_shape)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "ar1, ar2",
    [([], []), (mk_seq_array(np, (3,)), []), ([], mk_seq_array(np, (3,)))],
)
def test_empty_arrays(ar1, ar2):
    ar1_num = num.array(ar1)
    ar2_num = num.array(ar2)

    result = num.in1d(ar1_num, ar2_num)
    expected = np.in1d(ar1, ar2)
    assert allclose(result, expected)


@pytest.mark.parametrize("assume_unique", [False, True])
def test_assume_unique(assume_unique):
    a = mk_seq_array(np, (5,))
    b = mk_seq_array(np, (4,))
    a_num = num.array(a)
    b_num = num.array(b)
    result = num.in1d(a_num, b_num, assume_unique=assume_unique)
    expected = np.in1d(a, b, assume_unique=assume_unique)
    assert allclose(result, expected)


@pytest.mark.parametrize("invert", [False, True])
def test_invert(invert):
    a = mk_seq_array(np, (5,))
    b = mk_seq_array(np, (4,))
    a_num = num.array(a)
    b_num = num.array(b)
    result = num.in1d(a_num, b_num, invert=invert)
    expected = np.in1d(a, b, invert=invert)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
        (np.int32, np.complex64),
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ],
)
def test_in1d_float_complex(a_dtype, b_dtype):
    a = mk_seq_array(np, (5,)).astype(a_dtype)
    b = mk_seq_array(np, (4,)).astype(b_dtype)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_nan_values():
    a = mk_seq_array(np, (4,)).astype(np.float64)
    np.insert(a, 2, np.nan)
    b = mk_seq_array(np, (2,)).astype(np.float64)
    np.insert(b, 1, np.nan)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_inf_values():
    a = mk_seq_array(np, (4,)).astype(np.float64)
    np.insert(a, 1, np.inf)
    np.insert(a, 3, -np.inf)
    b = mk_seq_array(np, (2,)).astype(np.float64)
    np.insert(b, 1, np.inf)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_large_arrays():
    a = mk_seq_array(np, (10000,))
    b = mk_seq_array(np, (1000,))
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((5,), (5,)),
        ((4,), (4,)),
        ((1,), (1,)),
        ((0,), (0,)),
        ((2, 3), (3, 2)),
        ((3, 2), (2, 3)),
        ((2, 2, 2), (2, 2, 2)),
        ((1, 4, 5), (5,)),
        ((3, 1, 1), (1, 3, 1)),
        ((6, 0), (0, 6)),
        ((0, 6), (6, 0)),
        ((2, 2, 0), (0, 2, 2)),
        ((10, 1), (1, 10)),
        ((1, 1, 1), (1,)),
        ((7, 3), (3, 7)),
        ((2, 3, 4), (4, 3, 2)),
    ],
)
def test_in1d_invert(invert, shape1, shape2):
    a = mk_seq_array(np, shape1)
    b = mk_seq_array(np, shape2)
    a_num = num.array(a)
    b_num = num.array(b)
    result = num.in1d(a_num, b_num, invert=invert)
    expected = np.in1d(a, b, invert=invert)
    assert allclose(result, expected)


@pytest.mark.parametrize("kind", [None, "sort", "table"])
@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((5,), (5,)),
        ((4,), (4,)),
        ((1,), (1,)),
        ((0,), (0,)),
        ((2, 3), (3, 2)),
        ((3, 2), (2, 3)),
        ((2, 2, 2), (2, 2, 2)),
        ((1, 4, 5), (5,)),
        ((3, 1, 1), (1, 3, 1)),
        ((6, 0), (0, 6)),
        ((0, 6), (6, 0)),
        ((2, 2, 0), (0, 2, 2)),
        ((10, 1), (1, 10)),
        ((1, 1, 1), (1,)),
        ((7, 3), (3, 7)),
        ((2, 3, 4), (4, 3, 2)),
    ],
)
def test_in1d_kind(kind, shape1, shape2):
    a = mk_seq_array(np, shape1)
    b = mk_seq_array(np, shape2)
    a_num = num.array(a)
    b_num = num.array(b)

    expected = np.in1d(a, b, kind=kind)
    result = num.in1d(a_num, b_num, kind=kind)
    assert allclose(result, expected)


def test_in1d_scalar_input():
    a = np.array(10000)
    b = mk_seq_array(np, (3,))
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_in1d_scalar_ar2():
    a = mk_seq_array(np, (3,))
    b = np.array(10000)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_one_duplicate():
    a = mk_seq_array(np, (3,))
    b = mk_seq_array(np, (3,))
    b[1] = 10000
    b[2] = 10000
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_both_duplicates():
    a = mk_seq_array(np, (3,))
    a[0] = 10000
    a[2] = 10000
    b = mk_seq_array(np, (3,))
    b[1] = 10000
    b[2] = 10000
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_float_duplicates():
    a = mk_seq_array(np, (3,))
    a[0] = 10000.0
    a[2] = 10000.0
    b = mk_seq_array(np, (3,))
    b[1] = 10000.0
    b[2] = 10000.0
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_nan_duplicates():
    a = mk_seq_array(np, (3,)).astype(np.float64)
    np.insert(a, 0, np.nan)
    np.insert(a, 2, np.nan)
    b = mk_seq_array(np, (3,)).astype(np.float64)
    np.insert(b, 1, np.nan)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


@pytest.mark.parametrize("invert", [False, True])
def test_in1d_table_out_of_range(invert):
    # Large range, but only a few values in ar2
    ar1 = mk_seq_array(np, (7,))
    ar1[1] = -100000
    ar1[6] = 500000
    ar2 = mk_seq_array(np, (3,)) + 100000
    ar1_num = num.array(ar1)
    ar2_num = num.array(ar2)

    expected = np.in1d(ar1, ar2, kind="table", invert=invert)
    result = num.in1d(ar1_num, ar2_num, kind="table", invert=invert)
    assert allclose(result, expected)


def test_noncontiguous():
    a = np.random.rand(1000, 1000) * 10000 + 10000
    a_num = num.array(a)
    a = a.T
    a_num = a_num.T
    a_slice = a[100:200, 100:200]
    a_num_slice = a_num[100:200, 100:200]
    result = num.in1d(a_num_slice, a_num)
    expected = np.in1d(a_slice, a)
    assert allclose(result, expected)


# def test_strided_slices():
#     a = np.arange(100).reshape(10, 10)
#     a_num = num.array(a)

#     a_slice = a[::2, ::3]
#     a_num_slice = a_num[::2, ::3]

#     result = num.in1d(a_num_slice, a_num)
#     expected = np.in1d(a_slice, a)
#     assert allclose(result, expected)


@pytest.mark.parametrize(
    "a_factory, b",
    [
        # Large array with only a few distinct values - all matches
        (lambda: np.full(10**7, 5), np.array([5])),
        # Large array with few distinct values - partial matches
        (
            lambda: np.concatenate(
                [np.full(5 * 10**6, 5), np.full(5 * 10**6, 7)]
            ),
            np.array([5, 9]),
        ),
        # Large array with few distinct values - no matches
        (lambda: np.full(10**7, 5), np.array([1, 2, 3])),
    ],
)
def test_large_arrays_few_distinct_values(a_factory, b):
    a = a_factory()
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)


def test_large_complex_numbers():
    a = np.array(
        [
            complex(100000e6, 200000e6),
            complex(-300000e6, 400000e6),
            complex(500000e6, -600000e6),
            complex(-700000e6, -800000e6),
            complex(100000e6, 200000e6),
            complex(900000e6, 1000000e6),
            complex(0, 1000000e6),
            complex(1000000e6, 0),
        ],
        dtype=np.complex128,
    )

    b = np.array(
        [
            complex(10000e6, 200000e6),
            complex(-300000e6, 4000000e6),
            complex(100000e9, 200000e9),
            complex(0, 100000e8),
            complex(-100000e8, 0),
        ],
        dtype=np.complex128,
    )

    a_num = num.array(a)
    b_num = num.array(b)

    # Test basic in1d
    result = num.in1d(a_num, b_num)
    expected = np.in1d(a, b)
    assert allclose(result, expected)

    # Test with invert=True
    result_inv = num.in1d(a_num, b_num, invert=True)
    expected_inv = np.in1d(a, b, invert=True)
    assert allclose(result_inv, expected_inv)

    # Test with assume_unique=True
    result_unique = num.in1d(a_num, b_num, assume_unique=True)
    expected_unique = np.in1d(a, b, assume_unique=True)
    assert allclose(result_unique, expected_unique)


# ============================================================================
# Tests for isin function
# ============================================================================


@pytest.mark.parametrize(
    "element_shape, test_shape",
    [
        ((1,), (1,)),
        ((3,), (5,)),
        ((2, 3), (4,)),
        ((2, 3), (2, 3)),
        ((3, 4, 5), (10,)),
        ((4, 5), (2, 3, 4)),
        ((0,), (5,)),
        ((5,), (0,)),
        ((0,), (0,)),
        ((0, 5), (10,)),
        ((5, 0), (10,)),
        ((2, 0, 3), (5,)),
        ((0, 0), (0,)),
    ],
)
def test_isin_basic(element_shape, test_shape):
    """Test basic isin functionality with various shapes."""
    element = mk_seq_array(np, element_shape)
    test_elements = mk_seq_array(np, test_shape)
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)

    # Check that shape is preserved
    assert result.shape == element.shape
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "element, test_elements",
    [
        ([], []),
        (mk_seq_array(np, (3, 4)), []),
        ([], mk_seq_array(np, (3,))),
        (mk_seq_array(np, (2, 0, 3)), mk_seq_array(np, (4,))),
    ],
)
def test_isin_empty_arrays(element, test_elements):
    """Test isin with empty arrays."""
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert allclose(result, expected)


def test_isin_shape_preservation():
    """Test that isin preserves the shape of the element array."""
    shapes = [(5,), (2, 3), (2, 3, 4), (1, 1, 1), (10, 1), (1, 10)]

    for shape in shapes:
        element = mk_seq_array(np, shape)
        test_elements = mk_seq_array(np, (5,))
        element_num = num.array(element)
        test_elements_num = num.array(test_elements)

        result = num.isin(element_num, test_elements_num)
        expected = np.isin(element, test_elements)

        assert result.shape == shape
        assert expected.shape == shape
        assert allclose(result, expected)


@pytest.mark.parametrize("assume_unique", [False, True])
def test_isin_assume_unique(assume_unique):
    """Test isin with assume_unique parameter."""
    element = mk_seq_array(np, (3, 4))
    test_elements = mk_seq_array(np, (5,))
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(
        element_num, test_elements_num, assume_unique=assume_unique
    )
    expected = np.isin(element, test_elements, assume_unique=assume_unique)
    assert allclose(result, expected)


@pytest.mark.parametrize("invert", [False, True])
def test_isin_invert(invert):
    """Test isin with invert parameter."""
    element = mk_seq_array(np, (3, 4))
    test_elements = mk_seq_array(np, (5,))
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num, invert=invert)
    expected = np.isin(element, test_elements, invert=invert)
    assert allclose(result, expected)


@pytest.mark.parametrize("kind", [None, "sort", "table"])
def test_isin_kind(kind):
    """Test isin with different kind algorithms."""
    element = mk_seq_array(np, (3, 4))
    test_elements = mk_seq_array(np, (5,))
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num, kind=kind)
    expected = np.isin(element, test_elements, kind=kind)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "element_dtype, test_dtype",
    [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
        (np.int32, np.complex64),
        (np.complex64, np.float32),
        (np.complex128, np.float64),
        (np.int32, np.int64),
        (np.float32, np.float64),
    ],
)
def test_isin_mixed_dtypes(element_dtype, test_dtype):
    """Test isin with mixed dtypes."""
    element = mk_seq_array(np, (3, 4)).astype(element_dtype)
    test_elements = mk_seq_array(np, (5,)).astype(test_dtype)
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert allclose(result, expected)


def test_isin_nan_values():
    """Test isin with NaN values."""
    element = mk_seq_array(np, (3, 4)).astype(np.float64)
    element[1, 2] = np.nan
    test_elements = mk_seq_array(np, (3,)).astype(np.float64)
    test_elements[1] = np.nan

    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert allclose(result, expected)


def test_isin_inf_values():
    """Test isin with infinity values."""
    element = mk_seq_array(np, (3, 4)).astype(np.float64)
    element[0, 1] = np.inf
    element[2, 3] = -np.inf
    test_elements = mk_seq_array(np, (3,)).astype(np.float64)
    test_elements[1] = np.inf

    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert allclose(result, expected)


def test_isin_scalar_element():
    """Test isin with scalar element."""
    element = np.array(5)
    test_elements = mk_seq_array(np, (10,))
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert result.shape == element.shape
    assert allclose(result, expected)


def test_isin_scalar_test_elements():
    """Test isin with scalar test_elements."""
    element = mk_seq_array(np, (3, 4))
    test_elements = np.array(5)
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert allclose(result, expected)


def test_isin_duplicates():
    """Test isin with duplicate values."""
    element = mk_seq_array(np, (3, 4))
    element[0, 0] = 100
    element[2, 3] = 100
    test_elements = mk_seq_array(np, (5,))
    test_elements[1] = 100
    test_elements[4] = 100

    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert allclose(result, expected)


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize(
    "element_shape, test_shape",
    [
        ((2, 3), (5,)),
        ((4, 5), (10,)),
        ((1, 1), (1,)),
        ((10,), (20,)),
        ((3, 4, 5), (15,)),
    ],
)
def test_isin_invert_shapes(invert, element_shape, test_shape):
    """Test isin with invert parameter and various shapes."""
    element = mk_seq_array(np, element_shape)
    test_elements = mk_seq_array(np, test_shape)
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num, invert=invert)
    expected = np.isin(element, test_elements, invert=invert)
    assert result.shape == element_shape
    assert allclose(result, expected)


def test_isin_large_arrays():
    """Test isin with large arrays."""
    element = mk_seq_array(np, (100, 100))
    test_elements = mk_seq_array(np, (1000,))
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert result.shape == element.shape
    assert allclose(result, expected)


def test_isin_complex_numbers():
    """Test isin with complex numbers."""
    element = np.array(
        [
            [complex(1, 2), complex(3, 4), complex(5, 6)],
            [complex(7, 8), complex(1, 2), complex(9, 10)],
        ],
        dtype=np.complex128,
    )
    test_elements = np.array(
        [complex(1, 2), complex(5, 6), complex(11, 12)], dtype=np.complex128
    )

    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert result.shape == element.shape
    assert allclose(result, expected)


def test_isin_vs_in1d_difference():
    """Test that isin preserves shape while in1d flattens."""
    element = mk_seq_array(np, (3, 4))
    test_elements = mk_seq_array(np, (5,))
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    # isin should preserve shape
    result_isin = num.isin(element_num, test_elements_num)
    assert result_isin.shape == (3, 4)

    # in1d should flatten
    result_in1d = num.in1d(element_num, test_elements_num)
    assert result_in1d.shape == (12,)

    # But the flattened versions should be equal
    assert allclose(result_isin.ravel(), result_in1d)


def test_isin_noncontiguous():
    """Test isin with non-contiguous arrays."""
    element = np.random.rand(50, 50) * 100
    element_num = num.array(element)
    element = element.T
    element_num = element_num.T
    element_slice = element[10:20, 10:20]
    element_num_slice = element_num[10:20, 10:20]

    test_elements = mk_seq_array(np, (50,))
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num_slice, test_elements_num)
    expected = np.isin(element_slice, test_elements)
    assert result.shape == element_slice.shape
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "element_factory, test_elements",
    [
        # Large array with only a few distinct values - all matches
        (lambda: np.full((1000, 1000), 5), np.array([5])),
        # Large array with few distinct values - partial matches
        (
            lambda: np.concatenate(
                [np.full((500, 1000), 5), np.full((500, 1000), 7)]
            ),
            np.array([5, 9]),
        ),
        # Large array with few distinct values - no matches
        (lambda: np.full((1000, 1000), 5), np.array([1, 2, 3])),
    ],
)
def test_isin_large_arrays_few_distinct_values(element_factory, test_elements):
    """Test isin with large arrays containing few distinct values."""
    element = element_factory()
    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(element_num, test_elements_num)
    expected = np.isin(element, test_elements)
    assert result.shape == element.shape
    assert allclose(result, expected)


@pytest.mark.parametrize("invert", [False, True])
def test_isin_table_out_of_range(invert):
    """Test isin with table kind and large range values."""
    element = mk_seq_array(np, (3, 4))
    element[0, 1] = -100000
    element[2, 3] = 500000
    test_elements = mk_seq_array(np, (3,)) + 100000

    element_num = num.array(element)
    test_elements_num = num.array(test_elements)

    result = num.isin(
        element_num, test_elements_num, kind="table", invert=invert
    )
    expected = np.isin(element, test_elements, kind="table", invert=invert)
    assert result.shape == element.shape
    assert allclose(result, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
