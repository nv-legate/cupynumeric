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
from utils.generators import mk_seq_array

import cupynumeric as num


def test_scalar_inputs():
    messsage = "At least one array has zero dimension"
    with pytest.raises(ValueError, match=messsage):
        num.cross(5, 3)
    with pytest.raises(ValueError, match=messsage):
        np.cross(5, 3)


@pytest.mark.parametrize(
    "array_a, array_b",
    [
        ([1, 2, 3], [4, 5, 6]),
        ([[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [1, 0, 1]]),
        ([0, 0, 0], [1, 2, 3]),
    ],
)
def test_3d_cross_product(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b",
    [([1, 2], [3, 4]), ([[1, 2], [5, 6]], [[3, 4], [7, 8]])],
)
def test_2d_cross_product(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b",
    [([1, 2], [3, 4, 5]), ([[1, 2], [7, 8]], [[3, 4, 5], [1, 0, 1]])],
)
def test_mixed_2d_3d_vectors(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b", [([1, 2, 3], [2, 4, 6]), ([1, 2], [3, 6])]
)
def test_parallel_vectors(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b",
    [([1, 0, 0], [0, 1, 0]), ([0, 1, 0], [0, 0, 1]), ([0, 0, 1], [1, 0, 0])],
)
def test_orthogonal_vectors(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b", [([1, 2, 3], [-1, -2, -3]), ([1, 2], [-1, -2])]
)
def test_opposite_vectors(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b",
    [
        (np.array([1, 0, 0]), np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])),
        (np.array([[1, 0, 0], [0, 1, 0]]), np.array([0, 0, 1])),
        (np.array([[[1, 2, 3]]]), np.array([4, 5, 6])),  # a has extra dims
        (np.array([1, 2, 3]), np.array([[[4, 5, 6]]])),  # b has extra dims
        (np.random.rand(2, 1, 3), np.random.rand(1, 4, 3)),
    ],
)
def test_broadcasting(array_a, array_b):
    a_num = num.array(array_a)
    b_num = num.array(array_b)

    result = num.cross(a_num, b_num)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected)


def test_axis_override():
    a = num.array([[1, 2, 3], [4, 5, 6]]).T  # Shape (3, 2)
    b = num.array([[7, 8, 9], [10, 11, 12]]).T  # Shape (3, 2)

    # axis should override individual axis parameters
    result1 = num.cross(a, b, axis=0)
    result2 = num.cross(a, b, axisa=0, axisb=0, axisc=0)
    assert allclose(result1, result2)


@pytest.mark.parametrize(
    "a_shape, b_shape, axisa, axisb, axisc_expected",
    [
        ((1, 3, 5), (3, 4, 5), 1, 0, 2),  # Example: cross along different axes
        ((3, 1), (3, 1), 0, 0, 1),  # 2D vectors, different axes
    ],
)
def test_cross_with_different_axes(
    a_shape, b_shape, axisa, axisb, axisc_expected
):
    a = mk_seq_array(np, a_shape)
    b = mk_seq_array(np, b_shape)
    a_num = num.array(a)
    b_num = num.array(b)

    result = num.cross(a_num, b_num, axisa=axisa, axisb=axisb)
    expected = np.cross(a, b, axisa=axisa, axisb=axisb)
    assert allclose(result, expected)


def test_empty_arrays():
    a = np.array([])
    b = np.array([1, 2, 3])
    a_num = num.array(a)
    b_num = num.array(b)
    messsage = (
        "incompatible dimensions for cross "
        "product\n\\(dimension must be 2 or "
        "3\\)"
    )

    with pytest.raises(ValueError, match=messsage):
        np.cross(a, b)

    with pytest.raises(ValueError, match=messsage):
        num.cross(a_num, b_num)


@pytest.mark.parametrize(
    "array_a, array_b",
    [([1, np.nan, 3], [4, 5, 6]), ([1, np.inf, 3], [4, 5, 6])],
)
def test_nan_inf_values(array_a, array_b):
    a = num.array(array_a)
    b = num.array(array_b)
    result = num.cross(a, b)
    expected = np.cross(array_a, array_b)
    assert allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    "array_a, array_b",
    [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 2], [3, 4]),
        ([1, 2], [3, 4, 5]),
        ([1, 2, 3], [4, 5]),
    ],
)
@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        # Different dtypes
        (np.int32, np.float64),
        (np.float32, np.int64),
        (np.complex64, np.float32),
        (np.float64, np.complex128),
        (np.int64, np.complex64),
        (np.float32, np.float64),
        # Same dtype
        (np.int32, np.int32),
        (np.int64, np.int64),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
    ],
)
def test_dtypes(array_a, array_b, dtype_a, dtype_b):
    array1 = np.array(array_a, dtype=dtype_a)
    array2 = np.array(array_b, dtype=dtype_b)
    a = num.array(array1)
    b = num.array(array2)
    result = num.cross(a, b)
    expected = np.cross(array1, array2)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "array_a, array_b", [([1], [2, 3]), ([1, 2, 3, 4], [5, 6, 7])]
)
def test_incompatible_dimensions(array_a, array_b):
    a = num.array(array_a)
    b = num.array(array_b)
    messsage = (
        "incompatible dimensions for cross "
        "product\n\\(dimension must be 2 or "
        "3\\)"
    )
    with pytest.raises(ValueError, match=messsage):
        np.cross(a, b)

    with pytest.raises(ValueError, match=messsage):
        num.cross(a, b)


@pytest.mark.parametrize(
    "array_a, array_b",
    [
        ([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12], [13, 14, 15]]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]),
    ],
)
def test_incompatible_shapes(array_a, array_b):
    a = num.array(array_a)
    b = num.array(array_b)
    messsage = (
        r"shape mismatch: objects cannot "
        r"be broadcast to a single shape.  "
        r"Mismatch is between arg 0 with "
        r"shape \(2,\) and arg 1 with "
        r"shape \(3,\)."
    )

    with pytest.raises(ValueError, match=messsage):
        np.cross(a, b)

    with pytest.raises(ValueError, match=messsage):
        num.cross(a, b)


@pytest.mark.parametrize(
    "array_a, array_b",
    [
        (
            [[[1, 0, 0], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]]],
            [[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 1]]],
        ),
        (
            [[[1, 0], [0, 1]], [[1, 1], [2, 3]]],
            [[[0, 1], [1, 0]], [[3, 2], [1, 1]]],
        ),
    ],
)
def test_batch_cross_product(array_a, array_b):
    a = num.array(array_a)
    b = num.array(array_b)

    result = num.cross(a, b)
    expected = np.cross(a, b)
    assert allclose(result, expected)


def test_negative_axis_indices():
    a = num.array([[1, 2, 3], [4, 5, 6]]).T
    b = num.array([[7, 8, 9], [10, 11, 12]]).T

    result = num.cross(a, b, axisa=-2, axisb=-2)
    expected = np.cross(a, b, axisa=0, axisb=0)
    assert allclose(result, expected)


@pytest.mark.parametrize(
    "axis_name, axis_value, dim", [("axisa", 5, 1), ("axisb", -5, 1)]
)
def test_axis_validation(axis_name, axis_value, dim):
    a = mk_seq_array(np, (3,))
    b = mk_seq_array(np, (3,))
    a_num = mk_seq_array(num, (3,))
    b_num = mk_seq_array(num, (3,))

    kwargs = {axis_name: axis_value}

    message = (
        f"{axis_name}: axis {axis_value} is out of bounds for array of "
        f"dimension {dim}"
    )
    with pytest.raises(ValueError, match=message):
        num.cross(a_num, b_num, **kwargs)
    with pytest.raises(ValueError, match=message):
        np.cross(a, b, **kwargs)


def test_zero_dimension_arrays():
    a = np.array(5)
    b = mk_seq_array(np, (10000, 3))
    a_num = num.array(a)
    b_num = mk_seq_array(num, (10000, 3))

    messsage = "At least one array has zero dimension"

    with pytest.raises(ValueError, match=messsage):
        num.cross(a_num, b_num)
    with pytest.raises(ValueError, match=messsage):
        np.cross(a, b)


@pytest.mark.parametrize("order", ["C", "F"])
def test_memory_layout_variations(order):
    a = num.array([[1, 2, 3], [4, 5, 6]], order=order)
    b = num.array([[7, 8, 9], [10, 11, 12]], order=order)
    result = num.cross(a, b)
    expected = np.cross(a, b)
    assert allclose(result, expected)


def test_cross_product_with_sliced_arrays():
    a_large = mk_seq_array(np, (10000, 3))
    b_large = mk_seq_array(np, (10000, 3))
    a_large_num = mk_seq_array(num, (10000, 3))
    b_large_num = mk_seq_array(num, (10000, 3))

    a_slice = a_large[:, :3]
    b_slice = b_large[:, :3]
    a_slice_num = a_large_num[:, :3]
    b_slice_num = b_large_num[:, :3]

    result_slice_num = num.cross(a_slice_num, b_slice_num)
    result_slice = np.cross(a_slice, b_slice)
    assert allclose(result_slice_num, result_slice)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
