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
import os

import numpy as np
import pytest
from utils.generators import mk_seq_array

import cupynumeric as num

EAGER_TEST = os.environ.get("CUPYNUMERIC_FORCE_THUNK", None) == "eager"


class TestPadConstant:
    """Test constant padding mode."""

    @pytest.mark.parametrize("pad_width", (1, 2, 3, (1, 2), ((1, 2),)))
    def test_1d_array(self, pad_width):
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, pad_width, mode="constant")
        res_num = num.pad(num_array, pad_width, mode="constant")

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("pad_width", (1, 2, ((1, 2), (2, 1))))
    def test_2d_array(self, pad_width):
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(np_array, pad_width, mode="constant")
        res_num = num.pad(num_array, pad_width, mode="constant")

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("constant_values", (0, 5, -1, 3.14))
    def test_constant_values(self, constant_values):
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(
            np_array, 2, mode="constant", constant_values=constant_values
        )
        res_num = num.pad(
            num_array, 2, mode="constant", constant_values=constant_values
        )

        assert np.array_equal(res_np, res_num)

    def test_constant_values_scalar_and_matrix_eager(self) -> None:
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array(np_array)

        res_np = np.pad(np_array, 1, mode="constant", constant_values=(2, 2))
        res_num = num.pad(
            num_array, 1, mode="constant", constant_values=(2, 2)
        )
        assert np.array_equal(res_np, res_num)

        const_matrix = np.array([[1, 2], [3, 4]])
        res_np = np.pad(
            np_array, 1, mode="constant", constant_values=const_matrix
        )
        res_num = num.pad(
            num_array, 1, mode="constant", constant_values=const_matrix
        )
        assert np.array_equal(res_np, res_num)

    def test_asymmetric_padding(self):
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(
            np_array, ((1, 2), (2, 1)), mode="constant", constant_values=5
        )
        res_num = num.pad(
            num_array, ((1, 2), (2, 1)), mode="constant", constant_values=5
        )

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("ndim", range(1, 4))
    def test_nd_array(self, ndim):
        shape = tuple(np.random.randint(2, 5) for _ in range(ndim))
        np_array = mk_seq_array(np, shape)
        num_array = mk_seq_array(num, shape)

        pad_width = np.random.randint(1, 3)

        res_np = np.pad(np_array, pad_width, mode="constant")
        res_num = num.pad(num_array, pad_width, mode="constant")

        assert np.array_equal(res_np, res_num)


class TestPadEdge:
    """Test edge padding mode."""

    def test_1d_array(self):
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 2, mode="edge")
        res_num = num.pad(num_array, 2, mode="edge")

        assert np.array_equal(res_np, res_num)

    def test_2d_array(self):
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(np_array, 1, mode="edge")
        res_num = num.pad(num_array, 1, mode="edge")

        assert np.array_equal(res_np, res_num)

    def test_asymmetric_padding(self):
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        num_array = num.array([[1, 2, 3], [4, 5, 6]])

        res_np = np.pad(np_array, ((1, 2), (2, 1)), mode="edge")
        res_num = num.pad(num_array, ((1, 2), (2, 1)), mode="edge")

        assert np.array_equal(res_np, res_num)


class TestPadReflect:
    """Test reflect padding mode."""

    def test_1d_array(self):
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 2, mode="reflect")
        res_num = num.pad(num_array, 2, mode="reflect")

        assert np.array_equal(res_np, res_num)

    def test_2d_array(self):
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        num_array = num.array([[1, 2, 3], [4, 5, 6]])

        res_np = np.pad(np_array, 1, mode="reflect")
        res_num = num.pad(num_array, 1, mode="reflect")

        assert np.array_equal(res_np, res_num)


class TestPadSymmetric:
    """Test symmetric padding mode."""

    def test_1d_array(self):
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 2, mode="symmetric")
        res_num = num.pad(num_array, 2, mode="symmetric")

        assert np.array_equal(res_np, res_num)

    def test_2d_array(self):
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        num_array = num.array([[1, 2, 3], [4, 5, 6]])

        res_np = np.pad(np_array, 1, mode="symmetric")
        res_num = num.pad(num_array, 1, mode="symmetric")

        assert np.array_equal(res_np, res_num)


class TestPadWrap:
    """Test wrap padding mode."""

    def test_1d_array(self):
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 2, mode="wrap")
        res_num = num.pad(num_array, 2, mode="wrap")

        assert np.array_equal(res_np, res_num)

    def test_2d_array(self):
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        num_array = num.array([[1, 2, 3], [4, 5, 6]])

        res_np = np.pad(np_array, 1, mode="wrap")
        res_num = num.pad(num_array, 1, mode="wrap")

        assert np.array_equal(res_np, res_num)


class TestPadStatistics:
    """Test statistical padding modes (mean, maximum, minimum, median)."""

    @pytest.mark.parametrize("mode", ("mean", "maximum", "minimum", "median"))
    def test_1d_array(self, mode):
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        num_array = num.array([1.0, 2.0, 3.0, 4.0, 5.0])

        res_np = np.pad(np_array, 2, mode=mode)
        res_num = num.pad(num_array, 2, mode=mode)

        assert np.allclose(res_np, res_num)

    @pytest.mark.parametrize("mode", ("mean", "maximum", "minimum", "median"))
    def test_2d_array(self, mode):
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        num_array = num.array([[1.0, 2.0], [3.0, 4.0]])

        res_np = np.pad(np_array, 1, mode=mode)
        res_num = num.pad(num_array, 1, mode=mode)

        assert np.allclose(res_np, res_num)

    def test_stat_length_kwarg(self):
        np_array = np.array([1.0, 2.0, 3.0, 4.0])
        num_array = num.array([1.0, 2.0, 3.0, 4.0])

        stat_length = (2, 1)

        res_np = np.pad(np_array, (2, 3), mode="mean", stat_length=stat_length)
        res_num = num.pad(
            num_array, (2, 3), mode="mean", stat_length=stat_length
        )

        assert np.allclose(res_np, res_num)


class TestPadEmpty:
    """Test empty padding mode."""

    def test_1d_array(self):
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 2, mode="empty")
        res_num = num.pad(num_array, 2, mode="empty")

        # For empty mode, just check shape
        assert res_np.shape == res_num.shape


class TestPadErrors:
    """Test error handling."""

    def test_invalid_mode(self):
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        with pytest.raises(ValueError):
            np.pad(np_array, 2, mode="invalid_mode")
        with pytest.raises(ValueError):
            num.pad(num_array, 2, mode="invalid_mode")

    def test_negative_pad_width(self):
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        with pytest.raises(ValueError):
            np.pad(np_array, -1, mode="constant")
        with pytest.raises(ValueError):
            num.pad(num_array, -1, mode="constant")

    def test_unsupported_kwargs(self):
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        with pytest.raises(ValueError):
            np.pad(np_array, 2, mode="edge", constant_values=5)
        with pytest.raises(ValueError):
            num.pad(num_array, 2, mode="edge", constant_values=5)


class TestPadCornerCases:
    """Test corner cases and edge conditions."""

    def test_zero_padding(self):
        """Test padding with pad_width=0."""
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 0, mode="constant")
        res_num = num.pad(num_array, 0, mode="constant")

        assert np.array_equal(res_np, res_num)
        assert res_num.shape == (5,)

    def test_zero_padding_2d(self):
        """Test zero padding on 2D array."""
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(np_array, ((0, 0), (0, 0)), mode="constant")
        res_num = num.pad(num_array, ((0, 0), (0, 0)), mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_asymmetric_zero_padding(self):
        """Test asymmetric padding with some zeros."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(np_array, (0, 2), mode="constant")
        res_num = num.pad(num_array, (0, 2), mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_single_element_array(self):
        """Test padding a single element array."""
        np_array = np.array([42])
        num_array = num.array([42])

        res_np = np.pad(np_array, 3, mode="constant", constant_values=0)
        res_num = num.pad(num_array, 3, mode="constant", constant_values=0)

        assert np.array_equal(res_np, res_num)

    def test_single_element_edge_mode(self):
        """Test edge mode with single element."""
        np_array = np.array([42])
        num_array = num.array([42])

        res_np = np.pad(np_array, 3, mode="edge")
        res_num = num.pad(num_array, 3, mode="edge")

        assert np.array_equal(res_np, res_num)

    def test_empty_array(self):
        """Test padding an empty array."""
        np_array = np.array([])
        num_array = num.array([])

        res_np = np.pad(np_array, 2, mode="constant", constant_values=5)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=5)

        assert np.array_equal(res_np, res_num)

    def test_empty_array_error(self):
        """Test that non-constant modes fail on empty arrays."""
        np_array = np.array([])
        num_array = num.array([])

        with pytest.raises(ValueError):
            np.pad(np_array, 2, mode="edge")
        with pytest.raises(ValueError):
            num.pad(num_array, 2, mode="edge")

    def test_large_padding_wrap(self):
        """Test wrap mode with padding larger than array size."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        # Padding of 5 is larger than array size of 3
        res_np = np.pad(np_array, 5, mode="wrap")
        res_num = num.pad(num_array, 5, mode="wrap")

        assert np.array_equal(res_np, res_num)

    def test_large_padding_reflect(self):
        """Test reflect mode with padding larger than array size."""
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        # Padding of 8 requires iterative reflection
        res_np = np.pad(np_array, 8, mode="reflect")
        res_num = num.pad(num_array, 8, mode="reflect")

        assert np.array_equal(res_np, res_num)

    def test_large_padding_symmetric(self):
        """Test symmetric mode with padding larger than array size."""
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        # Padding of 10 requires iterative reflection
        res_np = np.pad(np_array, 10, mode="symmetric")
        res_num = num.pad(num_array, 10, mode="symmetric")

        assert np.array_equal(res_np, res_num)

    def test_singleton_dimension_reflect(self):
        """Test reflect mode with singleton dimension (special case)."""
        np_array = np.array([[1]])
        num_array = num.array([[1]])

        # Singleton dimensions use edge mode for reflect
        res_np = np.pad(np_array, 2, mode="reflect")
        res_num = num.pad(num_array, 2, mode="reflect")

        assert np.array_equal(res_np, res_num)

    def test_3d_array(self):
        """Test padding on 3D array."""
        np_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        num_array = num.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        res_np = np.pad(np_array, 1, mode="constant")
        res_num = num.pad(num_array, 1, mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_different_pad_per_axis_3d(self):
        """Test 3D array with different padding per axis."""
        np_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        num_array = num.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        res_np = np.pad(np_array, ((1, 2), (2, 1), (1, 1)), mode="constant")
        res_num = num.pad(num_array, ((1, 2), (2, 1), (1, 1)), mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_very_large_asymmetric_padding(self):
        """Test very asymmetric padding."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(np_array, (0, 10), mode="constant")
        res_num = num.pad(num_array, (0, 10), mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_wrap_2d_different_sizes(self):
        """Test wrap mode on 2D with different padding per axis."""
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        num_array = num.array([[1, 2, 3], [4, 5, 6]])

        res_np = np.pad(np_array, ((1, 2), (2, 3)), mode="wrap")
        res_num = num.pad(num_array, ((1, 2), (2, 3)), mode="wrap")

        assert np.array_equal(res_np, res_num)

    def test_edge_2d_asymmetric(self):
        """Test edge mode with asymmetric padding on 2D."""
        np_array = np.array([[1, 2], [3, 4], [5, 6]])
        num_array = num.array([[1, 2], [3, 4], [5, 6]])

        res_np = np.pad(np_array, ((2, 1), (1, 3)), mode="edge")
        res_num = num.pad(num_array, ((2, 1), (1, 3)), mode="edge")

        assert np.array_equal(res_np, res_num)

    def test_constant_with_different_dtypes_values(self):
        """Test constant padding preserves dtype even with mismatched constant."""
        # Float array with int constant
        np_array = np.array([1.5, 2.5, 3.5])
        num_array = num.array([1.5, 2.5, 3.5])

        res_np = np.pad(np_array, 2, mode="constant", constant_values=5)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=5)

        assert res_np.dtype == res_num.dtype
        assert np.array_equal(res_np, res_num)

    def test_complex_numbers(self):
        """Test padding with complex numbers."""
        np_array = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        num_array = num.array([1 + 2j, 3 + 4j, 5 + 6j])

        res_np = np.pad(np_array, 2, mode="constant", constant_values=0)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=0)

        assert np.array_equal(res_np, res_num)

    def test_boolean_array(self):
        """Test padding boolean arrays."""
        np_array = np.array([True, False, True])
        num_array = num.array([True, False, True])

        res_np = np.pad(np_array, 2, mode="constant", constant_values=False)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=False)

        assert np.array_equal(res_np, res_num)

    def test_pad_width_as_list(self):
        """Test pad_width provided as list."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(np_array, [2, 3], mode="constant")
        res_num = num.pad(num_array, [2, 3], mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_pad_width_as_tuple_of_tuples(self):
        """Test pad_width as nested tuples."""
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(np_array, ((1, 2), (3, 4)), mode="constant")
        res_num = num.pad(num_array, ((1, 2), (3, 4)), mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_constant_values_tuple(self):
        """Test constant mode with different values for before/after."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(np_array, 2, mode="constant", constant_values=(7, 9))
        res_num = num.pad(
            num_array, 2, mode="constant", constant_values=(7, 9)
        )

        assert np.array_equal(res_np, res_num)

    def test_constant_values_per_axis(self):
        """Test constant mode with different values per axis."""
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(
            np_array, 1, mode="constant", constant_values=((1, 2), (3, 4))
        )
        res_num = num.pad(
            num_array, 1, mode="constant", constant_values=((1, 2), (3, 4))
        )

        assert np.array_equal(res_np, res_num)

    def test_mean_mode_all_zeros(self):
        """Test mean mode when array is all zeros."""
        np_array = np.array([0.0, 0.0, 0.0])
        num_array = num.array([0.0, 0.0, 0.0])

        res_np = np.pad(np_array, 2, mode="mean")
        res_num = num.pad(num_array, 2, mode="mean")

        assert np.allclose(res_np, res_num)

    def test_maximum_mode_all_same(self):
        """Test maximum mode when all values are the same."""
        np_array = np.array([5.0, 5.0, 5.0, 5.0])
        num_array = num.array([5.0, 5.0, 5.0, 5.0])

        res_np = np.pad(np_array, 2, mode="maximum")
        res_num = num.pad(num_array, 2, mode="maximum")

        assert np.allclose(res_np, res_num)

    def test_wrap_exact_size(self):
        """Test wrap mode where padding equals array size."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(np_array, 3, mode="wrap")
        res_num = num.pad(num_array, 3, mode="wrap")

        assert np.array_equal(res_np, res_num)

    def test_reflect_minimum_size(self):
        """Test reflect mode with minimum array size."""
        np_array = np.array([1, 2])
        num_array = num.array([1, 2])

        res_np = np.pad(np_array, 1, mode="reflect")
        res_num = num.pad(num_array, 1, mode="reflect")

        assert np.array_equal(res_np, res_num)

    def test_symmetric_minimum_size(self):
        """Test symmetric mode with minimum array size."""
        np_array = np.array([1, 2])
        num_array = num.array([1, 2])

        res_np = np.pad(np_array, 2, mode="symmetric")
        res_num = num.pad(num_array, 2, mode="symmetric")

        assert np.array_equal(res_np, res_num)

    def test_mixed_zero_nonzero_padding(self):
        """Test 2D with zero padding on one axis."""
        np_array = np.array([[1, 2], [3, 4]])
        num_array = num.array([[1, 2], [3, 4]])

        res_np = np.pad(np_array, ((0, 0), (2, 2)), mode="constant")
        res_num = num.pad(num_array, ((0, 0), (2, 2)), mode="constant")

        assert np.array_equal(res_np, res_num)

    def test_edge_single_row(self):
        """Test edge mode on array with single row."""
        np_array = np.array([[1, 2, 3]])
        num_array = num.array([[1, 2, 3]])

        res_np = np.pad(np_array, ((2, 2), (1, 1)), mode="edge")
        res_num = num.pad(num_array, ((2, 2), (1, 1)), mode="edge")

        assert np.array_equal(res_np, res_num)

    def test_wrap_single_column(self):
        """Test wrap mode on array with single column."""
        np_array = np.array([[1], [2], [3]])
        num_array = num.array([[1], [2], [3]])

        res_np = np.pad(np_array, ((1, 1), (2, 2)), mode="wrap")
        res_num = num.pad(num_array, ((1, 1), (2, 2)), mode="wrap")

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("mode", ("constant", "edge", "wrap"))
    def test_very_large_array_shape(self, mode):
        """Test various modes with different array sizes."""
        np_array = np.arange(100).reshape(10, 10)
        num_array = num.arange(100).reshape(10, 10)

        res_np = np.pad(np_array, 2, mode=mode)
        res_num = num.pad(num_array, 2, mode=mode)

        assert np.array_equal(res_np, res_num)

    def test_negative_values_constant(self):
        """Test constant mode with negative constant value."""
        np_array = np.array([1, 2, 3])
        num_array = num.array([1, 2, 3])

        res_np = np.pad(np_array, 2, mode="constant", constant_values=-999)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=-999)

        assert np.array_equal(res_np, res_num)

    def test_float_constant_on_int_array(self):
        """Test float constant value on integer array (should convert)."""
        np_array = np.array([1, 2, 3], dtype=np.int32)
        num_array = num.array([1, 2, 3], dtype=np.int32)

        res_np = np.pad(np_array, 2, mode="constant", constant_values=5.7)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=5.7)

        assert np.array_equal(res_np, res_num)
        assert res_np.dtype == res_num.dtype

    @pytest.mark.parametrize("mode", ("mean", "maximum", "minimum"))
    def test_stat_modes_single_element(self, mode):
        """Test statistical modes with single element array."""
        np_array = np.array([42.0])
        num_array = num.array([42.0])

        res_np = np.pad(np_array, 3, mode=mode)
        res_num = num.pad(num_array, 3, mode=mode)

        assert np.allclose(res_np, res_num)

    def test_empty_mode_values_undefined(self):
        """Test empty mode - only check shape, not values."""
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 3, mode="empty")
        res_num = num.pad(num_array, 3, mode="empty")

        # Only check shape for empty mode
        assert res_np.shape == res_num.shape
        # Check that original values are preserved
        assert np.array_equal(res_np[3:8], res_num[3:8])

    def test_linear_ramp_to_zero(self):
        """Test linear_ramp with end_values=0."""
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        num_array = num.array([1.0, 2.0, 3.0, 4.0, 5.0])

        res_np = np.pad(np_array, 2, mode="linear_ramp", end_values=0)
        res_num = num.pad(num_array, 2, mode="linear_ramp", end_values=0)

        assert np.allclose(res_np, res_num)

    def test_linear_ramp_different_end_values(self):
        """Test linear_ramp with different end values for each side."""
        np_array = np.array([1.0, 2.0, 3.0])
        num_array = num.array([1.0, 2.0, 3.0])

        res_np = np.pad(np_array, 2, mode="linear_ramp", end_values=(-5, 10))
        res_num = num.pad(
            num_array, 2, mode="linear_ramp", end_values=(-5, 10)
        )

        assert np.allclose(res_np, res_num)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint8, np.uint16])
    def test_various_integer_dtypes(self, dtype):
        """Test with various integer dtypes."""
        np_array = np.array([1, 2, 3, 4, 5], dtype=dtype)
        num_array = num.array([1, 2, 3, 4, 5], dtype=dtype)

        res_np = np.pad(np_array, 2, mode="constant")
        res_num = num.pad(num_array, 2, mode="constant")

        assert res_np.dtype == res_num.dtype
        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("reflect_type", ("even", "odd"))
    def test_reflect_type_parameter(self, reflect_type):
        """Test reflect mode with reflect_type parameter."""
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(np_array, 2, mode="reflect", reflect_type=reflect_type)
        res_num = num.pad(
            num_array, 2, mode="reflect", reflect_type=reflect_type
        )

        assert np.array_equal(res_np, res_num)

    @pytest.mark.parametrize("reflect_type", ("even", "odd"))
    def test_symmetric_type_parameter(self, reflect_type):
        """Test symmetric mode with reflect_type parameter."""
        np_array = np.array([1, 2, 3, 4, 5])
        num_array = num.array([1, 2, 3, 4, 5])

        res_np = np.pad(
            np_array, 2, mode="symmetric", reflect_type=reflect_type
        )
        res_num = num.pad(
            num_array, 2, mode="symmetric", reflect_type=reflect_type
        )

        assert np.array_equal(res_np, res_num)

    def test_invalid_pad_mode(self) -> None:
        arr = num.array([[1, 2], [3, 4]])

        # Directly call thunk method to test C++ implementation path
        # Public API validates modes earlier, so we test internal path
        with pytest.raises(ValueError):
            arr._thunk.pad(((1, 1), (1, 1)), mode="reflect")

    def test_pad_edge_color_shape(self) -> None:
        np_arr = np.array([1])
        num_arr = num.array([1])
        np_result = np.pad(np_arr, ((0, 0)), mode="edge")
        num_result = num.pad(num_arr, ((0, 0)), mode="edge")
        assert np.array_equal(np_result, np.array(num_result))

    @pytest.mark.skipif(
        EAGER_TEST, reason="'EagerArray' DID NOT RAISE <class 'ValueError'>"
    )
    def test_pad_constant_no_value(self) -> None:
        arr = num.array([1, 2, 3])
        with pytest.raises(ValueError, match="constant mode requires"):
            # Call thunk method directly to test internal path
            arr._thunk.pad(((1, 1),), mode="constant")


class TestPadDtypes:
    """Test padding with different dtypes."""

    @pytest.mark.parametrize(
        "dtype", (np.int32, np.int64, np.float32, np.float64)
    )
    def test_constant_various_dtypes(self, dtype):
        np_array = np.array([1, 2, 3, 4, 5], dtype=dtype)
        num_array = num.array([1, 2, 3, 4, 5], dtype=dtype)

        res_np = np.pad(np_array, 2, mode="constant", constant_values=0)
        res_num = num.pad(num_array, 2, mode="constant", constant_values=0)

        assert res_np.dtype == res_num.dtype
        assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
