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

import numpy as np
import pytest
from utils.generators import broadcasts_to_along_axis, mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num

N = 10


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_ndim(ndim):
    shape = (N,) * ndim
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)
    shape_idx = (1,) * ndim
    np_indices = mk_seq_array(np, shape_idx) % N
    num_indices = mk_seq_array(num, shape_idx) % N
    for axis in range(-1, ndim):
        res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
        res_num = num.take_along_axis(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)
    np_indices = mk_seq_array(np, (3,))
    num_indices = mk_seq_array(num, (3,))
    res_np = np.take_along_axis(np_arr, np_indices, None)
    res_num = num.take_along_axis(num_arr, num_indices, None)
    assert np.array_equal(res_num, res_np)


@pytest.mark.parametrize(
    "axis", range(-1, 3), ids=lambda axis: f"(axis={axis})"
)
def test_full(axis):
    shape = (3, 4, 5)
    np_arr = mk_seq_array(np, shape)
    num_arr = mk_seq_array(num, shape)

    size = shape[axis]
    axis_values = (0, size - 1, size * 2)

    for shape_idx in broadcasts_to_along_axis(shape, axis, axis_values):
        np_indices = mk_seq_array(np, shape_idx) % shape[axis]
        num_indices = mk_seq_array(num, shape_idx) % shape[axis]
        res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
        res_num = num.take_along_axis(num_arr, num_indices, axis=axis)
        assert np.array_equal(res_num, res_np)


def test_empty_indice():
    np_arr = mk_seq_array(np, (10,))
    num_arr = mk_seq_array(num, (10,))
    np_indices = np.array([], dtype=int)
    num_indices = num.array([], dtype=int)
    res_np = np.take_along_axis(np_arr, np_indices, axis=0)
    res_num = num.take_along_axis(num_arr, num_indices, axis=0)
    assert np.array_equal(res_num, res_np)


def test_broadcasting():
    """Test broadcasting case that should use fancy indexing fallback (no recursion)."""
    np_arr = np.arange(100).reshape(10, 10)
    num_arr = num.array(np_arr)

    np_indices = np.array([[0]])
    num_indices = num.array(np_indices)

    # This should use fancy indexing fallback due to broadcasting
    res_np = np.take_along_axis(np_arr, np_indices, axis=0)
    res_num = num.take_along_axis(num_arr, num_indices, axis=0)

    assert np.array_equal(res_num, res_np)


def test_no_broadcasting():
    """Test non-broadcasting case that should use the optimized TAKE task."""
    np_arr = np.arange(60).reshape(3, 4, 5)
    num_arr = num.array(np_arr)

    np_indices = np.zeros((3, 4, 5), dtype=int)
    num_indices = num.array(np_indices)

    # This should use TAKE task (no broadcasting needed)
    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)


def test_argsort_use_case():
    """Test real-world use case with argsort."""
    np_arr = np.array([[30, 10, 20], [60, 40, 50]])
    num_arr = num.array(np_arr)

    np_indices = np.argsort(np_arr, axis=1)
    num_indices = num.argsort(num_arr, axis=1)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)
    # Verify it's actually sorted
    assert np.array_equal(res_np, [[10, 20, 30], [40, 50, 60]])


def test_4d_boundary():
    """Test 4D array (boundary case for TAKE task optimization)."""
    np.random.seed(42)
    np_arr = np.random.rand(2, 3, 5, 4)
    num_arr = num.array(np_arr)

    np_indices = np.random.randint(0, 5, size=(2, 3, 3, 4))
    num_indices = num.array(np_indices)

    # This should use TAKE task (ndim <= 4, no broadcasting)
    res_np = np.take_along_axis(np_arr, np_indices, axis=2)
    res_num = num.take_along_axis(num_arr, num_indices, axis=2)

    assert np.allclose(res_num, res_np)


def test_5d_fallback():
    """Test 5D array (should fall back to fancy indexing)."""
    np.random.seed(42)
    np_arr = np.random.rand(2, 3, 4, 5, 2)
    num_arr = num.array(np_arr)

    # Indices must match all dimensions except axis dimension
    np_indices = np.random.randint(0, 5, size=(2, 3, 4, 3, 2))
    num_indices = num.array(np_indices)

    # This should use fancy indexing fallback (ndim > 4)
    res_np = np.take_along_axis(np_arr, np_indices, axis=3)
    res_num = num.take_along_axis(num_arr, num_indices, axis=3)

    assert np.allclose(res_num, res_np)


def test_negative_indices():
    """Test that negative indices work correctly."""
    np_arr = np.array([[10, 20, 30], [40, 50, 60]])
    num_arr = num.array(np_arr)

    # Use negative indices (should wrap around)
    np_indices = np.array([[-1, 0, -2], [-1, -2, 0]])
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)


def test_single_element():
    """Test with single element arrays."""
    np_arr = np.array([[5]])
    num_arr = num.array(np_arr)

    np_indices = np.array([[0]])
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=0)
    res_num = num.take_along_axis(num_arr, num_indices, axis=0)

    assert np.array_equal(res_num, res_np)


def test_repeated_indices():
    """Test with repeated indices."""
    np_arr = np.array([[1, 2, 3], [4, 5, 6]])
    num_arr = num.array(np_arr)

    # Repeat index 1 multiple times
    np_indices = np.array([[1, 1, 1, 1], [0, 0, 2, 2]])
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)


def test_all_same_indices():
    """Test with all indices pointing to the same element."""
    np_arr = np.arange(20).reshape(4, 5)
    num_arr = num.array(np_arr)

    # All zeros
    np_indices = np.zeros((4, 3), dtype=int)
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)


def test_boundary_indices():
    """Test with indices at exact boundaries."""
    np_arr = np.arange(30).reshape(5, 6)
    num_arr = num.array(np_arr)

    # First and last indices
    np_indices = np.array(
        [[0, 5, 0], [0, 5, 5], [0, 0, 5], [5, 5, 0], [5, 5, 5]]
    )
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)


def test_single_column_selection():
    """Test selecting a single column/row."""
    np_arr = np.arange(12).reshape(3, 4)
    num_arr = num.array(np_arr)

    # Single column (shape: 3, 1)
    np_indices = np.array([[2], [1], [3]])
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)
    assert res_np.shape == (3, 1)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_different_dtypes(dtype):
    """Test with different data types."""
    np_arr = np.arange(12, dtype=dtype).reshape(3, 4)
    num_arr = num.array(np_arr)

    np_indices = np.array([[0, 2, 1], [3, 1, 0], [2, 3, 1]])
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)
    assert res_num.dtype == res_np.dtype


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_axis_variations(axis):
    """Test different axis values including negative."""
    np_arr = np.arange(24).reshape(2, 3, 4)
    num_arr = num.array(np_arr)

    shape_list = list(np_arr.shape)
    shape_list[axis] = 2  # Select 2 elements along axis
    np_indices = np.zeros(shape_list, dtype=int)
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=axis)
    res_num = num.take_along_axis(num_arr, num_indices, axis=axis)

    assert np.array_equal(res_num, res_np)


def test_axis_none_flattening():
    """Test axis=None with array flattening."""
    np_arr = np.arange(12).reshape(3, 4)
    num_arr = num.array(np_arr)

    # axis=None requires 1D indices
    np_indices = np.array([0, 5, 11, 3, 7])
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=None)
    res_num = num.take_along_axis(num_arr, num_indices, axis=None)

    assert np.array_equal(res_num, res_np)
    assert res_np.ndim == 1


def test_empty_result():
    """Test case that produces empty result."""
    np_arr = np.arange(12).reshape(3, 4)
    num_arr = num.array(np_arr)

    # Empty indices array
    np_indices = np.array([], dtype=int).reshape(3, 0)
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.array_equal(res_num, res_np)
    assert res_np.shape == (3, 0)


def test_large_array():
    """Test with larger array to stress the implementation."""
    np.random.seed(123)
    np_arr = np.random.rand(100, 200)
    num_arr = num.array(np_arr)

    np_indices = np.random.randint(0, 200, size=(100, 50))
    num_indices = num.array(np_indices)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.allclose(res_num, res_np)


def test_argmin_argmax_use_case():
    """Test with argmin/argmax results (common use case)."""
    np_arr = np.array([[3.2, 1.5, 4.8], [2.1, 5.0, 1.2]])
    num_arr = num.array(np_arr)

    # Get min and max indices with keepdims
    np_min_idx = np.argmin(np_arr, axis=1, keepdims=True)
    np_max_idx = np.argmax(np_arr, axis=1, keepdims=True)
    num_min_idx = num.argmin(num_arr, axis=1, keepdims=True)
    num_max_idx = num.argmax(num_arr, axis=1, keepdims=True)

    # Concatenate to get both min and max
    np_indices = np.concatenate([np_min_idx, np_max_idx], axis=1)
    num_indices = num.concatenate([num_min_idx, num_max_idx], axis=1)

    res_np = np.take_along_axis(np_arr, np_indices, axis=1)
    res_num = num.take_along_axis(num_arr, num_indices, axis=1)

    assert np.allclose(res_num, res_np)
    # Verify we got min and max values
    assert np.allclose(res_np[:, 0], np.min(np_arr, axis=1))
    assert np.allclose(res_np[:, 1], np.max(np_arr, axis=1))


def test_int32_indices() -> None:
    np_arr = np.array([[10, 20], [30, 40]])
    num_arr = num.array([[10, 20], [30, 40]])

    np_idx = np.array([[0, 0], [1, 1]], dtype=np.int32)
    num_idx = num.array([[0, 0], [1, 1]], dtype=np.int32)

    np_result = np.take_along_axis(np_arr, np_idx, axis=0)
    num_result = num.take_along_axis(num_arr, num_idx, axis=0)

    assert np.array_equal(np_result, np.array(num_result))


class TestTakeAlongAxisErrors:
    def setup_method(self):
        self.a = num.ones((3, 3))
        self.ai = num.ones((3, 3), dtype=int)

    @pytest.mark.parametrize("dtype", (bool, float), ids=str)
    def test_indices_bad_type(self, dtype):
        ai = num.ones((3, 3), dtype=dtype)
        msg = "`indices` must be an integer array"
        with pytest.raises(TypeError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "shape", ((3, 2), (3, 0)), ids=lambda shape: f"(shape={shape})"
    )
    def test_indices_bad_shape(self, shape):
        # In Numpy, it raises IndexError.
        # In cuPyNumeric, it raises ValueError.
        ai = num.ones(shape, dtype=int)
        msg = "shape mismatch: indexing arrays could not be broadcast"
        with pytest.raises(IndexError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.parametrize(
        "shape", ((1,), (3, 3, 1)), ids=lambda shape: f"(shape={shape})"
    )
    def test_indices_bad_dims(self, shape):
        ai = num.ones(shape, dtype=int)
        msg = "`indices` and `a` must have the same number of dimensions"
        with pytest.raises(ValueError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.parametrize(
        "value", (-4, 3), ids=lambda value: f"(value={value})"
    )
    def test_indices_out_of_bound(self, value):
        ai = num.full((3, 3), value, dtype=int)
        msg = "out of bounds"
        with pytest.raises(IndexError, match=msg):
            num.take_along_axis(self.a, ai, axis=0)

    @pytest.mark.parametrize(
        "axis", (2, -3), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound(self, axis):
        msg = "out of bounds"
        # In Numpy, it raises AxisError
        with pytest.raises(ValueError, match=msg):
            num.take_along_axis(self.a, self.ai, axis=axis)

    def test_axis_float(self):
        axis = 0.0
        msg = "integer argument expected"
        with pytest.raises(TypeError, match=msg):
            num.take_along_axis(self.a, self.ai, axis=axis)

    def test_axis_none_indice_not_1d(self):
        axis = None
        msg = "indices must be 1D if axis=None"
        with pytest.raises(ValueError, match=msg):
            num.take_along_axis(self.a, self.ai, axis=axis)

    def test_a_none(self):
        ai = num.array([1, 1, 1])
        msg = "object has no attribute 'ndim'"
        with pytest.raises(AttributeError, match=msg):
            num.take_along_axis(None, ai, axis=0)

    def test_indice_none(self):
        msg = "'NoneType' object has no attribute 'dtype'"
        with pytest.raises(AttributeError, match=msg):
            num.take_along_axis(self.a, None, axis=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
