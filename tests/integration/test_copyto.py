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
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


def test_basic_copy():
    dst = mk_seq_array(np, (5,))
    src = mk_seq_array(np, (5,))
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


def test_broadcasting_scalar():
    dst = mk_seq_array(np, (5,))
    src = np.array(100000)
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


def test_broadcasting_1d_to_2d():
    dst = mk_seq_array(np, (3, 4))
    src = mk_seq_array(np, (4,))
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


@pytest.mark.parametrize(
    "shape,where_shape",
    [
        # 1D case
        ((5,), (5,)),
        # 2D case with broadcasting
        ((3, 4), (3, 1)),
    ],
)
def test_where_condition(shape, where_shape):
    dst = mk_seq_array(np, shape)
    src = mk_seq_array(np, shape) + 10
    num_dst = num.array(dst)
    num_src = num.array(src)

    where = np.ones(where_shape, dtype=bool)
    num_where = num.array(where)

    np.copyto(dst, src, where=where)
    num.copyto(num_dst, num_src, where=num_where)
    assert allclose(dst, num_dst)


@pytest.mark.parametrize("casting", ["safe", "same_kind", "unsafe", "no"])
def test_casting_modes(casting):
    dst = mk_seq_array(np, (5,))
    src = mk_seq_array(np, (5,))
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src, casting=casting)
    num.copyto(num_dst, num_src, casting=casting)
    assert allclose(dst, num_dst)


def test_complex_dtypes():
    dst = np.zeros(3, dtype=np.complex128)
    src = np.array([1000 + 2000j, 3000 + 4000j, 5000 + 6000j])
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_very_large_arrays(ndim):
    shape = (4,) * ndim
    dst = mk_seq_array(np, shape)
    src = mk_seq_array(np, shape) + 100
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


def test_scalar_arrays():
    dst = np.array(40000)
    src = np.array(100000)
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


def test_empty_arrays():
    dst = np.array([])
    src = np.array([])
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


def test_where_boolean():
    dst = mk_seq_array(np, (5,))
    src = mk_seq_array(np, (5,))
    where = np.array([True, False, True, False, True])

    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    np.copyto(dst, src, where=where)
    num.copyto(num_dst, num_src, where=num_where)
    assert allclose(dst, num_dst)


def test_mixed_precision():
    dst = mk_seq_array(np, (3,)).astype(np.float32)
    src = np.array([10000.12349, 20000.98431, 30000.14159], dtype=np.float64)

    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src, casting="same_kind")
    num.copyto(num_dst, num_src, casting="same_kind")
    assert allclose(dst, num_dst)


def test_read_only_destination():
    dst = mk_seq_array(np, (3,))
    dst.flags.writeable = False
    src = mk_seq_array(np, (3,))

    num_dst = num.array(dst)
    num_dst.flags.writeable = False
    num_src = num.array(src)

    with pytest.raises(ValueError):
        np.copyto(dst, src)
    with pytest.raises(ValueError):
        num.copyto(num_dst, num_src)


def test_invalid_casting():
    dst = np.zeros(3, dtype=np.int32)
    src = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    num_dst = num.array(dst)
    num_src = num.array(src)

    with pytest.raises(TypeError, match="Cannot cast array data from dtype"):
        np.copyto(dst, src, casting="safe")
    with pytest.raises(TypeError, match="Cannot cast array data from dtype"):
        num.copyto(num_dst, num_src, casting="safe")


def test_casting_special():
    src = np.array([10000, 20000, 30000])
    dst = np.array([50000.876, 60000.123, 70000.987])

    num_src = num.array(src)
    num_dst = num.array(dst)

    # This should return src in the type of dst
    np.copyto(dst, src, casting="same_kind")
    num.copyto(num_dst, num_src, casting="same_kind")
    assert allclose(dst, num_dst)


def test_no_casting():
    dst = mk_seq_array(np, (3,)).astype(np.float32)
    src = mk_seq_array(np, (3,)).astype(np.int32)

    num_dst = num.array(dst)
    num_src = num.array(src)

    message = r"Cannot cast array data from dtype"
    r"\('int32'\) to dtype\('float32'\) "
    r"according to the rule 'no'"

    with pytest.raises(TypeError, match=message):
        np.copyto(dst, src, casting="no")
    with pytest.raises(TypeError, match=message):
        num.copyto(num_dst, num_src, casting="no")


def test_incompatible_shapes():
    dst = mk_seq_array(np, (3, 4))
    src = mk_seq_array(np, (2, 5))

    num_dst = num.array(dst)
    num_src = num.array(src)

    message = (
        r"could not broadcast input array from shape "
        r"\(2,\s*5\) into shape \(3,\s*4\)"
    )

    with pytest.raises(ValueError, match=message):
        np.copyto(dst, src)
    with pytest.raises(ValueError, match=message):
        num.copyto(num_dst, num_src)


def test_where_shape_mismatch():
    dst = mk_seq_array(np, (3, 4))
    src = mk_seq_array(np, (3, 4))
    where = np.array([True, False])

    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    with pytest.raises(ValueError):
        np.copyto(dst, src, where=where)
    with pytest.raises(ValueError):
        num.copyto(num_dst, num_src, where=num_where)


def test_overlapping_memory():
    # Test 1: Simple overlap within the same array
    arr_np = mk_seq_array(np, (10,))
    arr_num = num.array(arr_np)

    # Copy a slice to another slice within the same array
    np.copyto(arr_np[3:7], arr_np[0:4])
    num.copyto(arr_num[3:7], arr_num[0:4])
    assert allclose(arr_np, arr_num)

    # Test 2: Overlap with different views/strides
    arr_np_orig = mk_seq_array(np, (5, 5))
    arr_num_orig = num.array(arr_np_orig)

    # Create overlapping views (e.g., column from one array to row of
    # another view)
    dst_np = arr_np_orig[0, :]  # first row
    src_np = arr_np_orig[:, 0]  # first column
    dst_num = arr_num_orig[0, :]
    src_num = arr_num_orig[:, 0]

    np.copyto(dst_np, src_np)
    num.copyto(dst_num, src_num)
    assert allclose(dst_np, dst_num)  # Check the copied slice
    assert allclose(
        arr_np_orig, arr_num_orig
    )  # Check the whole array state after copy


def test_nan_inf_with_where():
    dst = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    src = np.array([np.nan, 4.0, np.inf], dtype=np.float32)
    where = np.array([True, False, True], dtype=bool)

    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    np.copyto(dst, src, where=where)
    num.copyto(num_dst, num_src, where=num_where)
    assert np.allclose(dst, num_dst, equal_nan=True)


def test_no_casting_valid():
    dst = mk_seq_array(np, (3,)).astype(np.float64)
    src = mk_seq_array(np, (3,)).astype(np.float64)
    num_dst = num.array(dst)
    num_src = num.array(src)
    np.copyto(dst, src, casting="no")
    num.copyto(num_dst, num_src, casting="no")
    assert allclose(dst, num_dst)


def test_same_kind_different_size_int():
    # large int64 value into int32 (will overflow/wrap for same_kind)
    src_val = 2**31  # This will cause overflow for int32
    src_np = np.array([src_val], dtype=np.int64)
    dst_np = np.zeros(1, dtype=np.int32)
    num_src = num.array(src_np)
    num_dst = num.array(dst_np)

    # same_kind should allow this, but the value will change due to overflow
    np.copyto(dst_np, src_np, casting="same_kind")
    num.copyto(num_dst, num_src, casting="same_kind")
    assert allclose(dst_np, num_dst)
    # Check that the overflow indeed happened for both
    assert dst_np[0] != src_val


@pytest.mark.parametrize(
    "dst_shape, src_shape",
    [
        ((2, 3, 4), (4,)),  # 1D src to higher dim dst
        ((2, 3, 4), (1, 4)),  # 2D src to higher dim dst with singleton
        ((2, 3, 4), (3, 1)),  # Another 2D src with singleton
        ((2, 3, 4), (2, 1, 4)),  # Matching dims, inner singleton
        ((2, 3, 4), (1, 3, 1)),  # Multiple singletons
        ((5,), (1,)),  # Scalar-like broadcast from 1-element array
    ],
)
def test_complex_broadcasting(dst_shape, src_shape):
    dst = mk_seq_array(np, dst_shape)
    src = mk_seq_array(np, src_shape)
    num_dst = num.array(dst)
    num_src = num.array(src)

    np.copyto(dst, src)
    num.copyto(num_dst, num_src)
    assert allclose(dst, num_dst)


def test_where_all_false():
    dst = mk_seq_array(np, (5,))
    src = mk_seq_array(np, (5,))
    where = np.zeros((5,), dtype=bool)
    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    np.copyto(dst, src, where=where)
    num.copyto(num_dst, num_src, where=num_where)
    # Nothing should be copied, so dst should remain unchanged
    assert allclose(dst, num_dst)
    assert allclose(dst, mk_seq_array(np, (5,)))


def test_where_empty_mask():
    dst = np.array([], dtype=np.float32)
    src = np.array([], dtype=np.float32)
    where = np.array([], dtype=bool)
    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    np.copyto(dst, src, where=where)
    num.copyto(num_dst, num_src, where=num_where)
    assert allclose(dst, num_dst)
    assert dst.size == 0 and num_dst.size == 0


def test_where_non_bool_mask():
    dst = mk_seq_array(np, (5,))
    src = mk_seq_array(np, (5,))
    where = np.array([1, 0, 1, 0, 1])
    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    # NumPy has a bug with error messages, where it says "safe" no matter what
    # casting is used
    with pytest.raises(TypeError):
        np.copyto(dst, src, where=where)
    with pytest.raises(TypeError):
        num.copyto(num_dst, num_src, where=num_where)


def test_where_non_bool_mask_unsafe():
    dst = mk_seq_array(np, (5,))
    src = mk_seq_array(np, (5,))
    where = np.array([1, 0, 1, 0, 1])
    num_dst = num.array(dst)
    num_src = num.array(src)
    num_where = num.array(where)

    # NumPy has a bug with error messages, where it says "safe" no matter what
    # casting is used
    with pytest.raises(TypeError):
        np.copyto(dst, src, where=where, casting="unsafe")
    with pytest.raises(TypeError):
        num.copyto(num_dst, num_src, where=num_where, casting="unsafe")


@pytest.mark.parametrize(
    "dtype1, dtype2", [(np.int8, np.float32), (np.float64, np.float32)]
)
def test_unsafe_casting_high_to_low_precision(dtype1, dtype2):
    src_dtype1 = mk_seq_array(np, (5,)).astype(dtype1)
    dst_dtype2 = np.zeros(5, dtype=dtype2)
    num_src_dtype1 = num.array(src_dtype1)
    num_dst_dtype2 = num.array(dst_dtype2)

    np.copyto(dst_dtype2, src_dtype1, casting="unsafe")
    num.copyto(num_dst_dtype2, num_src_dtype1, casting="unsafe")
    assert allclose(dst_dtype2, num_dst_dtype2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
