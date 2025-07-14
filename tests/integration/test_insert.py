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


# Error tests
@pytest.mark.parametrize("axis", [2, -3])
def test_invalid_axis(axis):
    arr = mk_seq_array(np, (3,))
    arr_num = num.array(arr)

    message = f"axis {axis} is out of bounds for array of dimension 1"
    with pytest.raises(IndexError, match=message):
        num.insert(arr_num, 1, 99, axis=axis)

    with pytest.raises(IndexError, match=message):
        np.insert(arr_num, 1, 99, axis=axis)


def test_bool_index_error():
    arr = mk_seq_array(np, (3,))
    arr_num = num.array(arr)
    mask = np.array([[True, False, True], [True, False, True]])
    mask_num = num.array(mask)

    message = "boolean array argument obj to insert must be one dimensional"
    with pytest.raises(ValueError, match=message):
        np.insert(arr, mask, 99)

    with pytest.raises(ValueError, match=message):
        num.insert(arr_num, mask_num, 99)


@pytest.mark.parametrize(
    "arr_shape, multidim_indices",
    [
        ((3,), mk_seq_array(np, (2, 2))),
        ((4,), mk_seq_array(np, (2, 2))),
        ((2, 2), mk_seq_array(np, (2, 2))),
    ],
)
def test_multidim_index_error(arr_shape, multidim_indices):
    arr = mk_seq_array(np, arr_shape)
    arr_num = num.array(arr)
    multidim_indices = np.array(multidim_indices)
    multidim_indices_num = num.array(multidim_indices)
    message = (
        "index array argument obj to insert must be one dimensional"
        " or scalar"
    )
    with pytest.raises(ValueError, match=message):
        num.insert(arr_num, multidim_indices_num, 99)
    with pytest.raises(ValueError, match=message):
        np.insert(arr_num, multidim_indices, 99)


@pytest.mark.parametrize("idx", [-4, 4])
def test_single_index_out_of_bounds(idx):
    arr = mk_seq_array(np, (3,))
    arr_num = num.array(arr)
    message = f"index {idx} is out of bounds for axis 0 with size 3"
    with pytest.raises(IndexError, match=message):
        num.insert(arr_num, idx, 99)
    with pytest.raises(IndexError, match=message):
        np.insert(arr_num, idx, 99)


# Basic functionality tests


@pytest.mark.parametrize(
    "arr_shape, idx",
    [
        ((1,), -1),
        ((2, 3), -2),
        ((2, 2, 2), -1),
    ],
)
def test_negative_index(arr_shape, idx):
    arr_np = mk_seq_array(np, arr_shape)
    arr_num = num.array(arr_np)

    result_np = np.insert(arr_np, idx, 99)
    result_num = num.insert(arr_num, idx, 99)
    assert allclose(result_np, result_num)


def test_array_value():
    arr_np = mk_seq_array(np, (4,))
    arr_num = num.array(arr_np)
    values = np.array([88, 99])

    result_np = np.insert(arr_np, 2, values)
    result_num = num.insert(arr_num, 2, values)
    assert allclose(result_np, result_num)


def test_with_slice():
    arr_np = mk_seq_array(np, (6,))
    arr_num = num.array(arr_np)
    values = np.array([88, 99])

    # Positive step
    result_np = np.insert(arr_np, slice(2, 4), values)
    result_num = num.insert(arr_num, slice(2, 4), values)
    assert allclose(result_np, result_num)

    # Negative step
    result_np_neg = np.insert(arr_np, slice(1, 0, -1), values)
    result_num_neg = num.insert(arr_num, slice(1, 0, -1), values)
    assert allclose(result_np_neg, result_num_neg)


def test_more_values_than_indices():
    arr_np = mk_seq_array(np, (6,))
    arr_num = num.array(arr_np)
    values = np.array([88, 99, 100])

    with pytest.raises(
        ValueError,
        match=(
            r"shape mismatch: value array of shape \([^)]+\) could not be "
            r"broadcast to indexing result of shape \([^)]+\)"
        ),
    ):
        np.insert(arr_np, slice(2, 4), values)
    with pytest.raises(
        ValueError,
        match=(
            r"Shape did not match along dimension \d+ and the value is not "
            r"equal to 1"
            r"|could not broadcast input array from shape \([^)]+\) into "
            r"shape \([^)]+\)"
            r"|shape mismatch: value array of shape \([^)]+\) could not be "
            r"broadcast to indexing result of shape \([^)]+\)"
        ),
    ):
        num.insert(arr_num, slice(2, 4), values)


@pytest.mark.parametrize(
    "test_slice",
    [
        slice(6, 7),  # Beyond array bounds
        slice(2, 2),  # Empty slice
    ],
)
def test_slice_edge_cases(test_slice):
    arr_np = mk_seq_array(np, (6,))
    arr_num = num.array(arr_np)
    values = np.array([88, 99])

    with pytest.raises(
        ValueError,
        match=(
            r"shape mismatch: value array of shape \([^)]+\) could not be "
            r"broadcast to indexing result of shape \([^)]+\)"
        ),
    ):
        np.insert(arr_np, test_slice, values)
    with pytest.raises(
        ValueError,
        # Different error messages in NumPy and CuPyNumeric
        match=(
            r"Shape did not match along dimension \d+ and the value is not "
            r"equal to 1"
            r"|could not broadcast input array from shape \([^)]+\) into "
            r"shape \([^)]+\)"
            r"|shape mismatch: value array of shape \([^)]+\) could not be "
            r"broadcast to indexing result of shape \([^)]+\)"
        ),
    ):
        num.insert(arr_num, test_slice, values)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_flatten_axis_none(axis):
    arr_np = mk_seq_array(np, (2, 2))
    arr_num = num.array(arr_np)

    result_np = np.insert(arr_np, 2, 99, axis=axis)
    result_num = num.insert(arr_num, 2, 99, axis=axis)
    assert allclose(result_np, result_num)


def test_duplicate_indices():
    arr_np = mk_seq_array(np, (4,))
    arr_num = num.array(arr_np)
    indices = np.array([2, 2])
    values = np.array([88, 99])

    result_np = np.insert(arr_np, indices, values)
    result_num = num.insert(arr_num, indices, values)
    assert allclose(result_np, result_num)


def test_bool_values():
    arr_np = mk_seq_array(np, (3,))
    arr_num = num.array(arr_np)

    result_np = np.insert(arr_np, 1, True)
    result_num = num.insert(arr_num, 1, True)
    assert allclose(result_np, result_num)


# Edge case tests
def test_empty_array():
    arr_np = np.array([])
    arr_num = num.array([])

    result_np = np.insert(arr_np, 0, 99)
    result_num = num.insert(arr_num, 0, 99)
    assert allclose(result_np, result_num)


def test_empty_values():
    arr_np = mk_seq_array(np, (3,))
    arr_num = num.array(arr_np)

    result_np = np.insert(arr_np, [], [])
    result_num = num.insert(arr_num, [], [])
    assert allclose(result_np, result_num)


@pytest.mark.parametrize(
    "arr_shape, idx, axis",
    [
        ((5,), 2, 0),  # 1D, insert in middle
        ((5,), 0, 0),  # 1D, insert at beginning
        ((5,), 5, 0),  # 1D, insert at end
        ((2, 3), 1, 0),  # 2D, axis 0, middle
        ((2, 3), 0, 0),  # 2D, axis 0, beginning
        ((2, 3), 2, 0),  # 2D, axis 0, end
        ((2, 3), 1, 1),  # 2D, axis 1, middle
        ((2, 3), 0, 1),  # 2D, axis 1, beginning
        ((2, 3), 3, 1),  # 2D, axis 1, end
        ((2, 3, 4), 1, 2),  # 3D, axis 2
        ((2, 3, 4), -1, -1),  # 3D, negative axis
        ((2, 3, 4), 0, -3),  # 3D, negative axis for first dim
    ],
)
def test_with_axis_general(arr_shape, idx, axis):
    arr_np = mk_seq_array(np, arr_shape)
    arr_num = num.array(arr_np)

    result_np = np.insert(arr_np, idx, 99, axis=axis)
    result_num = num.insert(arr_num, idx, 99, axis=axis)
    assert allclose(result_np, result_num)


def test_bool_mask():
    arr_np = mk_seq_array(np, (5,))
    arr_num = num.array(arr_np)
    mask = np.array([True, False, True, False, True])
    mask_num = num.array(mask)

    result_np = np.insert(arr_np, mask, 99)
    result_num = num.insert(arr_num, mask_num, 99)
    assert allclose(result_np, result_num)


def test_broadcastable_values():
    arr_np = mk_seq_array(np, (2, 3))
    arr_num = num.array(arr_np)
    values_np = mk_seq_array(np, (1, 3))
    values_num = num.array(values_np)

    result_np = np.insert(arr_np, 1, values_np, axis=0)
    result_num = num.insert(arr_num, 1, values_num, axis=0)
    assert allclose(result_np, result_num)


def test_broadcast_error():
    arr_np = mk_seq_array(np, (2, 3))
    arr_num = num.array(arr_np)
    values_np = mk_seq_array(np, (2, 2))
    values_num = num.array(values_np)

    with pytest.raises(
        ValueError,
        match=(
            r"could not broadcast input array from shape \([^)]+\) "
            r"into shape \([^)]+\)"
        ),
    ):
        np.insert(arr_np, 1, values_np, axis=0)
    with pytest.raises(
        ValueError,
        # Different error messages in NumPy and CuPyNumeric
        match=(
            r"Shape did not match along dimension \d+ and the value is not "
            r"equal to 1"
            r"|could not broadcast input array from shape \([^)]+\) into "
            r"shape \([^)]+\)"
            r"|shape mismatch: value array of shape \([^)]+\) could not be "
            r"broadcast to indexing result of shape \([^)]+\)"
        ),
    ):
        num.insert(arr_num, 1, values_num, axis=0)


def test_int_into_float():
    arr_np = mk_seq_array(np, (3,)).astype(float)
    arr_num = num.array(arr_np)
    values = np.array([100000, 200000], dtype=int)
    result_np = np.insert(arr_np, 1, values)
    result_num = num.insert(arr_num, 1, values)
    assert allclose(result_np, result_num)
    assert result_num.dtype == np.float64


def test_float_into_int():
    arr_np = mk_seq_array(np, (3,)).astype(int)
    arr_num = num.array(arr_np)
    values = np.array([1000000.7, 2000000.9], dtype=float)
    result_np = np.insert(arr_np, 1, values)
    result_num = num.insert(arr_num, 1, values)

    assert allclose(result_np, result_num)
    assert result_num.dtype == np.int64
    assert np.all(result_num == result_num.astype(int))


def test_complex_into_float():
    arr_np = mk_seq_array(np, (3,)).astype(float)
    arr_num = num.array(arr_np)
    values = np.array([1000000 + 2000000j, 3000000 + 4000000j], dtype=complex)

    result_np = np.insert(arr_np, 1, values)
    result_num = num.insert(arr_num, 1, values)
    assert allclose(result_np, result_num)


def test_float_into_complex():
    arr_np = mk_seq_array(np, (3,)).astype(complex)
    arr_num = num.array(arr_np)
    values = np.array([1000000.5, 2000000.5], dtype=float)
    result_np = np.insert(arr_np, 1, values)
    result_num = num.insert(arr_num, 1, values)
    assert allclose(result_np, result_num)
    assert result_num.dtype == np.complex128


def test_read_only_source():
    dst_np = mk_seq_array(np, (3,))
    src_np = mk_seq_array(np, (3,))
    src_np.flags.writeable = False  # Make source read-only

    num_dst = num.array(dst_np)
    num_src = num.array(src_np)
    num_src.flags.writeable = False

    # Should work fine as source is only read from
    np.insert(dst_np, 1, src_np)
    num.insert(num_dst, 1, num_src)
    # Note: insert returns new array, so check results, not in-place
    assert allclose(
        np.insert(dst_np, 1, src_np), num.insert(num_dst, 1, num_src)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
