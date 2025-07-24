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
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


class TestRavelMultiIndexErrors:
    def test_none_array(self):
        with pytest.raises(TypeError):
            np.ravel_multi_index((None,), (3,))
        with pytest.raises(TypeError):
            num.ravel_multi_index((None,), (3,))

    def test_indices_wrong_type(self):
        multi_index_np = (np.array([1.0]), np.array([2.0]))
        multi_index_num = (num.array([1.0]), num.array([2.0]))
        shape = (3, 3)

        message = "only int indices permitted"
        with pytest.raises(TypeError, match=message):
            np.ravel_multi_index(multi_index_np, shape)
        with pytest.raises(TypeError, match=message):
            num.ravel_multi_index(multi_index_num, shape)

    def test_invalid_shape(self):
        multi_index_np = (np.array([1]), np.array([2]))
        multi_index_num = (num.array([1]), num.array([2]))
        shape = (3, "3")

        message = "'str' object cannot be interpreted as an integer"
        with pytest.raises(TypeError, match=message):
            np.ravel_multi_index(multi_index_np, shape)
        with pytest.raises(TypeError, match=message):
            num.ravel_multi_index(multi_index_num, shape)

    def test_invalid_index(self):
        multi_index_np = (np.array([-1]), np.array([2]))
        multi_index_num = (num.array([-1]), num.array([2]))
        shape = (3, 3)

        message = "invalid entry in coordinates array"
        with pytest.raises(ValueError, match=message):
            np.ravel_multi_index(multi_index_np, shape)
        with pytest.raises(ValueError, match=message):
            num.ravel_multi_index(multi_index_num, shape)

    def test_index_out_of_bounds(self):
        multi_index_np = (np.array([2]), np.array([2]))
        multi_index_num = (num.array([2]), num.array([2]))
        shape = (2, 2)

        message = "invalid entry in coordinates array"
        with pytest.raises(ValueError, match=message):
            np.ravel_multi_index(multi_index_np, shape)
        with pytest.raises(ValueError, match=message):
            num.ravel_multi_index(multi_index_num, shape)

    def test_empty_shape(self):
        multi_index_np = (np.array([1]),)
        multi_index_num = (num.array([1]),)
        shape = (0,)

        message = r"cannot unravel if shape has zero entries \(is empty\)\."
        with pytest.raises(ValueError, match=message):
            np.ravel_multi_index(multi_index_np, shape)
        with pytest.raises(ValueError, match=message):
            num.ravel_multi_index(multi_index_num, shape)

    def test_empty_indices(self):
        multi_index_np = (np.array([]), np.array([]))
        multi_index_num = (num.array([]), num.array([]))
        shape = (3, 3)

        message = "only int indices permitted"
        with pytest.raises(TypeError, match=message):
            np.ravel_multi_index(multi_index_np, shape)
        with pytest.raises(TypeError, match=message):
            num.ravel_multi_index(multi_index_num, shape)

    def test_wrong_order(self):
        multi_index_np = (np.array([1]), np.array([2]))
        multi_index_num = (num.array([1]), num.array([2]))
        shape = (3, 3)

        message = "only 'C' or 'F' order is permitted"
        with pytest.raises(ValueError, match=message):
            np.ravel_multi_index(multi_index_np, shape, order="K")
        with pytest.raises(ValueError, match=message):
            num.ravel_multi_index(multi_index_num, shape, order="K")


def test_empty_indices():
    shape = (3, 3)
    multi_index = (mk_seq_array(np, 0), mk_seq_array(np, 0))
    multi_index_num = (mk_seq_array(num, 0), mk_seq_array(num, 0))
    res_num = num.ravel_multi_index(multi_index_num, shape)
    res_np = np.ravel_multi_index(multi_index, shape)
    assert np.allclose(res_num, res_np)


def test_large_shape():
    shape = (100, 100, 100)
    multi_index = ([1], [2], [3])
    res_np = np.ravel_multi_index(multi_index, shape)
    res_num = num.ravel_multi_index(multi_index, shape)
    assert np.allclose(res_num, res_np)


def test_large_indices():
    shape = (1000000001,)
    multi_index = ([1000000000],)
    res_np = np.ravel_multi_index(multi_index, shape)
    res_num = num.ravel_multi_index(multi_index, shape)
    assert np.allclose(res_num, res_np)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
@pytest.mark.parametrize("order", ("F", "C"))
def test_basic(ndim, order):
    shape = (6,) * ndim
    size = (6**ndim) % 2
    # Create multi_index with ndim arrays
    multi_index = tuple(mk_seq_array(np, size) for _ in range(ndim))
    multi_index_num = tuple(mk_seq_array(num, size) for _ in range(ndim))

    res_np = np.ravel_multi_index(multi_index, shape, order=order)
    res_num = num.ravel_multi_index(multi_index_num, shape, order=order)
    assert np.allclose(res_num, res_np)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE[:-1])
@pytest.mark.parametrize("order", ("F", "C"))
def test_uneven_shape(ndim, order):
    shape = np.random.randint(1, 6, ndim, dtype=int)
    size = ndim
    # Create multi_index with ndim arrays of valid indices
    multi_index = tuple(mk_seq_array(np, size) % shape[i] for i in range(ndim))
    multi_index_num = tuple(
        mk_seq_array(num, size) % shape[i] for i in range(ndim)
    )

    res_np = np.ravel_multi_index(multi_index, shape, order=order)
    res_num = num.ravel_multi_index(multi_index_num, shape, order=order)
    assert np.allclose(res_num, res_np)


@pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
def test_modes(mode):
    shape = (3, 3)
    multi_index = ([-1, 0, 3], [-1, 0, 3])
    multi_index_num = (num.array([-1, 0, 3]), num.array([-1, 0, 3]))

    if mode == "raise":
        message = "invalid entry in coordinates array"
        with pytest.raises(ValueError, match=message):
            np.ravel_multi_index(multi_index, shape, mode=mode)
        with pytest.raises(ValueError, match=message):
            num.ravel_multi_index(multi_index_num, shape, mode=mode)
    else:
        res_np = np.ravel_multi_index(multi_index, shape, mode=mode)
        res_num = num.ravel_multi_index(multi_index_num, shape, mode=mode)
        assert np.allclose(res_num, res_np)


def test_mixed_modes():
    shape = (3, 3, 3)
    multi_index = ([-1, 0, 3], [1, 1, 1], [-1, 3, 0])
    multi_index_num = tuple(num.array(x) for x in multi_index)

    modes = ("wrap", "raise", "clip")
    res_np = np.ravel_multi_index(multi_index, shape, mode=modes)
    res_num = num.ravel_multi_index(multi_index_num, shape, mode=modes)
    assert np.allclose(res_num, res_np)

    modes = ("clip", "wrap", "wrap")
    res_np = np.ravel_multi_index(multi_index, shape, mode=modes)
    res_num = num.ravel_multi_index(multi_index_num, shape, mode=modes)
    assert np.allclose(res_num, res_np)


def test_scalar_inputs():
    shape = (5, 4, 3)

    # Test with scalar inputs
    res_np = np.ravel_multi_index((2, 1, 0), shape)
    res_num = num.ravel_multi_index((2, 1, 0), shape)
    assert np.allclose(res_num, res_np)

    # Test with single-element arrays
    res_np = np.ravel_multi_index(([2], [1], [0]), shape)
    res_num = num.ravel_multi_index(([2], [1], [0]), shape)
    assert np.allclose(res_num, res_np)

    # Test with 0-D arrays
    multi_index_np = (np.array(2), np.array(1), np.array(0))
    multi_index_num = (num.array(2), num.array(1), num.array(0))
    res_np = np.ravel_multi_index(multi_index_np, shape)
    res_num = num.ravel_multi_index(multi_index_num, shape)
    assert np.allclose(res_num, res_np)


def test_mismatched_input_lengths():
    shape = (3, 3)

    # Wrong number of index arrays vs shape dimensions
    multi_index_wrong_np = (np.array([1]), np.array([0]), np.array([2]))
    multi_index_wrong_num = (num.array([1]), num.array([0]), num.array([2]))

    message = "parameter multi_index must be a sequence of length 2"
    with pytest.raises(ValueError, match=message):
        np.ravel_multi_index(multi_index_wrong_np, shape)
    with pytest.raises(ValueError, match=message):
        num.ravel_multi_index(multi_index_wrong_num, shape)


def test_degenerate_dimensions():
    ndim = np.random.randint(1, 5)
    shape_list = []
    multi_index_list = []
    for i in range(ndim):
        if np.random.random() < 0.5:  # 50% chance of being degenerate
            dim_size = 1
            idx = 0
        else:
            dim_size = np.random.randint(2, 10)
            idx = np.random.randint(0, dim_size - 1)
        shape_list.append(dim_size)
        multi_index_list.append([idx])


def test_non_broadcastable_arrays():
    shape = (3, 3)
    multi_index_np = (np.array([0, 1, 2]), np.array([0, 1]))
    multi_index_num = (num.array([0, 1, 2]), num.array([0, 1]))

    with pytest.raises(ValueError):
        np.ravel_multi_index(multi_index_np, shape)
    with pytest.raises(ValueError):
        num.ravel_multi_index(multi_index_num, shape)


def test_non_integer_shape_dims():
    multi_index_np = (np.array([0]), np.array([1]))
    multi_index_num = (num.array([0]), num.array([1]))
    shape = (2.3, 1.0)

    message = "'float' object cannot be interpreted as an integer"
    with pytest.raises(TypeError, match=message):
        np.ravel_multi_index(multi_index_np, shape)
    with pytest.raises(TypeError, match=message):
        num.ravel_multi_index(multi_index_num, shape)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_different_int_dtypes(dtype):
    shape = (10, 10)
    multi_index_np = (
        np.array([1, 5], dtype=dtype),
        np.array([2, 7], dtype=dtype),
    )
    multi_index_num = (
        num.array([1, 5], dtype=dtype),
        num.array([2, 7], dtype=dtype),
    )

    res_np = np.ravel_multi_index(multi_index_np, shape)
    res_num = num.ravel_multi_index(multi_index_num, shape)
    assert np.allclose(res_num, res_np)

    # Also test with a large value that fits in int64 but not smaller dtypes
    if np.iinfo(dtype).max >= 1000000000:
        large_val = 1000000000
        shape_large = (large_val + 1,)
        multi_index_large_np = (np.array([large_val], dtype=dtype),)
        multi_index_large_num = (num.array([large_val], dtype=dtype),)
        res_np_large = np.ravel_multi_index(multi_index_large_np, shape_large)
        res_num_large = num.ravel_multi_index(
            multi_index_large_num, shape_large
        )
        assert np.allclose(res_num_large, res_np_large)


def test_shape_as_list():
    multi_index_np = (np.array([1]), np.array([2]))
    multi_index_num = (num.array([1]), num.array([2]))
    shape_list = [3, 3]  # Using a list for shape

    res_np = np.ravel_multi_index(multi_index_np, shape_list)
    res_num = num.ravel_multi_index(multi_index_num, shape_list)
    assert np.allclose(res_num, res_np)


def test_empty_shape_and_empty_indices():
    res_np = np.ravel_multi_index((), ())
    res_num = num.ravel_multi_index((), ())
    assert res_num == res_np
    assert res_np == 0


def test_broadcastable_indices():
    shape = (3, 3, 3)
    # Broadcasting ([0], [0, 1], [0]) should result in (2, 2) intermediate
    # shape
    multi_index_np = (np.array([0]), np.array([0, 1]), np.array([0]))
    multi_index_num = (num.array([0]), num.array([0, 1]), num.array([0]))
    res_np = np.ravel_multi_index(multi_index_np, shape)
    res_num = num.ravel_multi_index(multi_index_num, shape)
    assert np.allclose(res_num, res_np)

    # Another broadcasting scenario
    multi_index_np_2 = (np.array([[0, 1]]), np.array([0, 1]).reshape(2, 1))
    multi_index_num_2 = (num.array([[0, 1]]), num.array([0, 1]).reshape(2, 1))
    shape_2 = (2, 2)
    res_np_2 = np.ravel_multi_index(multi_index_np_2, shape_2)
    res_num_2 = num.ravel_multi_index(multi_index_num_2, shape_2)
    assert np.allclose(res_num_2, res_np_2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
