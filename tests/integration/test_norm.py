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
from utils.comparisons import allclose
from utils.generators import mk_0to1_array
from utils.utils import MAX_DIM_RANGE, ONE_MAX_DIM_RANGE, TWO_MAX_DIM_RANGE

import cupynumeric as num

VECTOR_ORDS = [None, np.inf, -np.inf, 0, 1, -1, 2, -2]

# TODO: Add "nuc", 2, -2 once they are implemented
MATRIX_ORDS = [None, "fro", np.inf, -np.inf, 1, -1]

np_arrays = [mk_0to1_array(np, (3,) * ndim) - 0.5 for ndim in MAX_DIM_RANGE]
num_arrays = [mk_0to1_array(num, (3,) * ndim) - 0.5 for ndim in MAX_DIM_RANGE]


@pytest.mark.parametrize("ord", VECTOR_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize(
    "dtype", (np.float64, pytest.param(np.complex64, marks=pytest.mark.xfail))
)
def test_noaxis_1d(ord, keepdims, dtype):
    # for ord=0, dtype is np.complex64
    # Numpy output array is float32
    # cuPyNumeric output array is complex64
    np_res = np.linalg.norm(
        np_arrays[1].astype(dtype), ord=ord, keepdims=keepdims
    )
    num_res = num.linalg.norm(
        num_arrays[1].astype(dtype), ord=ord, keepdims=keepdims
    )
    assert allclose(np_res, num_res)


@pytest.mark.parametrize("ord", MATRIX_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", (np.float64, np.complex64))
def test_noaxis_2d(ord, keepdims, dtype):
    np_res = np.linalg.norm(
        np_arrays[2].astype(dtype), ord=ord, keepdims=keepdims
    )
    num_res = num.linalg.norm(
        num_arrays[2].astype(dtype), ord=ord, keepdims=keepdims
    )
    assert allclose(np_res, num_res)


@pytest.mark.parametrize("ndim", MAX_DIM_RANGE[:-1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", (np.float64, np.complex64))
def test_noaxis_other(ndim, keepdims, dtype):
    np_res = np.linalg.norm(np_arrays[ndim].astype(dtype), keepdims=keepdims)
    num_res = num.linalg.norm(
        num_arrays[ndim].astype(dtype), keepdims=keepdims
    )
    assert allclose(np_res, num_res)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE[:-1])
@pytest.mark.parametrize("ord", VECTOR_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
def test_axis_1d(ndim, ord, keepdims):
    np_res = np.linalg.norm(
        np_arrays[ndim], ord=ord, axis=0, keepdims=keepdims
    )
    num_res = num.linalg.norm(
        num_arrays[ndim], ord=ord, axis=0, keepdims=keepdims
    )
    assert allclose(np_res, num_res)


@pytest.mark.parametrize("ndim", TWO_MAX_DIM_RANGE[:-1])
@pytest.mark.parametrize("ord", MATRIX_ORDS)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize(
    "axis",
    ((0, 1), pytest.param((1, 0), marks=pytest.mark.xfail)),
    ids=lambda axis: f"(axis={axis})",
)
def test_axis_2d(ndim, ord, keepdims, axis):
    # For all cases when axis is (1, 0) and ord is None or fro,
    # output values of cuPyNumeric and Numpy are different and not close enough
    np_res = np.linalg.norm(
        np_arrays[ndim], ord=ord, axis=axis, keepdims=keepdims
    )
    num_res = num.linalg.norm(
        num_arrays[ndim], ord=ord, axis=axis, keepdims=keepdims
    )
    assert allclose(np_res, num_res)


class TestNormErrors:
    def test_axis_invalid_type(self):
        # In cuPyNumeric, raises error in normalize_axis_tuple
        expected_exc = TypeError
        x_np = np.array([1, 2, 3])
        x_num = num.array([1, 2, 3])
        axis = "string"

        with pytest.raises(expected_exc):
            np.linalg.norm(x_np, axis=axis)

        with pytest.raises(expected_exc):
            num.linalg.norm(x_num, axis=axis)

    @pytest.mark.parametrize(
        "axis",
        (3, -4, (1, 1), (1, 3), (1, 0, 2)),
        ids=lambda axis: f"(axis={axis})",
    )
    def test_axis_invalid_value(self, axis):
        # for (1, 1), In cuPyNumeric, raises error in normalize_axis_tuple
        expected_exc = ValueError
        ndim = 2

        with pytest.raises(expected_exc):
            np.linalg.norm(np_arrays[ndim], axis=axis)

        with pytest.raises(expected_exc):
            num.linalg.norm(num_arrays[ndim], axis=axis)

    def test_axis_out_of_bounds(self):
        # raise ValueError("Improper number of dimensions to norm")
        expected_exc = ValueError
        ndim = 3

        with pytest.raises(expected_exc):
            np.linalg.norm(np_arrays[ndim], ord=1)

        with pytest.raises(expected_exc):
            num.linalg.norm(num_arrays[ndim], ord=1)

    @pytest.mark.parametrize(
        "ndim_axis",
        ((1, None), (2, 0)),
        ids=lambda ndim_axis: f"(ndim_axis={ndim_axis})",
    )
    def test_invalid_ord_for_vector(self, ndim_axis):
        expected_exc = ValueError
        ndim, axis = ndim_axis
        ord = "fro"

        with pytest.raises(expected_exc):
            np.linalg.norm(np_arrays[ndim], ord=ord, axis=axis)

        with pytest.raises(expected_exc):
            num.linalg.norm(num_arrays[ndim], ord=ord, axis=axis)

    def test_invalid_ord_for_matrices(self):
        expected_exc = ValueError
        ndim = 2
        axis = (0, 1)
        ord = "unknown"

        with pytest.raises(expected_exc):
            np.linalg.norm(np_arrays[ndim], ord=ord, axis=axis)

        with pytest.raises(expected_exc):
            num.linalg.norm(num_arrays[ndim], ord=ord, axis=axis)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
