# Copyright 2022 NVIDIA Corporation
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

import warnings

import numpy as np
import pytest
from utils.comparisons import allclose

import cupynumeric as num

ALL_METHODS = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize("axes", (0, 1, (0, 1), (0,)))
@pytest.mark.parametrize(
    "qin_arr", (0.5, [0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49, 0.5])
)
@pytest.mark.parametrize("keepdims", (False,))
@pytest.mark.parametrize("overwrite_input", (False,))
def test_multi_axes(str_method, axes, qin_arr, keepdims, overwrite_input):
    eps = 1.0e-8
    arr = np.asarray(
        [
            [1.0, 2, 2, 40],
            [1, np.nan, 2, 1],
            [0, 10, np.nan, np.nan],
            [np.nan, np.nan, 0, 10],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    if num.isscalar(qin_arr):
        qs_arr = qin_arr
    else:
        qs_arr = np.array(qin_arr)

    # cupynumeric:
    # print("cupynumeric axis = %d:"%(axis))
    q_out = num.nanquantile(
        arr,
        qs_arr,
        axis=axes,
        method=str_method,
        keepdims=keepdims,
        overwrite_input=overwrite_input,
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    with warnings.catch_warnings():
        # NumPy may warn about all-NaN slices, just ignore that.
        warnings.simplefilter("ignore", RuntimeWarning)

        np_q_out = np.nanquantile(
            arr,
            qs_arr,
            axis=axes,
            method=str_method,
            keepdims=keepdims,
            overwrite_input=overwrite_input,
        )
    # print(np_q_out)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps, equal_nan=True)


@pytest.mark.parametrize("axes", (1, None))
def test_keepdims(axes: int | None) -> None:
    arr = np.asarray([[10.0, np.nan, 4.0], [4.0, 2.0, 2]])
    np_out = np.nanquantile(arr, 0.5, axis=axes, keepdims=True)
    num_out = num.nanquantile(arr, 0.5, axis=axes, keepdims=True)
    assert np.array_equal(np_out, num_out)


def test_complex_error() -> None:
    arr = np.array([1 + 2j, np.nan + 0j, 5 + 6j])
    msg_np = r"a must be an array of real numbers"
    with pytest.raises(TypeError, match=msg_np):
        np.nanquantile(arr, 0.5)

    msg_num = r"input array cannot be of complex type"
    with pytest.raises(TypeError, match=msg_num):
        num.nanquantile(arr, 0.5)


def test_output_shape() -> None:
    arr = np.array([[10.0, np.nan, 4.0], [4.0, 2.0, 2]])
    out_np = np.zeros(2)
    np.nanquantile(arr, 0.5, axis=1, out=out_np)

    out_num = num.zeros(2)
    num.nanquantile(arr, 0.5, axis=1, out=out_num)
    assert np.array_equal(out_np, out_num)


def test_wrong_output_shape() -> None:
    arr = np.array([[10.0, np.nan, 4.0], [4.0, 2.0, 2]])
    out_num = num.zeros((2, 2))
    msg = r"wrong shape on output array"
    with pytest.raises(ValueError, match=msg):
        num.nanquantile(arr, 0.5, axis=1, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
