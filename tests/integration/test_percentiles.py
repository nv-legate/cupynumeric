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
@pytest.mark.parametrize("axes", (0, 1, (0, 1), (0, 2)))
@pytest.mark.parametrize(
    "qin_arr", (50.0, [0.1, 37.0, 42.0, 67.0, 83.0, 100.0, 49.0, 50.0, 0.0])
)
@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize("overwrite_input", (False, True))
def test_multi_axes(str_method, axes, qin_arr, keepdims, overwrite_input):
    eps = 1.0e-8
    arr = np.ndarray(
        shape=(2, 3, 4),
        buffer=np.array(
            [
                1,
                2,
                2,
                40,
                1,
                1,
                2,
                1,
                0,
                10,
                3,
                3,
                40,
                15,
                3,
                7,
                5,
                4,
                7,
                3,
                5,
                1,
                0,
                9,
            ]
        ),
        dtype=int,
    )

    if num.isscalar(qin_arr):
        qs_arr = qin_arr
    else:
        qs_arr = np.array(qin_arr)

    # cupynumeric:
    # print("cupynumeric axis = %d:"%(axis))
    q_out = num.percentile(
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
    np_q_out = np.percentile(
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

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.skip(reason="simplified for debugging, only...")
@pytest.mark.parametrize("str_method", ALL_METHODS)
def test_random_simple(str_method):
    eps = 1.0e-8
    arr = np.random.random((3, 4, 5))

    qin_arr = [0.1, 37.0, 42.0, 67.0, 83.0, 100.0, 49.0, 50.0, 0.0]
    axes = 0

    q_out = num.percentile(arr, qin_arr, method=str_method, axis=axes)
    np_q_out = np.percentile(arr, qin_arr, method=str_method, axis=axes)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize(
    "qin_arr", (50.0, [0.1, 37.0, 42.0, 67.0, 83.0, 100.0, 49.0, 50.0, 0.0])
)
@pytest.mark.parametrize("axes", (None, 0))
def test_random_inlined(str_method, qin_arr, axes):
    eps = 1.0e-8
    arr = np.random.random((3, 4, 5))

    qin_arr = [0.1, 37.0, 42.0, 67.0, 83.0, 100.0, 49.0, 50.0, 0.0]
    axes = 0

    q_out = num.percentile(arr, qin_arr, method=str_method, axis=axes)
    np_q_out = np.percentile(arr, qin_arr, method=str_method, axis=axes)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
