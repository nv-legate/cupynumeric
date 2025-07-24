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
@pytest.mark.parametrize("axes", (-1, -2))
def test_quantiles_negative_axes(str_method, axes):
    keepdims = False
    qs_arr = 0.5
    eps = 1.0e-8
    original_shape = (2, 3, 4)
    arr = np.ndarray(
        shape=original_shape,
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
        dtype=float,
    )

    # cupynumeric:
    # print("cupynumeric axis = %d:"%(axis))
    num_q_out = num.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    # due to numpy bug https://github.com/numpy/numpy/issues/22544
    # out = <not-None> fails with keepdims = True
    #
    np_q_out = np.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(np_q_out)

    assert num_q_out.shape == np_q_out.shape
    assert num_q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, num_q_out, atol=eps)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)

    sys.exit(pytest.main(sys.argv))
