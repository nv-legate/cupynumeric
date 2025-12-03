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

import cupynumeric as num


def test_histogram2d_no_weights():
    eps = 1.0e-8

    # no border points:
    #
    a1x = np.array([1, 3, 4, 1, 3, 1, 3, 4, 1, 3, 4, 1, 3, 4], dtype=int)
    a1y = np.array([2, 2, 2, 4, 4, 5, 5, 5, 6, 6, 6, 8, 8, 8], dtype=int)
    bin2x = np.array([0, 2, 5], dtype=int)
    bin2y = np.array([1, 3, 7, 9], dtype=int)

    np_out, np_xb, np_yb = np.histogram2d(a1x, a1y, bins=[bin2x, bin2y])

    num_out, num_xb, num_yb = num.histogram2d(a1x, a1y, bins=[bin2x, bin2y])
    assert allclose(np_out, num_out, atol=eps)

    assert allclose(np_xb, num_xb, atol=eps)
    assert allclose(np_yb, num_yb, atol=eps)


def test_histogram2d_no_weights_border():
    eps = 1.0e-8

    # border points:
    #
    a1x = np.array(
        [1, 3, 4, 5, 3, 4, 5, 1, 3, 1, 3, 4, 1, 3, 4, 1, 1, 3, 4, 5, 2],
        dtype=int,
    )
    a1y = np.array(
        [2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8, 8, 9],
        dtype=int,
    )
    bin2x = np.array([0, 2, 5], dtype=int)
    bin2y = np.array([1, 3, 7, 9], dtype=int)

    np_out, np_xb, np_yb = np.histogram2d(a1x, a1y, bins=[bin2x, bin2y])

    num_out, num_xb, num_yb = num.histogram2d(a1x, a1y, bins=[bin2x, bin2y])
    assert allclose(np_out, num_out, atol=eps)

    assert allclose(np_xb, num_xb, atol=eps)
    assert allclose(np_yb, num_yb, atol=eps)


@pytest.mark.parametrize(
    "bin_x, bin_y",
    [
        (5, np.array([2.0, 4.0, 6.0, 8.0], dtype=float)),
        (np.array([1.0, 2.5, 4.0, 5.0], dtype=float), 6),
    ],
    ids=["int_array", "array_int"],
)
def test_histogram2d_mixed_bins(
    bin_x: int | np.ndarray, bin_y: int | np.ndarray
) -> None:
    eps = 1.0e-8

    a1x = np.array([1.5, 2.3, 3.1, 4.2, 1.8, 3.5, 4.0], dtype=float)
    a1y = np.array([2.1, 3.5, 4.2, 5.8, 2.9, 4.8, 6.1], dtype=float)

    bins = (bin_x, bin_y)

    np_out, np_xb, np_yb = np.histogram2d(a1x, a1y, bins=bins)
    num_out, num_xb, num_yb = num.histogram2d(a1x, a1y, bins=bins)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_xb, num_xb, atol=eps)
    assert allclose(np_yb, num_yb, atol=eps)


@pytest.mark.parametrize(
    "bin_x, bin_y, range_x, range_y",
    [
        (4, 5, [1.0, 5.0], [2.0, 8.0]),
        (
            5,
            np.array([2.0, 4.0, 6.0, 8.0], dtype=float),
            [1.0, 5.5],
            [2.0, 8.0],
        ),
        (
            np.array([1.0, 2.5, 4.0, 5.5], dtype=float),
            6,
            [1.0, 5.5],
            [2.0, 8.0],
        ),
    ],
    ids=["int_int", "int_array", "array_int"],
)
def test_histogram2d_with_range(
    bin_x: int | np.ndarray,
    bin_y: int | np.ndarray,
    range_x: list[float],
    range_y: list[float],
) -> None:
    eps = 1.0e-8

    a1x = np.array([1.5, 2.3, 3.1, 4.2, 1.8, 3.5, 4.0, 5.2], dtype=float)
    a1y = np.array([2.1, 3.5, 4.2, 5.8, 2.9, 4.8, 6.1, 7.3], dtype=float)

    bins = (bin_x, bin_y)
    range_param = [range_x, range_y]

    np_out, np_xb, np_yb = np.histogram2d(
        a1x, a1y, bins=bins, range=range_param
    )
    num_out, num_xb, num_yb = num.histogram2d(
        a1x, a1y, bins=bins, range=range_param
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_xb, num_xb, atol=eps)
    assert allclose(np_yb, num_yb, atol=eps)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
