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


@pytest.mark.parametrize("density", (False, True))
def test_histogramdd_weights(density):
    eps = 1.0e-8

    coords_array = np.ndarray(
        shape=(5, 3),
        buffer=np.array(
            [
                2.0,
                10.0,
                3.1,
                4.5,
                6.2,
                5.9,
                7.15,
                6.0,
                8.3,
                9.1,
                2.7,
                8.7,
                7.2,
                6.85,
                3.5,
            ]
        ),
        dtype=np.dtype(np.float64),
    )

    bin_x = np.array([1.9, 3.5, 7.0, 11.0])
    bin_y = np.array([2.0, 3.1, 4.3, 6.1, 7.7])
    bin_z = np.array([1.0, 3.0, 12.0])

    weights = np.array(
        [55.0, 34.0, 25.7, 77.5, 89.2], dtype=np.dtype(np.float64)
    )

    np_out, np_bins_out = np.histogramdd(
        coords_array,
        bins=[bin_x, bin_y, bin_z],
        weights=weights,
        density=density,
    )
    num_out, num_bins_out = num.histogramdd(
        coords_array,
        bins=[bin_x, bin_y, bin_z],
        weights=weights,
        density=density,
    )

    assert allclose(np_out, num_out, atol=eps)
    #
    # need to loop, because the bins arrays are of different sizes
    # hence .shape attribute cannot exist:
    #
    for np_bin, num_bin in zip(np_bins_out, num_bins_out):
        assert allclose(np_bin, num_bin, atol=eps)


@pytest.mark.parametrize("density", (False, True))
def test_histogramdd_weights_int_bins(density):
    eps = 1.0e-8

    coords_array = np.ndarray(
        shape=(5, 3),
        buffer=np.array(
            [
                2.0,
                10.0,
                3.1,
                4.5,
                6.2,
                5.9,
                7.15,
                6.0,
                8.3,
                9.1,
                2.7,
                8.7,
                7.2,
                6.85,
                3.5,
            ]
        ),
        dtype=np.dtype(np.float64),
    )

    bin_x = 5
    bin_y = 4
    bin_z = 3

    weights = np.array(
        [55.0, 34.0, 25.7, 77.5, 89.2], dtype=np.dtype(np.float64)
    )

    np_out, np_bins_out = np.histogramdd(
        coords_array,
        bins=[bin_x, bin_y, bin_z],
        weights=weights,
        density=density,
    )
    num_out, num_bins_out = num.histogramdd(
        coords_array,
        bins=[bin_x, bin_y, bin_z],
        weights=weights,
        density=density,
    )

    assert allclose(np_out, num_out, atol=eps)
    #
    # need to loop, because the bins arrays are of different sizes
    # hence .shape attribute cannot exist:
    #
    for np_bin, num_bin in zip(np_bins_out, num_bins_out):
        assert allclose(np_bin, num_bin, atol=eps)


@pytest.mark.parametrize("density", (False, True))
def test_histogramdd_weights_bins_ranges(density):
    eps = 1.0e-8

    coords_array = np.ndarray(
        shape=(5, 3),
        buffer=np.array(
            [
                2.0,
                10.0,
                3.1,
                4.5,
                6.2,
                5.9,
                7.15,
                6.0,
                8.3,
                9.1,
                2.7,
                8.7,
                7.2,
                6.85,
                3.5,
            ]
        ),
        dtype=np.dtype(np.float64),
    )

    bin_x = 5
    bin_y = 4
    bin_z = 3

    range_x = [3.0, 7.9]
    range_y = [4.2, 9.5]
    range_z = [4.0, 8.3]

    weights = np.array(
        [55.0, 34.0, 25.7, 77.5, 89.2], dtype=np.dtype(np.float64)
    )

    np_out, np_bins_out = np.histogramdd(
        coords_array,
        bins=[bin_x, bin_y, bin_z],
        range=[range_x, range_y, range_z],
        weights=weights,
        density=density,
    )
    num_out, num_bins_out = num.histogramdd(
        coords_array,
        bins=[bin_x, bin_y, bin_z],
        range=[range_x, range_y, range_z],
        weights=weights,
        density=density,
    )

    assert allclose(np_out, num_out, atol=eps)
    #
    # need to loop, because the bins arrays are of different sizes
    # hence .shape attribute cannot exist:
    #
    for np_bin, num_bin in zip(np_bins_out, num_bins_out):
        assert allclose(np_bin, num_bin, atol=eps)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
