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

    print("numpy:\n%s\n" % (str(np_out)))
    print("cunumeric:\n%s\n" % (str(num_out)))
    assert allclose(np_out, num_out, atol=eps)
    #
    # need to loop, because the bins arrays are of different sizes
    # hence .shape attribute cannot exist:
    #
    for np_bin, num_bin in zip(np_bins_out, num_bins_out):
        assert allclose(np_bin, num_bin, atol=eps)


def test_histogramdd_no_weights():
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

    bin_x = np.array([0.0, 3.5, 9.0])
    bin_y = np.array([2.0, 3.1, 4.3, 6.1, 7.7])
    bin_z = np.array([1.0, 3.0, 5.5, 11.0])

    np_out, np_bins_out = np.histogramdd(
        coords_array, bins=[bin_x, bin_y, bin_z]
    )
    num_out, num_bins_out = num.histogramdd(
        coords_array, bins=[bin_x, bin_y, bin_z]
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


@pytest.mark.parametrize("density", (False, True))
@pytest.mark.parametrize("dim", (2, 3, 4))
@pytest.mark.parametrize("limits", ((1.0, 9.0), (11.0, 12.0)))
def test_histogramdd_points_outside_of_range(density, dim, limits):
    points_1d = np.linspace(0.0, 10.0, 10)
    points_grid = np.meshgrid(*((points_1d,) * dim))
    np_points = np.array([a.flatten() for a in points_grid]).T
    edges_1d = np.linspace(limits[0], limits[1], 5)
    num_points = num.array(np_points)
    all_edges = (edges_1d,) * dim
    np_hist, np_bins = np.histogramdd(np_points, all_edges, density=density)
    num_hist, num_bins = num.histogramdd(
        num_points, all_edges, density=density
    )
    np_all_nan = np.all(np.isnan(np_hist))
    num_all_nan = num.all(num.isnan(num_hist))
    assert (np_all_nan and num_all_nan) or allclose(np_hist, num_hist)


def test_histogramdd_single_int_bins() -> None:
    eps = 1.0e-8

    coords_array = np.array(
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    )

    bins = 5

    np_out, np_bins_out = np.histogramdd(coords_array, bins=bins)
    num_out, num_bins_out = num.histogramdd(coords_array, bins=bins)

    assert allclose(np_out, num_out, atol=eps)
    for np_bin, num_bin in zip(np_bins_out, num_bins_out):
        assert allclose(np_bin, num_bin, atol=eps)


def test_histogramdd_non_monotonic_bins() -> None:
    coords_array = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])

    bin_x_np = np.array([1.0, 5.0, 3.0, 10.0])
    bin_y_np = np.array([2.0, 4.0, 6.0, 8.0])

    bin_x_num = num.array([1.0, 5.0, 3.0, 10.0])
    bin_y_num = num.array([2.0, 4.0, 6.0, 8.0])

    msg = r"monotonically"
    with pytest.raises(ValueError, match=msg):
        np.histogramdd(coords_array, bins=[bin_x_np, bin_y_np])

    with pytest.raises(ValueError, match=msg):
        num.histogramdd(num.array(coords_array), bins=[bin_x_num, bin_y_num])


def test_histogramdd_invalid_range() -> None:
    coords_array = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])

    bins = [3, 3]
    invalid_range = [[5.0, 2.0], [1.0, 10.0]]

    msg = r"(max must be larger than min|must be None or pairs of increasing values)"
    with pytest.raises(ValueError, match=msg):
        np.histogramdd(coords_array, bins=bins, range=invalid_range)

    with pytest.raises(ValueError, match=msg):
        num.histogramdd(
            num.array(coords_array), bins=bins, range=invalid_range
        )


def test_histogramdd_weights_size_mismatch() -> None:
    coords_array = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])

    bins = [3, 3]
    weights_np = np.array([1.0, 2.0])
    weights_num = num.array([1.0, 2.0])

    msg = r"(weights.*same|same.*length)"
    with pytest.raises(ValueError, match=msg):
        np.histogramdd(coords_array, bins=bins, weights=weights_np)

    with pytest.raises(ValueError, match=msg):
        num.histogramdd(
            num.array(coords_array), bins=bins, weights=weights_num
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
