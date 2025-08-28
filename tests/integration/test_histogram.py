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


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
def test_histogram_no_weights(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    if num.isscalar(bins):
        bins_maybe_array = bins
    else:
        bins_maybe_array = np.array(bins)
        assert bins_maybe_array.shape[0] > 0

    np_out, np_bins_out = np.histogram(src_array, bins_maybe_array)
    num_out, num_bins_out = num.histogram(src_array, bins_maybe_array)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_weights(src, bins, weights, density):
    eps = 1.0e-8
    src_array = np.array(src)

    if num.isscalar(bins):
        bins_maybe_array = bins
    else:
        bins_maybe_array = np.array(bins)

    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins_maybe_array, weights=weights_array, density=density
    )
    num_out, num_bins_out = num.histogram(
        src_array, bins_maybe_array, weights=weights_array, density=density
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3, 1.7],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_invalid_weights_ndim(src, bins, weights, density):
    src_array = np.array(src)

    if num.isscalar(bins):
        bins_maybe_array = bins
    else:
        bins_maybe_array = np.array(bins)

    weights_array = np.array(weights)
    msg = r"weights should have the same shape as a."
    with pytest.raises(ValueError, match=msg):
        np.histogram(
            src_array, bins_maybe_array, weights=weights_array, density=density
        )
    msg = r"`weights` array must be same shape for histogram"
    with pytest.raises(ValueError, match=msg):
        num.histogram(
            src_array, bins_maybe_array, weights=weights_array, density=density
        )


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
@pytest.mark.parametrize("ranges", ((3, 6), (1, 3)))
def test_histogram_ranges(src, bins, weights, density, ranges):
    eps = 1.0e-8
    src_array = np.array(src)
    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins, density=density, weights=weights_array, range=ranges
    )

    num_out, num_bins_out = num.histogram(
        src_array, bins, density=density, weights=weights_array, range=ranges
    )

    assert allclose(np_bins_out, num_bins_out, atol=eps)
    assert allclose(np_out, num_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
@pytest.mark.parametrize("ranges", ((6, 3), (3, 1)))
def test_histogram_invalid_ranges(src, bins, weights, density, ranges):
    src_array = np.array(src)
    weights_array = np.array(weights)
    msg = r"max must be larger than min in range parameter."
    with pytest.raises(ValueError, match=msg):
        np.histogram(
            src_array,
            bins,
            density=density,
            weights=weights_array,
            range=ranges,
        )
    msg = r"`range` must be a pair of increasing values."
    with pytest.raises(ValueError, match=msg):
        num.histogram(
            src_array,
            bins,
            density=density,
            weights=weights_array,
            range=ranges,
        )


@pytest.mark.parametrize(
    "src", ([0, 2, 1, 3, 1, 4, 3, 2], [4, 2, 3, 3, 2, 4, 3, 2])
)
@pytest.mark.parametrize(
    "bins", ([5, 8, 14], [0, 0.1, 0.7, 1.0, 1.2], [1, 2, 3, 3, 5, 6, 7])
)
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
def test_histogram_extreme_bins(src, bins, weights):
    eps = 1.0e-8
    src_array = np.array(src)
    bins_array = np.array(bins)
    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins_array, weights=weights_array
    )
    num_out, num_bins_out = num.histogram(
        src_array, bins_array, weights=weights_array
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([], [2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
def test_histogram_no_weights_scalar_bin(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    np_out, np_bins_out = np.histogram(src_array, bins)

    num_out, num_bins_out = num.histogram(src_array, bins)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_weights_scalar_bin(src, bins, weights, density):
    eps = 1.0e-8
    src_array = np.array(src)

    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins, weights=weights_array, density=density
    )
    num_out, num_bins_out = num.histogram(
        src_array, bins, weights=weights_array, density=density
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([[2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2]])
)
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
def test_histogram_src(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    if num.isscalar(bins):
        bins_maybe_array = bins
    else:
        bins_maybe_array = np.array(bins)
        assert bins_maybe_array.shape[0] > 0

    np_out, np_bins_out = np.histogram(src_array, bins_maybe_array)
    num_out, num_bins_out = num.histogram(src_array, bins_maybe_array)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([[5, 7], [5, 7]], [[1, 3], [0, 0]]))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_weights_2d_bin(src, bins, weights, density):
    src_array = np.array(src)
    weights_array = np.array(weights)
    msg = r"`bins` must be 1d, when an array"
    with pytest.raises(ValueError, match=msg):
        np.histogram(src_array, bins, weights=weights_array, density=density)
    with pytest.raises(ValueError, match=msg):
        num.histogram(src_array, bins, weights=weights_array, density=density)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5.5, 7.5 + 3j))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_weights_scalar_float_bin(src, bins, weights, density):
    src_array = np.array(src)
    weights_array = np.array(weights)

    msg = r"`bins` must be an integer, a string, or an array"
    with pytest.raises(TypeError, match=msg):
        np.histogram(src_array, bins, weights=weights_array, density=density)
    msg = r"`bins` must be array or integer type"
    with pytest.raises(TypeError, match=msg):
        num.histogram(src_array, bins, weights=weights_array, density=density)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([5, 14, 8], [3, 3, 1]))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
def test_histogram_decrease_bins(src, bins, weights):
    src_array = np.array(src)
    bins_array = np.array(bins)
    weights_array = np.array(weights)

    msg = r"`bins` must increase monotonically, when an array"
    with pytest.raises(ValueError, match=msg):
        np.histogram(src_array, bins_array, weights=weights_array)
    with pytest.raises(ValueError, match=msg):
        num.histogram(src_array, bins_array, weights=weights_array)


@pytest.mark.parametrize("src", ([], [2], [5]))
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
def test_histogram_singleton_empty(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    bins_array = np.array(bins)

    np_out, np_bins_out = np.histogram(src_array, bins_array)

    num_out, num_bins_out = num.histogram(src_array, bins_array)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([[1, 2, 3], [4, 5, 6]], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
)
@pytest.mark.parametrize("bins", (3, 5))
def test_histogram_2d_array_flatten(
    src: list[int] | list[float], bins: int
) -> None:
    eps = 1.0e-8
    src_array = np.array(src)

    assert src_array.ndim == 2

    np_out, np_bins_out = np.histogram(src_array, bins)
    num_out, num_bins_out = num.histogram(src_array, bins)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)

    total_elements = src_array.size
    assert abs(np_out.sum() - total_elements) < eps
    assert abs(num_out.sum() - total_elements) < eps


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
