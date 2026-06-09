# Copyright 2026 NVIDIA Corporation
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

from __future__ import annotations

import numpy as np
import pytest
from utils.comparisons import allclose

import cupynumeric as num
from itertools import product

scipy_ndimage = pytest.importorskip("scipy.ndimage")


SHAPES = [(1,), (7,), (8,), (5, 6), (6, 5), (4, 5, 6), (3, 4, 5, 6)]

SIGMAS = [0.0, 0.5, 2.0, (0.5, 1.0), (0.0, 1.0, 2.0), (0.5, 1.0, 1.5, 2.0)]

REAL_FFT_N = [-1, 0, 1, 2, 5, 8, 11]


def _valid_sigmas(ndim):
    return [
        sigma for sigma in SIGMAS if np.ndim(sigma) == 0 or len(sigma) == ndim
    ]


def _make_input(shape, dtype):
    rng = np.random.default_rng(1729 + len(shape) + sum(shape))
    real = rng.random(shape)
    imag = rng.random(shape)
    return (real + 1j * imag).astype(dtype)


def _assert_allclose(actual, expected, *, n):
    # SciPy returns NaNs for n == 0 due to division by zero semantics.
    assert allclose(
        np.asarray(actual), expected, rtol=1e-5, atol=1e-8, equal_nan=(n == 0)
    )


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_fourier_gaussian_complex_fft_all_axes(dtype, shape):
    ndim = len(shape)
    image_tf_np = np.fft.fftn(_make_input(shape, dtype)).astype(dtype)
    image_tf_num = num.asarray(image_tf_np)

    for sigma in _valid_sigmas(ndim):
        expected = scipy_ndimage.fourier_gaussian(image_tf_np, sigma, n=-1)
        actual = num.ndimage.fourier_gaussian(image_tf_num, sigma, n=-1)
        _assert_allclose(actual, expected, n=-1)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_fourier_gaussian_real_fft_axis_modes(dtype, shape):
    ndim = len(shape)
    image_tf_np = np.fft.fftn(_make_input(shape, dtype)).astype(dtype)
    image_tf_num = num.array(image_tf_np)

    for axis, n in product(range(-ndim, ndim), REAL_FFT_N):
        expected = scipy_ndimage.fourier_gaussian(
            image_tf_np, 2.0, n=n, axis=axis
        )
        actual = num.ndimage.fourier_gaussian(
            image_tf_num, 2.0, n=n, axis=axis
        )
        _assert_allclose(actual, expected, n=n)


@pytest.mark.parametrize("shape", [(7,), (5, 6), (4, 5, 6)])
def test_fourier_gaussian_matches_rfft_convention(shape):
    image_np = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    ndim = len(shape)

    for axis in range(ndim):
        image_tf_np = np.fft.rfft(image_np, axis=axis)
        image_tf_num = num.array(image_tf_np)
        n = shape[axis]

        expected = scipy_ndimage.fourier_gaussian(
            image_tf_np, 1.25, n=n, axis=axis
        )
        actual = num.ndimage.fourier_gaussian(
            image_tf_num, 1.25, n=n, axis=axis
        )
        _assert_allclose(actual, expected, n=n)


@pytest.mark.parametrize("shape", [(1,), (1, 5), (5, 1), (1, 4, 1)])
def test_fourier_gaussian_singleton_dimensions(shape):
    image_tf_np = _make_input(shape, np.complex128)
    image_tf_num = num.array(image_tf_np)

    sigma = tuple(float(i + 1) for i in range(len(shape)))
    expected = scipy_ndimage.fourier_gaussian(image_tf_np, sigma)
    actual = num.ndimage.fourier_gaussian(image_tf_num, sigma)

    _assert_allclose(actual, expected, n=-1)


@pytest.mark.parametrize("shape", [(8,), (5, 6), (4, 5, 6)])
@pytest.mark.parametrize("sigma", [0.0, (0.0,), 100.0])
def test_fourier_gaussian_corner_sigmas(shape, sigma):
    if np.ndim(sigma) != 0 and len(sigma) != len(shape):
        sigma = (0.0,) * len(shape)

    image_tf_np = _make_input(shape, np.complex128)
    image_tf_num = num.array(image_tf_np)

    expected = scipy_ndimage.fourier_gaussian(image_tf_np, sigma)
    actual = num.ndimage.fourier_gaussian(image_tf_num, sigma)

    _assert_allclose(actual, expected, n=-1)


def test_fourier_gaussian_output_argument():
    image_tf_np = _make_input((5, 6), np.complex128)
    image_tf_num = num.array(image_tf_np)

    output = num.empty_like(image_tf_num)
    result = num.ndimage.fourier_gaussian(image_tf_num, 2.0, output=output)

    expected = scipy_ndimage.fourier_gaussian(image_tf_np, 2.0)

    assert result is output
    _assert_allclose(output, expected, n=-1)


@pytest.mark.parametrize("bad_sigma", [(1.0, 2.0), (1.0, 2.0, 3.0)])
def test_fourier_gaussian_bad_sigma_length(bad_sigma):
    image_tf = num.ones((4,), dtype=np.complex128)

    with pytest.raises(RuntimeError):
        num.ndimage.fourier_gaussian(image_tf, bad_sigma)


@pytest.mark.parametrize("axis", [-4, 3])
def test_fourier_gaussian_bad_axis(axis):
    image_tf = num.ones((4, 5, 6), dtype=np.complex128)

    with pytest.raises((ValueError, np.exceptions.AxisError)):
        num.ndimage.fourier_gaussian(image_tf, 2.0, n=4, axis=axis)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_fourier_gaussian_0d_input(dtype):
    arr = num.array(np.array(1, dtype=dtype))

    with pytest.raises(RuntimeError, match="input must have rank > 0"):
        num.ndimage.fourier_gaussian(arr, 1.0)


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint64]
)
def test_fourier_gaussian_unsupported_dtype(dtype):
    arr = num.ones((8,), dtype=dtype)

    with pytest.raises(RuntimeError, match=r"input dtype .* not supported\."):
        num.ndimage.fourier_gaussian(arr, 1.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
