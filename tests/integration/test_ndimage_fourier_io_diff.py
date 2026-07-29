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

scipy_ndimage = pytest.importorskip("scipy.ndimage")


def _make_input(shape, dtype):
    rng = np.random.default_rng(1729 + len(shape) + sum(shape))
    real = rng.random(shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        imag = rng.random(shape)
        return (real + 1j * imag).astype(dtype)
    return real.astype(dtype)


def _assert_allclose(actual, expected, *, n):
    # SciPy returns NaNs for n == 0 due to division by zero semantics.
    assert allclose(
        np.asarray(actual), expected, rtol=1e-5, atol=1e-8, equal_nan=(n == 0)
    )


def test_fourier_in_out_diff_types():
    shape = (4, 4)

    in_dtype = np.float64
    out_dtype = np.float32
    sparam = 2.0

    assert not np.issubdtype(np.dtype(in_dtype), np.complexfloating)

    image_tf_np = np.fft.fftn(_make_input(shape, in_dtype)).astype(in_dtype)
    image_tf_num = num.asarray(image_tf_np)

    expected = np.empty(shape, dtype=out_dtype)
    actual = num.empty(shape, dtype=out_dtype)

    scipy_funcs = [
        scipy_ndimage.fourier_gaussian,
        scipy_ndimage.fourier_uniform,
        scipy_ndimage.fourier_ellipsoid,
    ]

    num_funcs = [
        num.ndimage.fourier_gaussian,
        num.ndimage.fourier_uniform,
        num.ndimage.fourier_ellipsoid,
    ]

    for scif, numf in list(zip(scipy_funcs, num_funcs)):
        scif(image_tf_np, sparam, n=-1, output=expected)
        numf(image_tf_num, sparam, n=-1, output=actual)

        _assert_allclose(actual, expected, n=-1)


def test_fourier_shift_in_out_diff_types():
    shape = (4, 4)

    # Fourier Shift expects a complex return,
    # even for shift=0, hence need complex outputs
    #
    in_dtype = np.complex128
    out_dtype = np.complex64

    shift = 2.0

    image_tf_np = np.fft.fftn(_make_input(shape, in_dtype)).astype(in_dtype)
    image_tf_num = num.asarray(image_tf_np)

    expected = np.empty(shape, dtype=out_dtype)
    actual = num.empty(shape, dtype=out_dtype)

    scipy_ndimage.fourier_shift(image_tf_np, shift, n=-1, output=expected)
    num.ndimage.fourier_shift(image_tf_num, shift, n=-1, output=actual)

    _assert_allclose(actual, expected, n=-1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
