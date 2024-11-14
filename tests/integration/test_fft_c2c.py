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
from utils.comparisons import allclose as _allclose
from utils.generators import mk_0to1_array

import cupynumeric as num


def allclose(A, B):
    if B.dtype == np.float32 or B.dtype == np.complex64:
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return _allclose(A, B)


def check_1d_c2c(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype) + np.random.rand(N).astype(dtype) * 1j
    Z_num = num.array(Z)

    all_kwargs = (
        {},
        {"norm": "forward"},
        {"n": N // 2},
        {"n": N // 2 + 1},
        {"n": N * 2},
        {"n": N * 2 + 1},
    )

    for kwargs in all_kwargs:
        print(f"=== 1D C2C {dtype}, args: {kwargs} ===")
        out = np.fft.fft(Z, **kwargs)
        out_num = num.fft.fft(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.ifft(Z, **kwargs)
        out_num = num.fft.ifft(Z_num, **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfft(Z.real)
    out_num = num.fft.rfft(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_2d_c2c(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
    Z_num = num.array(Z)

    all_kwargs = (
        {},
        {"norm": "forward"},
        {"s": (N[0] // 2, N[1] - 2)},
        {"s": (N[0] + 1, N[0] + 2)},
        {"s": (N[0] // 2 + 1, N[0] + 2)},
        {"axes": (0,)},
        {"axes": (1,)},
        {"axes": (-1,)},
        {"axes": (-2,)},
        {"axes": (0, 1)},
        {"axes": (1, 0)},
        {"axes": (1, 0, 1)},
    )

    for kwargs in all_kwargs:
        print(f"=== 2D C2C {dtype}, args: {kwargs} ===")
        out = np.fft.fft2(Z, **kwargs)
        out_num = num.fft.fft2(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.fft2(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.fft2(num.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifft2(Z, **kwargs)
        out_num = num.fft.ifft2(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifft2(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.ifft2(num.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfft2(Z.real)
    out_num = num.fft.rfft2(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_3d_c2c(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
    Z_num = num.array(Z)

    all_kwargs = (
        (
            {},
            {"norm": "forward"},
            {"norm": "ortho"},
            {"s": (N[0] - 1, N[1] - 2, N[2] // 2)},
            {"s": (N[0] + 1, N[1] + 2, N[2] + 3)},
        )
        + tuple({"axes": (i,)} for i in range(3))
        + tuple({"axes": (-i,)} for i in range(1, 4))
        + tuple({"axes": (i + 1, i)} for i in range(2))
        + ({"axes": (0, 2, 1, 1, -1)},)
    )

    for kwargs in all_kwargs:
        print(f"=== 3D C2C {dtype}, args: {kwargs} ===")
        out = np.fft.fftn(Z, **kwargs)
        out_num = num.fft.fftn(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.fftn(np.swapaxes(Z, 2, 0), **kwargs)
        out_num = num.fft.fftn(num.swapaxes(Z_num, 2, 0), **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifftn(Z, **kwargs)
        out_num = num.fft.ifftn(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifftn(np.swapaxes(Z, 2, 0), **kwargs)
        out_num = num.fft.ifftn(num.swapaxes(Z_num, 2, 0), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfftn(Z.real)
    out_num = num.fft.rfftn(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_4d_c2c(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
    Z_num = num.array(Z)

    all_kwargs = (
        (
            {},
            {"norm": "forward"},
            {"norm": "ortho"},
            {"s": (N[0] - 1, N[1] - 2, N[2] // 2, N[3])},
            {"s": (N[0] + 1, N[1] + 2, N[2] + 3, N[3] - 1)},
            {"s": (N[0] + 1, N[1] + 2, N[2] + 3, N[3] // 2)},
        )
        + tuple({"axes": (i,)} for i in range(4))
        + tuple({"axes": (-i,)} for i in range(1, 5))
        + tuple({"axes": (i + 1, i)} for i in range(3))
        + ({"axes": (0, 2, 3, 1, 1, -1)},)
    )

    for kwargs in all_kwargs:
        print(f"=== 4D C2C {dtype}, args: {kwargs} ===")
        out = np.fft.fftn(Z, **kwargs)
        out_num = num.fft.fftn(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.fftn(np.swapaxes(Z, 2, 0), **kwargs)
        out_num = num.fft.fftn(num.swapaxes(Z_num, 2, 0), **kwargs)
        assert allclose(out, out_num)

        out = np.fft.fftn(np.swapaxes(Z, 3, 1), **kwargs)
        out_num = num.fft.fftn(num.swapaxes(Z_num, 3, 1), **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifftn(Z, **kwargs)
        out_num = num.fft.ifftn(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifftn(np.swapaxes(Z, 2, 0), **kwargs)
        out_num = num.fft.ifftn(num.swapaxes(Z_num, 2, 0), **kwargs)
        assert allclose(out, out_num)

        out = np.fft.ifftn(np.swapaxes(Z, 3, 1), **kwargs)
        out_num = num.fft.ifftn(num.swapaxes(Z_num, 3, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_1d():
    check_1d_c2c(N=153)
    check_1d_c2c(N=153, dtype=np.float32)


def test_2d():
    check_2d_c2c(N=(9, 100))
    check_2d_c2c(N=(9, 100), dtype=np.float32)


def test_3d():
    check_3d_c2c(N=(9, 10, 11))
    check_3d_c2c(N=(9, 10, 11), dtype=np.float32)


def test_4d():
    check_4d_c2c(N=(9, 10, 11, 12))
    check_4d_c2c(N=(9, 10, 11, 12), dtype=np.float32)


@pytest.mark.parametrize(
    "dtype",
    (
        np.complex64,
        np.float32,
        np.float64,
        pytest.param(np.int32, marks=pytest.mark.xfail),
        pytest.param(np.uint64, marks=pytest.mark.xfail),
        pytest.param(np.float16, marks=pytest.mark.xfail),
        # NumPy accepts the dtypes
        # cuPyNumeric raises
        # TypeError: FFT input not supported (missing a conversion?)
    ),
    ids=str,
)
@pytest.mark.parametrize("func", ("fftn", "ifftn"), ids=str)
def test_fftn_dtype(dtype, func):
    shape = (3, 2, 4)
    arr_np = mk_0to1_array(np, shape, dtype=dtype)
    arr_num = mk_0to1_array(num, shape, dtype=dtype)
    out_np = getattr(np.fft, func)(arr_np)
    out_num = getattr(num.fft, func)(arr_num)
    assert allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
