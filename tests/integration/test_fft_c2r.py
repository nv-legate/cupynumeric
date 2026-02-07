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

import cupynumeric as num
from cupynumeric.config import FFTCode, FFTDirection


def allclose(A: np.ndarray, B: np.ndarray) -> bool:
    if (
        B.dtype == np.float32
        or B.dtype == np.float64
        or B.dtype == np.complex64
        or B.dtype == np.complex128
    ):
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return _allclose(A, B)


def check_1d_c2r(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype) + np.random.rand(N).astype(dtype) * 1j
    Z_num = num.array(Z)

    all_kwargs = (
        {},
        {"norm": "forward"},
        {"norm": "ortho"},
        {"norm": "backward"},
        {"n": N // 2},
        {"n": N // 2 + 1},
        {"n": N * 2},
        {"n": N * 2 + 1},
    )

    for kwargs in all_kwargs:
        print(f"=== 1D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfft(Z, **kwargs)
        out_num = num.fft.irfft(Z_num, **kwargs)
        assert allclose(out, out_num)

        out = np.fft.hfft(Z, **kwargs)
        out_num = num.fft.hfft(Z_num, **kwargs)
        assert allclose(out, out_num)
        assert allclose(Z, Z_num)

    # Odd types
    out = np.fft.rfft(Z.real)
    out_num = num.fft.rfft(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_2d_c2r(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
    Z_num = num.array(Z)

    all_kwargs = (
        {},
        {"norm": "forward"},
        {"norm": "ortho"},
        {"norm": "backward"},
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
        print(f"=== 2D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfft2(Z, **kwargs)
        out_num = num.fft.irfft2(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfft2(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.irfft2(num.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfft2(Z.real)
    out_num = num.fft.rfft2(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_3d_c2r(N, dtype=np.float64):
    print(f"\n=== 3D C2R {dtype}               ===")
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
        print(f"=== 3D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfftn(Z, **kwargs)
        out_num = num.fft.irfftn(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.irfftn(num.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 2, 1), **kwargs)
        out_num = num.fft.irfftn(num.swapaxes(Z_num, 2, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfftn(Z.real)
    out_num = num.fft.rfftn(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_irfft_single_precision_cast(monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.random.rand(8).astype(np.float32) + 1j * np.random.rand(
        8
    ).astype(np.float32)
    orig_irfftn = np.fft.irfftn

    def irfftn_force_float64(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = orig_irfftn(*args, **kwargs)
        if out.dtype != np.float64:
            out = out.astype(np.float64)
        return out

    monkeypatch.setattr(np.fft, "irfftn", irfftn_force_float64)

    result = num.fft.irfft(num.array(data))
    assert result.dtype == np.float32


def test_irfft_default_n_infers_output_shape() -> None:
    z = (
        np.random.rand(2).astype(np.float32)
        + 1j * np.random.rand(2).astype(np.float32)
    ).astype(np.complex64)
    z_num = num.array(z)

    out_np = np.fft.irfft(z)

    kind = FFTCode.complex_to_real_code(z_num.dtype)
    out_num = z_num.fft(
        s=None, axes=None, kind=kind, direction=FFTDirection.INVERSE, norm=None
    )
    out_num_np = np.array(out_num)
    assert allclose(out_np, out_num_np)


def check_4d_c2r(N, dtype=np.float64):
    print(f"\n=== 4D C2R {dtype}               ===")
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
        print(f"=== 4D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfftn(Z, **kwargs)
        out_num = num.fft.irfftn(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.irfftn(num.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 2, 1), **kwargs)
        out_num = num.fft.irfftn(num.swapaxes(Z_num, 2, 1), **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 3, 1), **kwargs)
        out_num = num.fft.irfftn(num.swapaxes(Z_num, 3, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfftn(Z.real)
    out_num = num.fft.rfftn(Z_num.real)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z.real)
    out_num = num.fft.ihfft(Z_num.real)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_1d():
    check_1d_c2r(N=78)
    check_1d_c2r(N=78, dtype=np.float32)


def test_2d():
    check_2d_c2r(N=(28, 10))
    check_2d_c2r(N=(28, 10), dtype=np.float32)


def test_3d():
    check_3d_c2r(N=(6, 12, 10))
    check_3d_c2r(N=(6, 12, 10), dtype=np.float32)


def test_4d():
    check_4d_c2r(N=(6, 12, 10, 8))
    check_4d_c2r(N=(6, 12, 10, 8), dtype=np.float32)


def test_1d_int() -> None:
    Z = np.random.randint(1, 10, size=8)
    Z_num = num.array(Z)
    msg = r"Data type for FFT not supported"
    with pytest.raises(TypeError, match=msg):
        num.fft.rfft(Z_num)
    msg = r"Data type for FFT not supported"
    with pytest.raises(TypeError, match=msg):
        num.fft.irfft(Z_num)
    msg = r"Data type for FFT not supported"
    with pytest.raises(TypeError, match=msg):
        num.fft.ihfft(Z_num)


def test_norm_invalid() -> None:
    Z = (
        np.random.rand(8).astype(np.float64)
        + np.random.rand(8).astype(np.float64) * 1j
    )
    Z_num = num.array(Z)
    msg = r"Invalid norm value"
    with pytest.raises(ValueError, match=msg):
        np.fft.rfft(Z, norm="other")
    with pytest.raises(ValueError, match=msg):
        num.fft.rfft(Z_num, norm="other")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
