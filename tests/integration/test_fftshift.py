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

import cupynumeric as num


def test_fftshift_1d():
    freqs = np.fft.fftfreq(10, 0.1)
    a_np = np.fft.fftshift(freqs)
    a_num = num.fft.fftshift(freqs)
    assert np.array_equal(a_num, a_np)


def test_fftshift_2d():
    freqs = np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3)
    a_np = np.fft.fftshift(freqs)
    a_num = num.fft.fftshift(freqs)
    assert np.array_equal(a_num, a_np)


def test_fftshift_axis():
    freqs = np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3)
    a_np = np.fft.fftshift(freqs, axes=(1,))
    a_num = num.fft.fftshift(freqs, axes=(1,))
    assert np.array_equal(a_num, a_np)


def test_fftshift_axis_int() -> None:
    freqs = np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3)
    a_np = np.fft.fftshift(freqs, axes=1)
    a_num = num.fft.fftshift(freqs, axes=1)
    assert np.array_equal(a_num, a_np)


def test_ifftshift_1d():
    freqs = np.fft.fftshift(np.fft.fftfreq(10, 0.1))
    a_np = np.fft.ifftshift(freqs)
    a_num = num.fft.ifftshift(freqs)
    assert np.array_equal(a_num, a_np)


def test_ifftshift_2d():
    freqs = np.fft.fftshift(np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3))
    a_np = np.fft.ifftshift(freqs)
    a_num = num.fft.ifftshift(freqs)
    assert np.array_equal(a_num, a_np)


def test_ifftshift_axis():
    freqs = np.fft.fftshift(
        np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3), axes=(1,)
    )
    a_np = np.fft.ifftshift(freqs, axes=(1,))
    a_num = num.fft.ifftshift(freqs, axes=(1,))

    assert np.array_equal(a_num, a_np)


def test_ifftshift_axis_int() -> None:
    freqs = np.fft.fftshift(
        np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3), axes=(1,)
    )
    a_np = np.fft.ifftshift(freqs, axes=1)
    a_num = num.fft.ifftshift(freqs, axes=1)
    assert np.array_equal(a_num, a_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
