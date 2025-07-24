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

import pytest
from utils.comparisons import allclose

import cupynumeric as num


def test_interpolation_x():
    import scipy.signal as signal

    nz = 100
    nx = 200
    hs = 2
    nvariables = 4
    shape = (nvariables, nz + 2 * hs, nx + 2 * hs)
    nelements = num.prod(shape)

    kernel = num.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=num.float64
    ).reshape(1, 1, 4)
    state = num.arange(nelements).astype(num.float64).reshape(shape)
    out_legate = num.convolve(state[:, 2 : nz + 2, :], kernel, mode="same")
    out_scipy = signal.convolve(state[:, 2 : nz + 2, :], kernel, mode="same")

    assert allclose(out_scipy, out_legate)


def test_interpolation_z():
    import scipy.signal as signal

    nz = 100
    nx = 200
    hs = 2
    nvariables = 4
    shape = (nvariables, nz + 2 * hs, nx + 2 * hs)
    nelements = num.prod(shape)

    kernel = num.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=num.float64
    ).reshape(1, 4, 1)
    state = num.arange(nelements).astype(num.float64).reshape(shape)
    out_legate = num.convolve(state[:, :, 2 : nx + 2], kernel, mode="same")
    out_scipy = signal.convolve(state[:, :, 2 : nx + 2], kernel, mode="same")

    assert allclose(out_scipy, out_legate)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
