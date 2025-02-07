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
from utils.random import ModuleGenerator, assert_distribution

import cupynumeric as num

BITGENERATOR_ARGS = [
    ModuleGenerator,
    num.random.XORWOW,
    num.random.MRG32k3a,
    num.random.PHILOX4_32_10,
]


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_gamma_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.1415
    theta = 1.414
    a = gen.gamma(k, theta, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = k * theta
    theo_std = np.sqrt(k) * theta
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_gamma_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.1415
    theta = 1.414
    a = gen.gamma(k, theta, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = k * theta
    theo_std = np.sqrt(k) * theta
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_standard_gamma_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.1415
    theta = 1.0
    a = gen.standard_gamma(k, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = k * theta
    theo_std = np.sqrt(k) * theta
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_standard_gamma_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.1415
    theta = 1.0
    a = gen.standard_gamma(k, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = k * theta
    theo_std = np.sqrt(k) * theta
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_chisquare_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.0
    a = gen.chisquare(k, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = k
    theo_std = np.sqrt(2.0 * k)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_chisquare_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.0
    a = gen.chisquare(k, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = k
    theo_std = np.sqrt(2.0 * k)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_noncentral_chisquare_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.0
    lam = 1.414
    a = gen.noncentral_chisquare(k, lam, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = k + lam
    theo_std = np.sqrt(2.0 * (k + 2.0 * lam))
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_noncentral_chisquare_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.0
    lam = 1.414
    a = gen.noncentral_chisquare(k, lam, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = k + lam
    theo_std = np.sqrt(2.0 * (k + 2.0 * lam))
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("func", ("gamma", "noncentral_chisquare"), ids=str)
@pytest.mark.parametrize("size", ((2048 * 2048), (4096,), 25535), ids=str)
def test_gamma_sizes(func, size):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(3.1415, 1.414, size=size)
    a_num = getattr(gen_num, func)(3.1415, 1.414, size=size)
    assert a_np.shape == a_num.shape


@pytest.mark.xfail
@pytest.mark.parametrize("func", ("gamma", "noncentral_chisquare"), ids=str)
def test_gamma_size_none(func):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(3.1415, 1.414, size=None)
    a_num = getattr(gen_num, func)(3.1415, 1.414, size=None)
    # cuPyNumeric returns singleton array
    # NumPy returns scalar
    assert np.ndim(a_np) == np.ndim(a_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
