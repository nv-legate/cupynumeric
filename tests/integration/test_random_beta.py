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
def test_beta_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 2.0
    beta = 5.0
    a = gen.beta(alpha, beta, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = alpha / (alpha + beta)
    theo_std = np.sqrt(alpha * beta / (alpha + beta + 1.0)) / (alpha + beta)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_beta_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 2.0
    beta = 5.0
    a = gen.beta(alpha, beta, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = alpha / (alpha + beta)
    theo_std = np.sqrt(alpha * beta / (alpha + beta + 1.0)) / (alpha + beta)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_f_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    d1 = 1.0
    d2 = 48.0
    a = gen.f(d1, d2, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = d2 / (d2 - 2.0)
    theo_std = np.sqrt(
        (2.0 * d2**2 * (d1 + d2 - 2.0)) / (d1 * (d2 - 4.0))
    ) / (d2 - 2.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_f_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    d1 = 1.0
    d2 = 48.0
    a = gen.f(d1, d2, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = d2 / (d2 - 2.0)
    theo_std = np.sqrt(
        (2.0 * d2**2 * (d1 + d2 - 2.0)) / (d1 * (d2 - 4.0))
    ) / (d2 - 2.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_logseries(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    p = 0.66
    a = gen.logseries(p, size=(1024 * 1024,), dtype=np.uint32)
    theo_mean = -1 / (np.log(1.0 - p)) * p / (1 - p)
    theo_std = -np.sqrt(-(p**2 + p * np.log(1.0 - p))) / (
        (1 - p) * np.log(1.0 - p)
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_noncentral_f_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    d1 = 1.0
    d2 = 48.0
    nonc = 1.414
    a = gen.noncentral_f(d1, d2, nonc, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = (d2 * (d1 + nonc)) / (d1 * (d2 - 2.0))
    theo_std = np.sqrt(
        2.0
        * ((d1 + nonc) ** 2 + (d1 + 2.0 * nonc) * (d2 - 2.0))
        / ((d2 - 2.0) ** 2 * (d2 - 4.0))
        * (d2 / d1) ** 2
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_noncentral_f_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    d1 = 1.0
    d2 = 48.0
    nonc = 1.414
    a = gen.noncentral_f(d1, d2, nonc, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = (d2 * (d1 + nonc)) / (d1 * (d2 - 2.0))
    theo_std = np.sqrt(
        2.0
        * ((d1 + nonc) ** 2 + (d1 + 2.0 * nonc) * (d2 - 2.0))
        / ((d2 - 2.0) ** 2 * (d2 - 4.0))
        * (d2 / d1) ** 2
    )
    assert_distribution(a, theo_mean, theo_std)


FUNC_ARGS = (
    ("beta", (2.0, 5.0)),
    ("f", (1.0, 48.0)),
    ("logseries", (0.66,)),
    ("noncentral_f", (1.0, 48.0, 1.414)),
)


@pytest.mark.parametrize("func, args", FUNC_ARGS, ids=str)
@pytest.mark.parametrize("size", ((2048 * 2048), (4096,), 25535), ids=str)
def test_beta_sizes(func, args, size):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(*args, size=size)
    a_num = getattr(gen_num, func)(*args, size=size)
    assert a_np.shape == a_num.shape


@pytest.mark.xfail
@pytest.mark.parametrize("func, args", FUNC_ARGS, ids=str)
def test_beta_size_none(func, args):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(*args, size=None)
    a_num = getattr(gen_num, func)(*args, size=None)
    # cuPyNumeric returns singleton array
    # NumPy returns scalar
    assert np.ndim(a_np) == np.ndim(a_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
