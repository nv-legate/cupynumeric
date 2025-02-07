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
def test_bitgenerator_type(t):
    print(f"testing for type = {t}")
    bitgen = t(seed=42)
    bitgen.random_raw(256)
    bitgen.random_raw((512, 256))
    r = bitgen.random_raw(256)  # deferred is None
    print(f"256 sum = {r.sum()}")
    r = bitgen.random_raw((1024, 1024))
    print(f"1k² sum = {r.sum()}")
    r = bitgen.random_raw(1024 * 1024)
    print(f"1M sum = {r.sum()}")
    bitgen = None
    print(f"DONE for type = {t}")


@pytest.mark.parametrize("size", ((2048 * 2048), (4096,), 25535), ids=str)
def test_bitgenerator_size(size):
    seed = 42
    gen_np = np.random.PCG64(seed=seed)
    gen_num = num.random.XORWOW(seed=seed)
    a_np = gen_np.random_raw(size=size)
    a_num = gen_num.random_raw(shape=size)
    assert a_np.shape == a_num.shape


@pytest.mark.xfail
def test_bitgenerator_size_none():
    seed = 42
    gen_np = np.random.PCG64(seed=seed)
    gen_num = num.random.XORWOW(seed=seed)
    a_np = gen_np.random_raw(size=None)
    a_num = gen_num.random_raw(shape=None)
    # cuPyNumeric returns singleton array
    # NumPy returns scalar
    assert np.ndim(a_np) == np.ndim(a_num)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_force_build(t):
    t(42, True)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_int64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(512, 653548, size=(1024,))
    print(f"1024 sum = {a.sum()}")
    a = gen.integers(512, 653548, size=(1024 * 1024,))
    print(f"1024*1024 sum = {a.sum()}")


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_int32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(512, 653548, size=(1024,), dtype=np.int32)
    print(f"1024 sum = {a.sum()}")
    a = gen.integers(512, 653548, size=(1024 * 1024,), dtype=np.int32)
    print(f"1024*1024 sum = {a.sum()}")


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_int16(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(512, 653548, size=(1024,), dtype=np.int16)
    print(f"1024 sum = {a.sum()}")
    a = gen.integers(512, 653548, size=(1024 * 1024,), dtype=np.int16)
    print(f"1024*1024 sum = {a.sum()}")


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_endpoint(t):
    high = 10
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(high, size=100, dtype=np.int16, endpoint=True)
    assert np.max(a) == high


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_random_float32(t):
    # top-level random function has a different signature
    if t is ModuleGenerator:
        return
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.random(size=(1024 * 1024,), dtype=np.float32)
    assert_distribution(a, 0.5, num.sqrt(1.0 / 12.0))


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_random_float64(t):
    # top-level random function has a different signature
    if t == ModuleGenerator:
        return
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.random(size=(1024 * 1024,), dtype=np.float64)
    assert_distribution(a, 0.5, num.sqrt(1.0 / 12.0))


def test_random_out_stats():
    seed = 42
    dtype = np.float32
    out_shape = (2, 3, 1)
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    out_np = np.empty(out_shape, dtype=dtype)
    out_num = num.empty(out_shape, dtype=dtype)
    gen_np.random(out=out_np, dtype=dtype)
    gen_num.random(out=out_num, dtype=dtype)
    assert out_np.shape == out_num.shape
    assert out_np.dtype == out_np.dtype


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_lognormal_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.lognormal(mu, sigma, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = num.exp(mu + sigma * sigma / 2.0)
    theo_std = num.sqrt(
        (num.exp(sigma * sigma) - 1) * num.exp(2 * mu + sigma * sigma)
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_lognormal_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.lognormal(mu, sigma, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = num.exp(mu + sigma * sigma / 2.0)
    theo_std = num.sqrt(
        (num.exp(sigma * sigma) - 1) * num.exp(2 * mu + sigma * sigma)
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_normal_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.normal(mu, sigma, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = mu
    theo_std = sigma
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_normal_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.normal(mu, sigma, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = mu
    theo_std = sigma
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_uniform_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    low = 1.414
    high = 3.14
    a = gen.uniform(low, high, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = (low + high) / 2.0
    theo_std = (high - low) / np.sqrt(12.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_uniform_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    low = 1.414
    high = 3.14
    a = gen.uniform(low, high, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = (low + high) / 2.0
    theo_std = (high - low) / np.sqrt(12.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_poisson(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    lam = 42
    a = gen.poisson(lam, size=(1024 * 1024,))
    theo_mean = lam
    theo_std = num.sqrt(lam)
    assert_distribution(a, theo_mean, theo_std)


FUNCS = ("poisson", "normal", "lognormal", "uniform")


@pytest.mark.parametrize("func", FUNCS, ids=str)
@pytest.mark.parametrize("size", ((2048 * 2048), (4096,), 25535), ids=str)
def test_random_sizes(func, size):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(size=size)
    a_num = getattr(gen_num, func)(size=size)
    assert a_np.shape == a_num.shape


@pytest.mark.xfail
@pytest.mark.parametrize("func", FUNCS, ids=str)
def test_random_size_none(func):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(size=None)
    a_num = getattr(gen_num, func)(size=None)
    # cuPyNumeric returns singleton array
    # NumPy returns scalar
    assert np.ndim(a_np) == np.ndim(a_num)


class TestBitGeneratorErrors:
    def test_init_bitgenerator(self):
        expected_exc = NotImplementedError
        with pytest.raises(expected_exc):
            np.random.BitGenerator()
        with pytest.raises(expected_exc):
            num.random.BitGenerator()

    @pytest.mark.xfail
    @pytest.mark.parametrize("dtype", (np.int32, str))
    def test_random_invalid_dtype(self, dtype):
        expected_exc = TypeError
        seed = 42
        gen_np = np.random.Generator(np.random.PCG64(seed=seed))
        gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
        with pytest.raises(expected_exc):
            gen_np.random(size=(1024 * 1024,), dtype=dtype)
            # TypeError: Unsupported dtype dtype('int32') for random
        with pytest.raises(expected_exc):
            gen_num.random(size=(1024 * 1024,), dtype=dtype)
            # NotImplementedError: type for random.uniform has to be float64
            # or float32

    def test_random_out_dtype_mismatch(self):
        expected_exc = TypeError
        seed = 42
        dtype = np.float32
        out_shape = (3, 2, 3)
        out_dtype = np.float64
        gen_np = np.random.Generator(np.random.PCG64(seed=seed))
        gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
        out_np = np.empty(out_shape, dtype=out_dtype)
        out_num = num.empty(out_shape, dtype=out_dtype)
        with pytest.raises(expected_exc):
            gen_np.random(out=out_np, dtype=dtype)
        with pytest.raises(expected_exc):
            gen_num.random(out=out_num, dtype=dtype)

    def test_random_out_shape_mismatch(self):
        expected_exc = ValueError
        seed = 42
        size = (1024 * 1024,)
        out_shape = (3, 2, 3)
        gen_np = np.random.Generator(np.random.PCG64(seed=seed))
        gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
        out_np = np.empty(out_shape)
        out_num = num.empty(out_shape)
        with pytest.raises(expected_exc):
            gen_np.random(size=size, out=out_np)
        with pytest.raises(expected_exc):
            gen_num.random(size=size, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
