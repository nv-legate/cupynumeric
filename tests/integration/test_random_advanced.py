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
import os

import numpy as np
import pytest
from utils.random import ModuleGenerator, assert_distribution

import cupynumeric as num

LEGATE_TEST = os.environ.get("LEGATE_TEST", None) == "1"

BITGENERATOR_ARGS = [
    ModuleGenerator,
    num.random.XORWOW,
    num.random.MRG32k3a,
    num.random.PHILOX4_32_10,
]


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_standard_t_float32(t: object):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    nu = 3.1415
    a = gen.standard_t(df=nu, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = 0
    theo_std = np.sqrt(nu / (nu - 2.0))
    assert_distribution(a, theo_mean, theo_std, 0.1)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_standard_t_float64(t: object):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    nu = 3.1415
    a = gen.standard_t(df=nu, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = 0
    theo_std = np.sqrt(nu / (nu - 2.0))
    assert_distribution(a, theo_mean, theo_std, 0.1)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_vonmises_float32(t: object):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.vonmises(mu, kappa, size=(1024 * 1024,), dtype=np.float32)
    ref_a = np.random.vonmises(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_vonmises_float64(t: object):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.vonmises(mu, kappa, size=(1024 * 1024,), dtype=np.float64)
    ref_a = np.random.vonmises(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize("kappa", (1e-9, 1e-6), ids=str)
def test_vonmises_edge_cases(t: object, kappa: float):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    a = gen.vonmises(mu, kappa, size=(1024 * 1024,))
    ref_a = np.random.vonmises(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize(
    "ngood, nbad, nsample",
    ((60, 440, 200), (440, 60, 300), (20.0, 77, 1), (77, 20.0, 1)),
    ids=str,
)
def test_hypergeometric(t: object, ngood: int, nbad: int, nsample: int):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    N = ngood + nbad
    K = ngood
    n = nsample
    a = gen.hypergeometric(
        ngood, nbad, nsample, size=(1024 * 1024,), dtype=np.uint32
    )
    theo_mean = n * K / N
    theo_std = np.sqrt(n * (K * (N - K) * (N - n)) / (N * N * (N - 1)))
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize("p", (0.707, 0.333))
def test_geometric(t: object, p: float):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.geometric(p, size=(1024 * 1024,), dtype=np.uint32)
    theo_mean = 1 / p
    theo_std = np.sqrt(1 - p) / p
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize(
    "s",
    (
        7.5,
        pytest.param(
            (1.2, 3.1415),
            marks=pytest.mark.xfail,
            # NumPy returns 1-dim array
            # cuPyNumeric raises TypeError: float() argument must be a string
            # or a real number, not 'tuple'
        ),
    ),
    ids=str,
)
def test_zipf(t: object, s: float):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.zipf(a=s, size=(1024 * 1024,), dtype=np.uint32)
    ref_a = np.random.zipf(s, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std, 0.2)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_wald_float32(t: object):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.wald(mu, kappa, size=(1024 * 1024,), dtype=np.float32)
    ref_a = np.random.wald(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_wald_float64(t: object):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.wald(mu, kappa, size=(1024 * 1024,), dtype=np.float64)
    ref_a = np.random.wald(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize("n", (15, 200))
@pytest.mark.parametrize("p", (0.444, 0.666))
def test_binomial(t: object, n: int, p: float):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.binomial(ntrials=n, p=p, size=(1024 * 1024,), dtype=np.uint32)
    theo_mean = n * p
    theo_std = np.sqrt(n * p * (1 - p))
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize("n", (15, 200))
@pytest.mark.parametrize("p", (0.444, 0.666))
def test_negative_binomial(t: object, n: int, p: float):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.negative_binomial(
        ntrials=n, p=p, size=(1024 * 1024,), dtype=np.uint32
    )
    ref_a = np.random.negative_binomial(n, p, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


FUNC_ARGS = (
    ("binomial", (15, 0.666)),
    ("negative_binomial", (15, 0.666)),
    ("geometric", (0.707,)),
    ("hypergeometric", (60, 440, 200)),
    ("standard_t", (3.1415,)),
    ("vonmises", (1.414, 3.1415)),
    ("wald", (1.414, 3.1415)),
    ("zipf", (7.5,)),
)


@pytest.mark.parametrize("func, args", FUNC_ARGS, ids=str)
@pytest.mark.parametrize("size", ((2048 * 2048), (4096,), 25535), ids=str)
def test_random_sizes(
    func: str, args: tuple[int | float, ...], size: int | tuple[int, ...]
):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(*args, size=size)
    a_num = getattr(gen_num, func)(*args, size=size)
    assert a_np.shape == a_num.shape


@pytest.mark.xfail
@pytest.mark.parametrize("func, args", FUNC_ARGS, ids=str)
def test_random_size_none(func: str, args: tuple[int | float, ...]):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
    a_np = getattr(gen_np, func)(*args, size=None)
    a_num = getattr(gen_num, func)(*args, size=None)
    # cuPyNumeric returns singleton array
    # NumPy returns scalar
    assert np.ndim(a_np) == np.ndim(a_num)


class TestRandomErrors:
    # cuPyNumeric zipf hangs on the invalid args when LEGATE_TEST=1
    @pytest.mark.skipif(LEGATE_TEST, reason="Test hang when LEGATE_TEST=1")
    @pytest.mark.parametrize(
        "dist, expected_exc",
        (
            (0.77, ValueError),
            (-5, ValueError),
            (None, TypeError),
            ((1, 5, 3), ValueError),
        ),
        ids=str,
    )
    def test_zipf_invalid_dist(self, dist: float, expected_exc: Exception):
        seed = 42
        gen_np = np.random.Generator(np.random.PCG64(seed=seed))
        gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
        with pytest.raises(expected_exc):
            gen_np.zipf(dist)
        with pytest.raises(expected_exc):
            gen_num.zipf(dist)

    @pytest.mark.skipif(LEGATE_TEST, reason="Test hang when LEGATE_TEST=1")
    @pytest.mark.parametrize(
        "ngood, nbad, nsample", ((200, 60, 500), ((1, 5, 7), 6, 22)), ids=str
    )
    def test_hypergeometric_invalid_args(
        self, ngood: int, nbad: int, nsample: int
    ):
        expected_exc = ValueError
        seed = 42
        gen_np = np.random.Generator(np.random.PCG64(seed=seed))
        gen_num = num.random.Generator(num.random.XORWOW(seed=seed))
        with pytest.raises(expected_exc):
            gen_np.hypergeometric(ngood, nbad, nsample)
        with pytest.raises(expected_exc):
            gen_num.hypergeometric(ngood, nbad, nsample)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
