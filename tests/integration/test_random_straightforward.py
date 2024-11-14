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
import math

import legate.install_info as info
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
def test_exponential_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    a = gen.exponential(scale, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = scale
    theo_std = scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_exponential_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    a = gen.exponential(scale, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = scale
    theo_std = scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_gumbel_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.gumbel(loc, scale, size=(1024 * 1024,), dtype=np.float32)
    euler_mascheroni = 0.5772156649015328606065120900824024310421
    theo_mean = loc + euler_mascheroni * scale
    theo_std = np.pi * scale / np.sqrt(6.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_gumbel_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.gumbel(loc, scale, size=(1024 * 1024,), dtype=np.float64)
    euler_mascheroni = 0.5772156649015328606065120900824024310421
    theo_mean = loc + euler_mascheroni * scale
    theo_std = np.pi * scale / np.sqrt(6.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_laplace_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.laplace(loc, scale, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = loc
    theo_std = np.sqrt(2.0) * scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_laplace_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.laplace(loc, scale, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = loc
    theo_std = np.sqrt(2.0) * scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_logistic_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.logistic(loc, scale, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = loc
    theo_std = np.pi * scale / np.sqrt(3.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_logistic_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.logistic(loc, scale, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = loc
    theo_std = np.pi * scale / np.sqrt(3.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_pareto_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 30.0
    a = gen.pareto(alpha, size=(1024 * 1024,), dtype=np.float32)
    xm = 1.0
    theo_mean = alpha * xm / (alpha - 1.0) - 1.0
    theo_std = np.sqrt(xm * xm * alpha / (alpha - 2)) / (alpha - 1)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_pareto_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 30.0
    a = gen.pareto(alpha, size=(1024 * 1024,), dtype=np.float64)
    xm = 1.0
    theo_mean = alpha * xm / (alpha - 1.0) - 1.0
    theo_std = np.sqrt(xm * xm * alpha / (alpha - 2)) / (alpha - 1)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_power_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 3.0
    a = gen.power(alpha, size=(1024 * 1024,), dtype=np.float32)
    # power function distribution is Beta distribution with
    # alpha > 0 and beta = 1
    beta = 1.0
    theo_mean = alpha / (alpha + beta)
    theo_std = np.sqrt((alpha * beta) / (alpha + beta + 1.0)) / (alpha + beta)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_power_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 3.0
    a = gen.power(alpha, size=(1024 * 1024,), dtype=np.float64)
    # power function distribution is Beta distribution with
    # alpha > 0 and beta = 1
    beta = 1.0
    theo_mean = alpha / (alpha + beta)
    theo_std = np.sqrt((alpha * beta) / (alpha + beta + 1.0)) / (alpha + beta)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_rayleigh_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    sigma = np.pi
    a = gen.rayleigh(sigma, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = sigma * np.sqrt(np.pi / 2.0)
    theo_std = sigma * np.sqrt((4.0 - np.pi) / 2.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_rayleigh_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    sigma = np.pi
    a = gen.rayleigh(sigma, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = sigma * np.sqrt(np.pi / 2.0)
    theo_std = sigma * np.sqrt((4.0 - np.pi) / 2.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_cauchy_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    x0 = 1.414
    a = gen.cauchy(x0, 1.0, size=(1024 * 1024,), dtype=np.float32)
    # cauchy's distribution mean and stdev are not defined...
    aa = np.array(a)
    median = np.median(aa)
    assert np.abs(median - x0) < 0.01


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_cauchy_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    x0 = 1.414
    a = gen.cauchy(x0, 1.0, size=(1024 * 1024,), dtype=np.float64)
    # cauchy's distribution mean and stdev are not defined...
    aa = np.array(a)
    median = np.median(aa)
    assert np.abs(median - x0) < 0.01


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_stdcauchy_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    x0 = 0.0
    a = gen.standard_cauchy(size=(1024 * 1024,), dtype=np.float32)
    # cauchy's distribution mean and stdev are not defined...
    aa = np.array(a)
    median = np.median(aa)
    assert np.abs(median - x0) < 0.01


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_stdcauchy_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    x0 = 0.0
    a = gen.standard_cauchy(size=(1024 * 1024,), dtype=np.float64)
    # cauchy's distribution mean and stdev are not defined...
    aa = np.array(a)
    median = np.median(aa)
    assert np.abs(median - x0) < 0.01


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_stdexponential_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.0
    a = gen.standard_exponential(size=(1024 * 1024,), dtype=np.float32)
    theo_mean = scale
    theo_std = scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_stdexponential_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.0
    a = gen.standard_exponential(size=(1024 * 1024,), dtype=np.float64)
    theo_mean = scale
    theo_std = scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_triangular_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    lo = 1.414
    mi = 2.7
    hi = 3.1415
    a = gen.triangular(lo, mi, hi, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = (lo + mi + hi) / 3.0
    theo_std = np.sqrt(
        (lo**2 + mi**2 + hi**2 - lo * mi - mi * hi - hi * lo) / 18.0
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_triangular_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    lo = 1.414
    mi = 2.7
    hi = 3.1415
    a = gen.triangular(lo, mi, hi, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = (lo + mi + hi) / 3.0
    theo_std = np.sqrt(
        (lo**2 + mi**2 + hi**2 - lo * mi - mi * hi - hi * lo) / 18.0
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_weibull_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.1415
    a = gen.weibull(k, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = math.gamma(1.0 + 1.0 / k)
    theo_std = np.sqrt(
        math.gamma(1.0 + 2.0 / k) - math.gamma(1.0 + 1.0 / k) ** 2
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_weibull_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    k = 3.1415
    a = gen.weibull(k, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = math.gamma(1.0 + 1.0 / k)
    theo_std = np.sqrt(
        math.gamma(1.0 + 2.0 / k) - math.gamma(1.0 + 1.0 / k) ** 2
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.skipif(
    not info.use_cuda, reason="generators do not exist in STL random"
)
@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_bytes(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.bytes(length=1024 * 1024)
    theo_mean = 255.0 / 2.0
    theo_std = 255.0 / np.sqrt(12.0)
    assert_distribution(a, theo_mean, theo_std)


FUNC_ARGS = (
    ("exponential", ()),
    ("gumbel", ()),
    ("laplace", ()),
    ("logistic", ()),
    ("pareto", (30.0,)),
    ("power", (3.0,)),
    ("rayleigh", (np.pi,)),
    ("standard_cauchy", ()),
    ("standard_exponential", ()),
    ("triangular", (1.414, 2.7, 3.1415)),
    ("weibull", (3.1415,)),
)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize("func, args", FUNC_ARGS, ids=str)
@pytest.mark.parametrize("size", ((2048 * 2048), (4096,), 25535), ids=str)
def test_beta_sizes(t, func, args, size):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(t(seed=seed))
    a_np = getattr(gen_np, func)(*args, size=size)
    a_num = getattr(gen_num, func)(*args, size=size)
    assert a_np.shape == a_num.shape


@pytest.mark.xfail(
    reason="cuPyNumeric returns singleton array; NumPy returns scalar"
)
@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
@pytest.mark.parametrize("func, args", FUNC_ARGS, ids=str)
def test_beta_size_none(t, func, args):
    seed = 42
    gen_np = np.random.Generator(np.random.PCG64(seed=seed))
    gen_num = num.random.Generator(t(seed=seed))
    a_np = getattr(gen_np, func)(*args, size=None)
    a_num = getattr(gen_num, func)(*args, size=None)
    # cuPyNumeric returns singleton array
    # NumPy returns scalar
    # print("a_np: %s, a_num=%s\n"%(str(a_np), str(a_num)))
    assert (1 + np.ndim(a_np)) == np.ndim(a_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
