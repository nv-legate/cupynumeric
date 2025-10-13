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

import cupynumeric as num


DEFAULT_MEAN_TOL = 0.05


class ModuleGenerator:
    def __init__(
        self, seed=None, force_build=False
    ):  # seed parameter is ignored here
        self.seed = seed
        self.bit_generator = num.random.XORWOW(seed)

    def random_raw(self, shape):
        gen = num.random._generator.get_static_generator()
        return gen.bit_generator.random_raw(shape)

    def integers(self, low, high, size, dtype, endpoint):
        return num.random._generator.get_static_generator().integers(
            low, high, size, dtype, endpoint
        )

    # functions exposed in random module
    def beta(self, a, b, shape, dtype):
        return num.random.beta(a, b, shape, dtype)

    def binomial(self, ntrials, p, shape, dtype):
        return num.random.binomial(ntrials, p, shape, dtype)

    def bytes(self, length):
        return num.random.bytes(length)

    def cauchy(self, x0, gamma, shape, dtype):
        return gamma * num.random.standard_cauchy(shape, dtype) + x0

    def chisquare(self, df, nonc, shape, dtype):
        if nonc == 0.0:
            return num.random.chisquare(df, shape, dtype)
        else:
            return num.random.noncentral_chisquare(df, nonc, shape, dtype)

    def exponential(self, scale, shape, dtype):
        return num.random.exponential(scale, shape, dtype)

    def f(self, dfnum, dfden, shape, dtype):
        return num.random.f(dfnum, dfden, shape, dtype)

    def gamma(self, k, theta, shape, dtype):
        return num.random.gamma(k, theta, shape, dtype)

    def geometric(self, p, shape, dtype):
        return num.random.geometric(p, shape, dtype)

    def gumbel(self, mu, beta, shape, dtype):
        return num.random.gumbel(mu, beta, shape, dtype)

    def hypergeometric(self, ngood, nbad, nsample, shape, dtype):
        return num.random.hypergeometric(ngood, nbad, nsample, shape, dtype)

    def laplace(self, mu, beta, shape, dtype):
        return num.random.laplace(mu, beta, shape, dtype)

    def logistic(self, mu, beta, shape, dtype):
        return num.random.logistic(mu, beta, shape, dtype)

    def lognormal(self, mean, sigma, shape, dtype):
        return num.random.lognormal(mean, sigma, shape, dtype)

    def logseries(self, p, shape, dtype):
        return num.random.logseries(p, shape, dtype)

    def negative_binomial(self, ntrials, p, shape, dtype):
        return num.random.negative_binomial(ntrials, p, shape, dtype)

    def noncentral_chisquare(self, df, nonc, shape, dtype):
        return num.random.noncentral_chisquare(df, nonc, shape, dtype)

    def noncentral_f(self, dfnum, dfden, nonc, shape, dtype):
        return num.random.noncentral_f(dfnum, dfden, nonc, shape, dtype)

    def normal(self, mean, sigma, shape, dtype):
        return num.random.normal(mean, sigma, shape, dtype)

    def pareto(self, alpha, shape, dtype):
        return num.random.pareto(alpha, shape, dtype)

    def poisson(self, lam, size):
        return num.random.poisson(lam, size)

    def power(self, alpha, shape, dtype):
        return num.random.power(alpha, shape, dtype)

    def random(self, size, dtype, out):
        return num.random.random(size, dtype, out)

    def rayleigh(self, sigma, shape, dtype):
        return num.random.rayleigh(sigma, shape, dtype)

    def standard_cauchy(self, size, dtype):
        return num.random.standard_cauchy(size, dtype)

    def standard_exponential(self, size, dtype):
        return num.random.standard_exponential(size, dtype)

    def standard_gamma(self, shape, size, dtype):
        return num.random.standard_gamma(shape, size, dtype)

    def standard_t(self, df, shape, dtype):
        return num.random.standard_t(df, shape, dtype)

    def triangular(self, a, b, c, shape, dtype):
        return num.random.triangular(
            left=a, mode=c, right=b, size=shape, dtype=dtype
        )

    def uniform(self, low, high, size, dtype):
        return num.random.uniform(low, high, size, dtype)

    def vonmises(self, mu, kappa, shape, dtype):
        return num.random.vonmises(mu, kappa, shape, dtype)

    def wald(self, mean, scale, shape, dtype):
        return num.random.wald(mean, scale, shape, dtype)

    def weibull(self, lam, k, shape, dtype):
        return lam * num.random.weibull(k, shape, dtype)

    def zipf(self, alpha, shape, dtype):
        return num.random.zipf(alpha, shape, dtype)


def assert_distribution(
    a, theo_mean, theo_stdev, mean_tol=DEFAULT_MEAN_TOL, stdev_tol=1.0
):
    if True:
        aa = np.array(a)
        average = np.mean(aa)
        stdev = np.std(aa)
    else:  # keeping this path for further investigation
        average = num.mean(a)
        stdev = num.sqrt(
            num.mean((a - average) ** 2)
        )  # num.std(a) -> does not work
    print(
        f"average = {average} - theoretical {theo_mean}"
        + f", stdev = {stdev} - theoretical {theo_stdev}\n"
    )
    assert abs(theo_mean - average) < mean_tol * max(1.0, abs(theo_mean))
    # the theoretical standard deviation can't be 0
    assert theo_stdev != 0
    # TODO: this check is not a good proxy to validating that the samples
    #       respect the assumed random distribution unless we draw
    #       extremely many samples. until we find a better validation
    #       method, we make the check lenient to avoid random
    #       failures in the CI. (we still need the check to catch
    #       the cases that are obviously wrong.)
    assert abs(theo_stdev - stdev) / min(theo_stdev, stdev) <= stdev_tol
