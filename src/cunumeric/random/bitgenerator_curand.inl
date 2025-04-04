/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <mutex>

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/random/rnd_types.h"
#include "cunumeric/random/randutil/randutil.h"

namespace cunumeric {

using namespace legate;

template <VariantKind kind>
struct CURANDGeneratorBuilder;

#pragma region wrapper to randutil

struct CURANDGenerator {
  randutilGenerator_t gen_;
  uint64_t seed_;
  uint64_t generatorId_;
  randRngType type_;

 protected:
  CURANDGenerator(BitGeneratorType gentype, uint64_t seed, uint64_t generatorId)
    : type_(get_rndRngType(gentype)), seed_(seed), generatorId_(generatorId)
  {
    randutil_log().debug() << "CURANDGenerator::create";
  }

  CURANDGenerator(const CURANDGenerator&) = delete;

 public:
  virtual ~CURANDGenerator() { randutil_log().debug() << "CURANDGenerator::destroy"; }

  void generate_raw(uint64_t count, uint32_t* out)
  {
    CHECK_RND_ENGINE(::randutilGenerateRawUInt32(gen_, out, count));
  }
  void generate_integer_64(uint64_t count, int64_t* out, int64_t low, int64_t high)
  {
    CHECK_RND_ENGINE(::randutilGenerateIntegers64(gen_, out, count, low, high));
  }
  void generate_integer_16(uint64_t count, int16_t* out, int16_t low, int16_t high)
  {
    CHECK_RND_ENGINE(::randutilGenerateIntegers16(gen_, out, count, low, high));
  }
  void generate_integer_32(uint64_t count, int32_t* out, int32_t low, int32_t high)
  {
    CHECK_RND_ENGINE(::randutilGenerateIntegers32(gen_, out, count, low, high));
  }
  void generate_uniform_64(uint64_t count, double* out, double low, double high)
  {
    CHECK_RND_ENGINE(::randutilGenerateUniformDoubleEx(gen_, out, count, low, high));
  }
  void generate_uniform_32(uint64_t count, float* out, float low, float high)
  {
    CHECK_RND_ENGINE(::randutilGenerateUniformEx(gen_, out, count, low, high));
  }
  void generate_lognormal_64(uint64_t count, double* out, double mean, double stdev)
  {
    CHECK_RND_ENGINE(::randutilGenerateLogNormalDoubleEx(gen_, out, count, mean, stdev));
  }
  void generate_lognormal_32(uint64_t count, float* out, float mean, float stdev)
  {
    CHECK_RND_ENGINE(::randutilGenerateLogNormalEx(gen_, out, count, mean, stdev));
  }
  void generate_normal_64(uint64_t count, double* out, double mean, double stdev)
  {
    CHECK_RND_ENGINE(::randutilGenerateNormalDoubleEx(gen_, out, count, mean, stdev));
  }
  void generate_normal_32(uint64_t count, float* out, float mean, float stdev)
  {
    CHECK_RND_ENGINE(::randutilGenerateNormalEx(gen_, out, count, mean, stdev));
  }
  void generate_poisson(uint64_t count, uint32_t* out, double lam)
  {
    CHECK_RND_ENGINE(::randutilGeneratePoissonEx(gen_, out, count, lam));
  }
  void generate_exponential_64(uint64_t count, double* out, double scale)
  {
    CHECK_RND_ENGINE(::randutilGenerateExponentialDoubleEx(gen_, out, count, scale));
  }
  void generate_exponential_32(uint64_t count, float* out, float scale)
  {
    CHECK_RND_ENGINE(::randutilGenerateExponentialEx(gen_, out, count, scale));
  }
  void generate_gumbel_64(uint64_t count, double* out, double mu, double beta)
  {
    CHECK_RND_ENGINE(::randutilGenerateGumbelDoubleEx(gen_, out, count, mu, beta));
  }
  void generate_gumbel_32(uint64_t count, float* out, float mu, float beta)
  {
    CHECK_RND_ENGINE(::randutilGenerateGumbelEx(gen_, out, count, mu, beta));
  }
  void generate_laplace_64(uint64_t count, double* out, double mu, double beta)
  {
    CHECK_RND_ENGINE(::randutilGenerateLaplaceDoubleEx(gen_, out, count, mu, beta));
  }
  void generate_laplace_32(uint64_t count, float* out, float mu, float beta)
  {
    CHECK_RND_ENGINE(::randutilGenerateLaplaceEx(gen_, out, count, mu, beta));
  }
  void generate_logistic_64(uint64_t count, double* out, double mu, double beta)
  {
    CHECK_RND_ENGINE(::randutilGenerateLogisticDoubleEx(gen_, out, count, mu, beta));
  }
  void generate_logistic_32(uint64_t count, float* out, float mu, float beta)
  {
    CHECK_RND_ENGINE(::randutilGenerateLogisticEx(gen_, out, count, mu, beta));
  }
  void generate_pareto_64(uint64_t count, double* out, double alpha)
  {
    CHECK_RND_ENGINE(::randutilGenerateParetoDoubleEx(gen_, out, count, 1.0, alpha));
  }
  void generate_pareto_32(uint64_t count, float* out, float alpha)
  {
    CHECK_RND_ENGINE(::randutilGenerateParetoEx(gen_, out, count, 1.0f, alpha));
  }
  void generate_power_64(uint64_t count, double* out, double alpha)
  {
    CHECK_RND_ENGINE(::randutilGeneratePowerDoubleEx(gen_, out, count, alpha));
  }
  void generate_power_32(uint64_t count, float* out, float alpha)
  {
    CHECK_RND_ENGINE(::randutilGeneratePowerEx(gen_, out, count, alpha));
  }
  void generate_rayleigh_64(uint64_t count, double* out, double sigma)
  {
    CHECK_RND_ENGINE(::randutilGenerateRayleighDoubleEx(gen_, out, count, sigma));
  }
  void generate_rayleigh_32(uint64_t count, float* out, float sigma)
  {
    CHECK_RND_ENGINE(::randutilGenerateRayleighEx(gen_, out, count, sigma));
  }
  void generate_cauchy_64(uint64_t count, double* out, double x0, double gamma)
  {
    CHECK_RND_ENGINE(::randutilGenerateCauchyDoubleEx(gen_, out, count, x0, gamma));
  }
  void generate_cauchy_32(uint64_t count, float* out, float x0, float gamma)
  {
    CHECK_RND_ENGINE(::randutilGenerateCauchyEx(gen_, out, count, x0, gamma));
  }
  void generate_triangular_64(uint64_t count, double* out, double a, double b, double c)
  {
    CHECK_RND_ENGINE(::randutilGenerateTriangularDoubleEx(gen_, out, count, a, b, c));
  }
  void generate_triangular_32(uint64_t count, float* out, float a, float b, float c)
  {
    CHECK_RND_ENGINE(::randutilGenerateTriangularEx(gen_, out, count, a, b, c));
  }
  void generate_weibull_64(uint64_t count, double* out, double lam, double k)
  {
    CHECK_RND_ENGINE(::randutilGenerateWeibullDoubleEx(gen_, out, count, lam, k));
  }
  void generate_weibull_32(uint64_t count, float* out, float lam, float k)
  {
    CHECK_RND_ENGINE(::randutilGenerateWeibullEx(gen_, out, count, lam, k));
  }
  void generate_beta_64(uint64_t count, double* out, double a, double b)
  {
    CHECK_RND_ENGINE(::randutilGenerateBetaDoubleEx(gen_, out, count, a, b));
  }
  void generate_beta_32(uint64_t count, float* out, float a, float b)
  {
    CHECK_RND_ENGINE(::randutilGenerateBetaEx(gen_, out, count, a, b));
  }
  void generate_f_64(uint64_t count, double* out, double dfnum, double dfden)
  {
    CHECK_RND_ENGINE(::randutilGenerateFisherSnedecorDoubleEx(gen_, out, count, dfnum, dfden));
  }
  void generate_f_32(uint64_t count, float* out, float dfnum, float dfden)
  {
    CHECK_RND_ENGINE(::randutilGenerateFisherSnedecorEx(gen_, out, count, dfnum, dfden));
  }
  void generate_logseries(uint64_t count, uint32_t* out, double p)
  {
    CHECK_RND_ENGINE(::randutilGenerateLogSeriesEx(gen_, out, count, p));
  }
  void generate_noncentral_f_64(
    uint64_t count, double* out, double dfnum, double dfden, double nonc)
  {
    CHECK_RND_ENGINE(
      ::randutilGenerateFisherSnedecorDoubleEx(gen_, out, count, dfnum, dfden, nonc));
  }
  void generate_noncentral_f_32(uint64_t count, float* out, float dfnum, float dfden, float nonc)
  {
    CHECK_RND_ENGINE(::randutilGenerateFisherSnedecorEx(gen_, out, count, dfnum, dfden, nonc));
  }
  void generate_chisquare_64(uint64_t count, double* out, double df, double nonc)
  {
    CHECK_RND_ENGINE(::randutilGenerateChiSquareDoubleEx(gen_, out, count, df, nonc));
  }
  void generate_chisquare_32(uint64_t count, float* out, float df, float nonc)
  {
    CHECK_RND_ENGINE(::randutilGenerateChiSquareEx(gen_, out, count, df, nonc));
  }
  void generate_gamma_64(uint64_t count, double* out, double k, double theta)
  {
    CHECK_RND_ENGINE(::randutilGenerateGammaDoubleEx(gen_, out, count, k, theta));
  }
  void generate_gamma_32(uint64_t count, float* out, float k, float theta)
  {
    CHECK_RND_ENGINE(::randutilGenerateGammaEx(gen_, out, count, k, theta));
  }
  void generate_standard_t_64(uint64_t count, double* out, double df)
  {
    CHECK_RND_ENGINE(::randutilGenerateStandardTDoubleEx(gen_, out, count, df));
  }
  void generate_standard_t_32(uint64_t count, float* out, float df)
  {
    CHECK_RND_ENGINE(::randutilGenerateStandardTEx(gen_, out, count, df));
  }
  void generate_hypergeometric(
    uint64_t count, uint32_t* out, int64_t ngood, int64_t nbad, int64_t nsample)
  {
    CHECK_RND_ENGINE(::randutilGenerateHyperGeometricEx(gen_, out, count, ngood, nbad, nsample));
  }
  void generate_vonmises_64(uint64_t count, double* out, double mu, double kappa)
  {
    CHECK_RND_ENGINE(::randutilGenerateVonMisesDoubleEx(gen_, out, count, mu, kappa));
  }
  void generate_vonmises_32(uint64_t count, float* out, float mu, float kappa)
  {
    CHECK_RND_ENGINE(::randutilGenerateVonMisesEx(gen_, out, count, mu, kappa));
  }
  void generate_zipf(uint64_t count, uint32_t* out, double a)
  {
    CHECK_RND_ENGINE(::randutilGenerateZipfEx(gen_, out, count, a));
  }
  void generate_geometric(uint64_t count, uint32_t* out, double p)
  {
    CHECK_RND_ENGINE(::randutilGenerateGeometricEx(gen_, out, count, p));
  }
  void generate_wald_64(uint64_t count, double* out, double mean, double scale)
  {
    CHECK_RND_ENGINE(::randutilGenerateWaldDoubleEx(gen_, out, count, mean, scale));
  }
  void generate_wald_32(uint64_t count, float* out, float mean, float scale)
  {
    CHECK_RND_ENGINE(::randutilGenerateWaldEx(gen_, out, count, mean, scale));
  }
  void generate_binomial(uint64_t count, uint32_t* out, uint32_t ntrials, double p)
  {
    CHECK_RND_ENGINE(::randutilGenerateBinomialEx(gen_, out, count, ntrials, p));
  }
  void generate_negative_binomial(uint64_t count, uint32_t* out, uint32_t ntrials, double p)
  {
    CHECK_RND_ENGINE(::randutilGenerateNegativeBinomialEx(gen_, out, count, ntrials, p));
  }
};

#pragma endregion

struct generate_fn {
  template <int32_t DIM>
  size_t operator()(CURANDGenerator& gen, legate::Store& output)
  {
    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    const auto proc = Processor::get_executing_processor();
    randutil_log().debug() << "proc=" << proc << " - shape = " << rect;

    if (volume > 0) {
      auto out = output.write_accessor<uint32_t, DIM>(rect);

      uint32_t* p = out.ptr(rect);

      gen.generate_raw(volume, p);
    }

    return volume;
  }
};

#pragma region generators

#pragma region integer

template <typename output_t>
struct integer_generator;
template <>
struct integer_generator<int64_t> {
  int64_t low_, high_;

  integer_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_(intparams[0]), high_(intparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, int64_t* p) const
  {
    gen.generate_integer_64(count, p, low_, high_);
  }
};
template <>
struct integer_generator<int32_t> {
  int32_t low_, high_;

  integer_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_((int32_t)intparams[0]), high_((int32_t)intparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, int32_t* p) const
  {
    gen.generate_integer_32(count, p, low_, high_);
  }
};
template <>
struct integer_generator<int16_t> {
  int16_t low_, high_;

  integer_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_((int16_t)intparams[0]), high_((int16_t)intparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, int16_t* p) const
  {
    gen.generate_integer_16(count, p, low_, high_);
  }
};

#pragma endregion

#pragma region uniform

template <typename output_t>
struct uniform_generator;
template <>
struct uniform_generator<double> {
  double low_, high_;  // high exclusive

  uniform_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_(doubleparams[0]), high_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_uniform_64(count, p, low_, high_);
  }
};
template <>
struct uniform_generator<float> {
  float low_, high_;

  uniform_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : low_(floatparams[0]), high_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_uniform_32(count, p, low_, high_);
  }
};

#pragma endregion

#pragma region lognormal

template <typename output_t>
struct lognormal_generator;
template <>
struct lognormal_generator<double> {
  double mean_, stdev_;

  lognormal_generator(const std::vector<int64_t>& intparams,
                      const std::vector<float>& floatparams,
                      const std::vector<double>& doubleparams)
    : mean_(doubleparams[0]), stdev_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_lognormal_64(count, p, mean_, stdev_);
  }
};
template <>
struct lognormal_generator<float> {
  float mean_, stdev_;

  lognormal_generator(const std::vector<int64_t>& intparams,
                      const std::vector<float>& floatparams,
                      const std::vector<double>& doubleparams)
    : mean_(floatparams[0]), stdev_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_lognormal_32(count, p, mean_, stdev_);
  }
};

#pragma endregion

#pragma region normal

template <typename output_t>
struct normal_generator;
template <>
struct normal_generator<double> {
  double mean_, stdev_;

  normal_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : mean_(doubleparams[0]), stdev_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_normal_64(count, p, mean_, stdev_);
  }
};
template <>
struct normal_generator<float> {
  float mean_, stdev_;

  normal_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : mean_(floatparams[0]), stdev_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_normal_32(count, p, mean_, stdev_);
  }
};

#pragma endregion

#pragma region poisson

template <typename output_t>
struct poisson_generator;
template <>
struct poisson_generator<uint32_t> {
  double lam_;

  poisson_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : lam_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, uint32_t* p) const
  {
    gen.generate_poisson(count, p, lam_);
  }
};

#pragma endregion

#pragma region exponential

template <typename output_t>
struct exponential_generator;
template <>
struct exponential_generator<double> {
  double scale_;

  exponential_generator(const std::vector<int64_t>& intparams,
                        const std::vector<float>& floatparams,
                        const std::vector<double>& doubleparams)
    : scale_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_exponential_64(count, p, scale_);
  }
};
template <>
struct exponential_generator<float> {
  float scale_;

  exponential_generator(const std::vector<int64_t>& intparams,
                        const std::vector<float>& floatparams,
                        const std::vector<double>& doubleparams)
    : scale_(floatparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_exponential_32(count, p, scale_);
  }
};

#pragma endregion

#pragma region gumbel

template <typename output_t>
struct gumbel_generator;
template <>
struct gumbel_generator<double> {
  double mu_, beta_;

  gumbel_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : mu_(doubleparams[0]), beta_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_gumbel_64(count, p, mu_, beta_);
  }
};
template <>
struct gumbel_generator<float> {
  float mu_, beta_;

  gumbel_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : mu_(floatparams[0]), beta_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_gumbel_32(count, p, mu_, beta_);
  }
};

#pragma endregion

#pragma region laplace

template <typename output_t>
struct laplace_generator;
template <>
struct laplace_generator<double> {
  double mu_, beta_;

  laplace_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : mu_(doubleparams[0]), beta_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_laplace_64(count, p, mu_, beta_);
  }
};
template <>
struct laplace_generator<float> {
  float mu_, beta_;

  laplace_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : mu_(floatparams[0]), beta_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_laplace_32(count, p, mu_, beta_);
  }
};

#pragma endregion

#pragma region logistic

template <typename output_t>
struct logistic_generator;
template <>
struct logistic_generator<double> {
  double mu_, beta_;

  logistic_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : mu_(doubleparams[0]), beta_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_logistic_64(count, p, mu_, beta_);
  }
};
template <>
struct logistic_generator<float> {
  float mu_, beta_;

  logistic_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : mu_(floatparams[0]), beta_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_logistic_32(count, p, mu_, beta_);
  }
};

#pragma endregion

#pragma region pareto

template <typename output_t>
struct pareto_generator;
template <>
struct pareto_generator<double> {
  double alpha_;

  pareto_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : alpha_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_pareto_64(count, p, alpha_);
  }
};
template <>
struct pareto_generator<float> {
  float alpha_;

  pareto_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : alpha_(floatparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_pareto_32(count, p, alpha_);
  }
};

#pragma endregion

#pragma region power

template <typename output_t>
struct power_generator;
template <>
struct power_generator<double> {
  double alpha_;

  power_generator(const std::vector<int64_t>& intparams,
                  const std::vector<float>& floatparams,
                  const std::vector<double>& doubleparams)
    : alpha_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_power_64(count, p, alpha_);
  }
};
template <>
struct power_generator<float> {
  float alpha_;

  power_generator(const std::vector<int64_t>& intparams,
                  const std::vector<float>& floatparams,
                  const std::vector<double>& doubleparams)
    : alpha_(floatparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_power_32(count, p, alpha_);
  }
};

#pragma endregion

#pragma region rayleigh

template <typename output_t>
struct rayleigh_generator;
template <>
struct rayleigh_generator<double> {
  double sigma_;

  rayleigh_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : sigma_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_rayleigh_64(count, p, sigma_);
  }
};
template <>
struct rayleigh_generator<float> {
  float sigma_;

  rayleigh_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : sigma_(floatparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_rayleigh_32(count, p, sigma_);
  }
};

#pragma endregion

#pragma region cauchy

template <typename output_t>
struct cauchy_generator;
template <>
struct cauchy_generator<double> {
  double x0_, gamma_;

  cauchy_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : x0_(doubleparams[0]), gamma_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_cauchy_64(count, p, x0_, gamma_);
  }
};
template <>
struct cauchy_generator<float> {
  float x0_, gamma_;

  cauchy_generator(const std::vector<int64_t>& intparams,
                   const std::vector<float>& floatparams,
                   const std::vector<double>& doubleparams)
    : x0_(floatparams[0]), gamma_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_cauchy_32(count, p, x0_, gamma_);
  }
};

#pragma endregion

#pragma region triangular

template <typename output_t>
struct triangular_generator;
template <>
struct triangular_generator<double> {
  double a_, b_, c_;

  triangular_generator(const std::vector<int64_t>& intparams,
                       const std::vector<float>& floatparams,
                       const std::vector<double>& doubleparams)
    : a_(doubleparams[0]), b_(doubleparams[1]), c_(doubleparams[2])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_triangular_64(count, p, a_, b_, c_);
  }
};
template <>
struct triangular_generator<float> {
  float a_, b_, c_;

  triangular_generator(const std::vector<int64_t>& intparams,
                       const std::vector<float>& floatparams,
                       const std::vector<double>& doubleparams)
    : a_(floatparams[0]), b_(floatparams[1]), c_(floatparams[2])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_triangular_32(count, p, a_, b_, c_);
  }
};

#pragma endregion

#pragma region weibull

template <typename output_t>
struct weibull_generator;
template <>
struct weibull_generator<double> {
  double lam_, k_;

  weibull_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : lam_(doubleparams[0]), k_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_weibull_64(count, p, lam_, k_);
  }
};
template <>
struct weibull_generator<float> {
  float lam_, k_;

  weibull_generator(const std::vector<int64_t>& intparams,
                    const std::vector<float>& floatparams,
                    const std::vector<double>& doubleparams)
    : lam_(floatparams[0]), k_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_weibull_32(count, p, lam_, k_);
  }
};

#pragma endregion

#pragma region bytes

template <typename output_t>
struct bytes_generator;
template <>
struct bytes_generator<unsigned char> {
  bytes_generator(const std::vector<int64_t>& intparams,
                  const std::vector<float>& floatparams,
                  const std::vector<double>& doubleparams)
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned char* p) const
  {
    // TODO/ verify assumption that allocation is rounded up...
    gen.generate_integer_32((count + 3) / 4, (int32_t*)p, -2147483648, 2147483647);
  }
};

#pragma endregion

#pragma region beta

template <typename output_t>
struct beta_generator;
template <>
struct beta_generator<double> {
  double a_, b_;

  beta_generator(const std::vector<int64_t>& intparams,
                 const std::vector<float>& floatparams,
                 const std::vector<double>& doubleparams)
    : a_(doubleparams[0]), b_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_beta_64(count, p, a_, b_);
  }
};
template <>
struct beta_generator<float> {
  float a_, b_;

  beta_generator(const std::vector<int64_t>& intparams,
                 const std::vector<float>& floatparams,
                 const std::vector<double>& doubleparams)
    : a_(floatparams[0]), b_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_beta_32(count, p, a_, b_);
  }
};

#pragma endregion

#pragma region f

template <typename output_t>
struct f_generator;
template <>
struct f_generator<double> {
  double dfnum_, dfden_;

  f_generator(const std::vector<int64_t>& intparams,
              const std::vector<float>& floatparams,
              const std::vector<double>& doubleparams)
    : dfnum_(doubleparams[0]), dfden_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_f_64(count, p, dfnum_, dfden_);
  }
};
template <>
struct f_generator<float> {
  float dfnum_, dfden_;

  f_generator(const std::vector<int64_t>& intparams,
              const std::vector<float>& floatparams,
              const std::vector<double>& doubleparams)
    : dfnum_(floatparams[0]), dfden_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_f_32(count, p, dfnum_, dfden_);
  }
};

#pragma endregion

#pragma region logseries

template <typename output_t>
struct logseries_generator;
template <>
struct logseries_generator<unsigned> {
  double p_;

  logseries_generator(const std::vector<int64_t>& intparams,
                      const std::vector<float>& floatparams,
                      const std::vector<double>& doubleparams)
    : p_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned* p) const
  {
    gen.generate_logseries(count, p, p_);
  }
};

#pragma endregion

#pragma region noncentral_f

template <typename output_t>
struct noncentral_f_generator;
template <>
struct noncentral_f_generator<double> {
  double dfnum_, dfden_, nonc_;

  noncentral_f_generator(const std::vector<int64_t>& intparams,
                         const std::vector<float>& floatparams,
                         const std::vector<double>& doubleparams)
    : dfnum_(doubleparams[0]), dfden_(doubleparams[1]), nonc_(doubleparams[2])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_noncentral_f_64(count, p, dfnum_, dfden_, nonc_);
  }
};
template <>
struct noncentral_f_generator<float> {
  float dfnum_, dfden_, nonc_;

  noncentral_f_generator(const std::vector<int64_t>& intparams,
                         const std::vector<float>& floatparams,
                         const std::vector<double>& doubleparams)
    : dfnum_(floatparams[0]), dfden_(floatparams[1]), nonc_(floatparams[2])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_noncentral_f_32(count, p, dfnum_, dfden_, nonc_);
  }
};

#pragma endregion

#pragma region chisquare

template <typename output_t>
struct chisquare_generator;
template <>
struct chisquare_generator<double> {
  double df_, nonc_;

  chisquare_generator(const std::vector<int64_t>& intparams,
                      const std::vector<float>& floatparams,
                      const std::vector<double>& doubleparams)
    : df_(doubleparams[0]), nonc_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_chisquare_64(count, p, df_, nonc_);
  }
};
template <>
struct chisquare_generator<float> {
  float df_, nonc_;

  chisquare_generator(const std::vector<int64_t>& intparams,
                      const std::vector<float>& floatparams,
                      const std::vector<double>& doubleparams)
    : df_(floatparams[0]), nonc_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_chisquare_32(count, p, df_, nonc_);
  }
};

#pragma endregion

#pragma region gamma

template <typename output_t>
struct gamma_generator;
template <>
struct gamma_generator<double> {
  double k_, theta_;

  gamma_generator(const std::vector<int64_t>& intparams,
                  const std::vector<float>& floatparams,
                  const std::vector<double>& doubleparams)
    : k_(doubleparams[0]), theta_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_gamma_64(count, p, k_, theta_);
  }
};
template <>
struct gamma_generator<float> {
  float k_, theta_;

  gamma_generator(const std::vector<int64_t>& intparams,
                  const std::vector<float>& floatparams,
                  const std::vector<double>& doubleparams)
    : k_(floatparams[0]), theta_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_gamma_32(count, p, k_, theta_);
  }
};

#pragma endregion

#pragma region hypergeometric

template <typename output_t>
struct hypergeometric_generator;
template <>
struct hypergeometric_generator<unsigned> {
  int64_t ngood_, nbad_, nsample_;

  hypergeometric_generator(const std::vector<int64_t>& intparams,
                           const std::vector<float>& floatparams,
                           const std::vector<double>& doubleparams)
    : ngood_(intparams[0]), nbad_(intparams[1]), nsample_(intparams[2])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned* p) const
  {
    gen.generate_hypergeometric(count, p, ngood_, nbad_, nsample_);
  }
};

#pragma endregion

#pragma region zipf

template <typename output_t>
struct zipf_generator;
template <>
struct zipf_generator<unsigned> {
  double a_;

  zipf_generator(const std::vector<int64_t>& intparams,
                 const std::vector<float>& floatparams,
                 const std::vector<double>& doubleparams)
    : a_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned* p) const
  {
    gen.generate_zipf(count, p, a_);
  }
};

#pragma endregion

#pragma region geometric

template <typename output_t>
struct geometric_generator;
template <>
struct geometric_generator<unsigned> {
  double p_;

  geometric_generator(const std::vector<int64_t>& intparams,
                      const std::vector<float>& floatparams,
                      const std::vector<double>& doubleparams)
    : p_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned* p) const
  {
    gen.generate_geometric(count, p, p_);
  }
};

#pragma endregion

#pragma region standard_t

template <typename output_t>
struct standard_t_generator;
template <>
struct standard_t_generator<double> {
  double df_;

  standard_t_generator(const std::vector<int64_t>& intparams,
                       const std::vector<float>& floatparams,
                       const std::vector<double>& doubleparams)
    : df_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_standard_t_64(count, p, df_);
  }
};
template <>
struct standard_t_generator<float> {
  float df_;

  standard_t_generator(const std::vector<int64_t>& intparams,
                       const std::vector<float>& floatparams,
                       const std::vector<double>& doubleparams)
    : df_(floatparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_standard_t_32(count, p, df_);
  }
};

#pragma endregion

#pragma region vonmises

template <typename output_t>
struct vonmises_generator;
template <>
struct vonmises_generator<double> {
  double mu_, kappa_;

  vonmises_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : mu_(doubleparams[0]), kappa_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_vonmises_64(count, p, mu_, kappa_);
  }
};
template <>
struct vonmises_generator<float> {
  float mu_, kappa_;

  vonmises_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : mu_(floatparams[0]), kappa_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_vonmises_32(count, p, mu_, kappa_);
  }
};

#pragma endregion

#pragma region wald

template <typename output_t>
struct wald_generator;
template <>
struct wald_generator<double> {
  double mean_, scale_;

  wald_generator(const std::vector<int64_t>& intparams,
                 const std::vector<float>& floatparams,
                 const std::vector<double>& doubleparams)
    : mean_(doubleparams[0]), scale_(doubleparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, double* p) const
  {
    gen.generate_wald_64(count, p, mean_, scale_);
  }
};
template <>
struct wald_generator<float> {
  float mean_, scale_;

  wald_generator(const std::vector<int64_t>& intparams,
                 const std::vector<float>& floatparams,
                 const std::vector<double>& doubleparams)
    : mean_(floatparams[0]), scale_(floatparams[1])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, float* p) const
  {
    gen.generate_wald_32(count, p, mean_, scale_);
  }
};

#pragma endregion

#pragma region binomial

template <typename output_t>
struct binomial_generator;
template <>
struct binomial_generator<uint32_t> {
  uint32_t n_;
  double p_;

  binomial_generator(const std::vector<int64_t>& intparams,
                     const std::vector<float>& floatparams,
                     const std::vector<double>& doubleparams)
    : n_(intparams[0]), p_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned* p) const
  {
    gen.generate_binomial(count, p, n_, p_);
  }
};

#pragma endregion

#pragma region negative_binomial

template <typename output_t>
struct negative_binomial_generator;
template <>
struct negative_binomial_generator<uint32_t> {
  uint32_t n_;
  double p_;

  negative_binomial_generator(const std::vector<int64_t>& intparams,
                              const std::vector<float>& floatparams,
                              const std::vector<double>& doubleparams)
    : n_(intparams[0]), p_(doubleparams[0])
  {
  }

  void generate(CURANDGenerator& gen, uint64_t count, unsigned* p) const
  {
    gen.generate_negative_binomial(count, p, n_, p_);
  }
};

#pragma endregion

#pragma endregion

template <typename output_t, typename generator_t>
struct generate_distribution {
  const generator_t& generator_;

  generate_distribution(const generator_t& generator) : generator_(generator) {}

  template <int32_t DIM>
  size_t operator()(CURANDGenerator& gen, legate::Store& output)
  {
    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    const auto proc = Processor::get_executing_processor();
    randutil_log().debug() << "proc=" << proc << " - shape = " << rect;

    if (volume > 0) {
      auto out = output.write_accessor<output_t, DIM>(rect);

      output_t* p = out.ptr(rect);

      generator_.generate(gen, volume, p);
    }

    return volume;
  }

  static void generate(legate::Store& res,
                       CURANDGenerator& cugen,
                       const std::vector<int64_t>& intparams,
                       const std::vector<float>& floatparams,
                       const std::vector<double>& doubleparams)
  {
    generator_t dist_gen(intparams, floatparams, doubleparams);
    generate_distribution<output_t, generator_t> generate_func(dist_gen);
    dim_dispatch(res.dim(), generate_func, cugen, res);
  }
};

template <VariantKind kind>
struct generator_map {
  generator_map() {}
  ~generator_map()
  {
    if (m_generators.size() != 0) {
      randutil_log().debug() << "some generators have not been freed - cleaning-up !";
      // actually destroy
      for (auto kv = m_generators.begin(); kv != m_generators.end(); ++kv) {
        auto cugenptr = kv->second;
        CURANDGeneratorBuilder<kind>::destroy(cugenptr);
      }
      m_generators.clear();
    }
  }

  std::map<uint32_t, CURANDGenerator*> m_generators;

  bool has(uint32_t generatorID) { return m_generators.find(generatorID) != m_generators.end(); }

  CURANDGenerator* get(uint32_t generatorID)
  {
    if (m_generators.find(generatorID) == m_generators.end()) {
      randutil_log().fatal() << "internal error : generator ID <" << generatorID
                             << "> does not exist (get) !";
      assert(false);
    }
    return m_generators[generatorID];
  }

  // called by the processor later using the generator
  void create(uint32_t generatorID, BitGeneratorType gentype, uint64_t seed, uint32_t flags)
  {
    const auto proc = Processor::get_executing_processor();
    CURANDGenerator* cugenptr =
      CURANDGeneratorBuilder<kind>::build(gentype, seed, (uint64_t)proc.id, flags);

    // safety check
    if (m_generators.find(generatorID) != m_generators.end()) {
      randutil_log().fatal() << "internal error : generator ID <" << generatorID
                             << "> already in use !";
      assert(false);
    }
    m_generators[generatorID] = cugenptr;
  }

  void destroy(uint32_t generatorID)
  {
    CURANDGenerator* cugenptr;
    // verify it existed, and otherwise remove it from list
    {
      if (m_generators.find(generatorID) != m_generators.end()) {
        cugenptr = m_generators[generatorID];
        m_generators.erase(generatorID);
      } else
        // in some cases, destroy is forced, but processor never created the instance
        return;
    }

    CURANDGeneratorBuilder<kind>::destroy(cugenptr);
  }
};

template <VariantKind kind>
struct BitGeneratorImplBody {
  using generator_map_t = generator_map<kind>;

  static std::mutex lock_generators;
  static std::map<Processor, std::unique_ptr<generator_map_t>> m_generators;

 private:
  static generator_map_t& get_generator_map()
  {
    const auto proc = Processor::get_executing_processor();
    std::lock_guard<std::mutex> guard(lock_generators);
    if (m_generators.find(proc) == m_generators.end()) {
      m_generators[proc] = std::make_unique<generator_map_t>();
    }
    generator_map_t* res = m_generators[proc].get();
    return *res;
  }

 public:
  void operator()(BitGeneratorOperation op,
                  int32_t generatorID,
                  BitGeneratorType generatorType,  // to allow for lazy initialization,
                                                   // generatorType is always passed
                  uint64_t seed,   // to allow for lazy initialization, seed is always passed
                  uint32_t flags,  // for future use - ordering, etc.
                  BitGeneratorDistribution distribution,
                  const DomainPoint& strides,
                  std::vector<int64_t> intparams,
                  std::vector<float> floatparams,
                  std::vector<double> doubleparams,
                  std::vector<legate::Store>& output,
                  std::vector<legate::Store>& args)
  {
    generator_map_t& genmap = get_generator_map();
    switch (op) {
      case BitGeneratorOperation::CREATE: {
        genmap.create(generatorID, generatorType, seed, flags);

        randutil_log().debug() << "created generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::DESTROY: {
        genmap.destroy(generatorID);

        randutil_log().debug() << "destroyed generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::RAND_RAW: {
        // allow for lazy initialization
        if (!genmap.has(generatorID)) genmap.create(generatorID, generatorType, seed, flags);
        // get the generator
        CURANDGenerator* genptr = genmap.get(generatorID);
        if (output.size() != 0) {
          legate::Store& res     = output[0];
          CURANDGenerator& cugen = *genptr;
          dim_dispatch(res.dim(), generate_fn{}, cugen, res);
        }
        break;
      }
      case BitGeneratorOperation::DISTRIBUTION: {
        // allow for lazy initialization
        if (!genmap.has(generatorID)) genmap.create(generatorID, generatorType, seed, flags);
        // get the generator
        CURANDGenerator* genptr = genmap.get(generatorID);
        if (output.size() != 0) {
          legate::Store& res     = output[0];
          CURANDGenerator& cugen = *genptr;
          switch (distribution) {
            case BitGeneratorDistribution::INTEGERS_16:
              generate_distribution<int16_t, integer_generator<int16_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::INTEGERS_32:
              generate_distribution<int32_t, integer_generator<int32_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::INTEGERS_64:
              generate_distribution<int64_t, integer_generator<int64_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::UNIFORM_32:
              generate_distribution<float, uniform_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::UNIFORM_64:
              generate_distribution<double, uniform_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LOGNORMAL_32:
              generate_distribution<float, lognormal_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LOGNORMAL_64:
              generate_distribution<double, lognormal_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::NORMAL_32:
              generate_distribution<float, normal_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::NORMAL_64:
              generate_distribution<double, normal_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::POISSON:
              generate_distribution<uint32_t, poisson_generator<uint32_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::EXPONENTIAL_32:
              generate_distribution<float, exponential_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::EXPONENTIAL_64:
              generate_distribution<double, exponential_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::GUMBEL_32:
              generate_distribution<float, gumbel_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::GUMBEL_64:
              generate_distribution<double, gumbel_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LAPLACE_32:
              generate_distribution<float, laplace_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LAPLACE_64:
              generate_distribution<double, laplace_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LOGISTIC_32:
              generate_distribution<float, logistic_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LOGISTIC_64:
              generate_distribution<double, logistic_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::PARETO_32:
              generate_distribution<float, pareto_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::PARETO_64:
              generate_distribution<double, pareto_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::POWER_32:
              generate_distribution<float, power_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::POWER_64:
              generate_distribution<double, power_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::RAYLEIGH_32:
              generate_distribution<float, rayleigh_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::RAYLEIGH_64:
              generate_distribution<double, rayleigh_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::CAUCHY_32:
              generate_distribution<float, cauchy_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::CAUCHY_64:
              generate_distribution<double, cauchy_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::TRIANGULAR_32:
              generate_distribution<float, triangular_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::TRIANGULAR_64:
              generate_distribution<double, triangular_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::WEIBULL_32:
              generate_distribution<float, weibull_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::WEIBULL_64:
              generate_distribution<double, weibull_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::BYTES:
              generate_distribution<unsigned char, bytes_generator<unsigned char>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::BETA_32:
              generate_distribution<float, beta_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::BETA_64:
              generate_distribution<double, beta_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::F_32:
              generate_distribution<float, f_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::F_64:
              generate_distribution<double, f_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::LOGSERIES:
              generate_distribution<uint32_t, logseries_generator<unsigned>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::NONCENTRAL_F_32:
              generate_distribution<float, noncentral_f_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::NONCENTRAL_F_64:
              generate_distribution<double, noncentral_f_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::CHISQUARE_32:
              generate_distribution<float, chisquare_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::CHISQUARE_64:
              generate_distribution<double, chisquare_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::GAMMA_32:
              generate_distribution<float, gamma_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::GAMMA_64:
              generate_distribution<double, gamma_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::STANDARD_T_32:
              generate_distribution<float, standard_t_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::STANDARD_T_64:
              generate_distribution<double, standard_t_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::VONMISES_32:
              generate_distribution<float, vonmises_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::VONMISES_64:
              generate_distribution<double, vonmises_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::HYPERGEOMETRIC:
              generate_distribution<uint32_t, hypergeometric_generator<unsigned>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::ZIPF:
              generate_distribution<uint32_t, zipf_generator<unsigned>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::GEOMETRIC:
              generate_distribution<uint32_t, geometric_generator<unsigned>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::WALD_32:
              generate_distribution<float, wald_generator<float>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::WALD_64:
              generate_distribution<double, wald_generator<double>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::BINOMIAL:
              generate_distribution<uint32_t, binomial_generator<uint32_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            case BitGeneratorDistribution::NEGATIVE_BINOMIAL:
              generate_distribution<uint32_t, negative_binomial_generator<uint32_t>>::generate(
                res, cugen, intparams, floatparams, doubleparams);
              break;
            default: LEGATE_ABORT;
          }
        }
        break;
      }
      default: LEGATE_ABORT;
    }
  }
};

}  // namespace cunumeric
