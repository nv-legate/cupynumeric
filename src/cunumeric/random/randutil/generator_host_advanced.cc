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

// MacOS host variant:
//
#if defined(__APPLE__) && defined(__MACH__)
#define USE_STL_RANDOM_ENGINE_
#endif

#include "generator.h"

#pragma region beta

#include "generator_beta.inl"

extern "C" rnd_status_t randutilGenerateBetaEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  beta_t<float> func;
  func.a = a;
  func.b = b;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateBetaDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  beta_t<double> func;
  func.a = a;
  func.b = b;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region FisherSnedecor

#include "generator_f.inl"

extern "C" rnd_status_t randutilGenerateFisherSnedecorEx(randutilGenerator_t generator,
                                                         float* outputPtr,
                                                         size_t n,
                                                         float dfnum,
                                                         float dfden,
                                                         float nonc)  // 0.0f is F distribution
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  if (nonc == 0.0f) {
    f_t<float> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
  } else {
    noncentralf_t<float> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    func.nonc  = nonc;
    return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
  }
}

extern "C" rnd_status_t randutilGenerateFisherSnedecorDoubleEx(
  randutilGenerator_t generator,
  double* outputPtr,
  size_t n,
  double dfnum,
  double dfden,
  double nonc)  // 0.0 is F distribution
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  if (nonc == 0.0) {
    f_t<double> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
  } else {
    noncentralf_t<double> func;
    func.dfnum = dfnum;
    func.dfden = dfden;
    func.nonc  = nonc;
    return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
  }
}

#pragma endregion

#pragma region logseries

#include "generator_logseries.inl"

extern "C" rnd_status_t randutilGenerateLogSeriesEx(randutilGenerator_t generator,
                                                    uint32_t* outputPtr,
                                                    size_t n,
                                                    double p)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  logseries_t<double> func;
  func.p = p;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region ChiSquared

#include "generator_chisquare.inl"

extern "C" rnd_status_t randutilGenerateChiSquareEx(
  randutilGenerator_t generator,
  float* outputPtr,
  size_t n,
  float df,
  float nonc)  // <> 0.0f is non-central distribution
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  if (nonc == 0.0f) {
    chisquare_t<float> func;
    func.df = df;
    return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
  } else {
    noncentralchisquare_t<float> func;
    func.df   = df;
    func.nonc = nonc;
    return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
  }
}

extern "C" rnd_status_t randutilGenerateChiSquareDoubleEx(
  randutilGenerator_t generator,
  double* outputPtr,
  size_t n,
  double df,
  double nonc)  // <> 0.0 is non-central distribution
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  if (nonc == 0.0) {
    chisquare_t<double> func;
    func.df = df;
    return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
  } else {
    noncentralchisquare_t<double> func;
    func.df   = df;
    func.nonc = nonc;
    return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
  }
}

#pragma endregion

#pragma region gamma

#include "generator_gamma.inl"

extern "C" rnd_status_t randutilGenerateGammaEx(randutilGenerator_t generator,
                                                float* outputPtr,
                                                size_t n,
                                                float shape,
                                                float scale)  // = 1.0f is standard_gamma
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gamma_t<float> func;
  func.shape = shape;
  func.scale = scale;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateGammaDoubleEx(randutilGenerator_t generator,
                                                      double* outputPtr,
                                                      size_t n,
                                                      double shape,
                                                      double scale)  // = 1.0 is standard_gamma
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gamma_t<double> func;
  func.shape = shape;
  func.scale = scale;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region standart_t

#include "generator_standard_t.inl"

extern "C" rnd_status_t randutilGenerateStandardTEx(randutilGenerator_t generator,
                                                    float* outputPtr,
                                                    size_t n,
                                                    float df)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  standard_t_t<float> func;
  func.df = df;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateStandardTDoubleEx(randutilGenerator_t generator,
                                                          double* outputPtr,
                                                          size_t n,
                                                          double df)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  standard_t_t<double> func;
  func.df = df;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region von mises

#include "generator_vonmises.inl"

extern "C" rnd_status_t randutilGenerateVonMisesEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float kappa)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  vonmises_t<float> func;
  func.mu    = mu;
  func.kappa = kappa;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateVonMisesDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double kappa)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  vonmises_t<double> func;
  func.mu    = mu;
  func.kappa = kappa;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region hypergeometric

#include "generator_hypergeometric.inl"

extern "C" rnd_status_t randutilGenerateHyperGeometricEx(randutilGenerator_t generator,
                                                         uint32_t* outputPtr,
                                                         size_t n,
                                                         int64_t ngood,
                                                         int64_t nbad,
                                                         int64_t nsample)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  hypergeometric_t<int64_t> func;
  func.ngood   = ngood;
  func.nbad    = nbad;
  func.nsample = nsample;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region zipf

#include "generator_zipf.inl"

extern "C" rnd_status_t randutilGenerateZipfEx(randutilGenerator_t generator,
                                               uint32_t* outputPtr,
                                               size_t n,
                                               double a)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  zipf_t<double> func;
  func.a = a;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region geometric

#include "generator_geometric.inl"

extern "C" rnd_status_t randutilGenerateGeometricEx(randutilGenerator_t generator,
                                                    uint32_t* outputPtr,
                                                    size_t n,
                                                    double p)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  geometric_t<double> func;
  func.p = p;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region wald

#include "generator_wald.inl"

extern "C" rnd_status_t randutilGenerateWaldEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float lambda)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  wald_t<float> func;
  func.mu     = mu;
  func.lambda = lambda;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateWaldDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double lambda)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  wald_t<double> func;
  func.mu     = mu;
  func.lambda = lambda;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region binomial

#include "generator_binomial.inl"

extern "C" rnd_status_t randutilGenerateBinomialEx(
  randutilGenerator_t generator, uint32_t* outputPtr, size_t n, uint32_t ntrials, double p)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  binomial_t<uint32_t> func;
  func.n = ntrials;
  func.p = p;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region negative binomial

#include "generator_negative_binomial.inl"

extern "C" rnd_status_t randutilGenerateNegativeBinomialEx(
  randutilGenerator_t generator, uint32_t* outputPtr, size_t n, uint32_t ntrials, double p)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  negative_binomial_t<uint32_t> func;
  func.n = ntrials;
  func.p = p;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion
