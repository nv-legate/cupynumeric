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

#pragma region exponential

#include "generator_exponential.inl"

extern "C" rnd_status_t randutilGenerateExponentialEx(randutilGenerator_t generator,
                                                      float* outputPtr,
                                                      size_t n,
                                                      float scale)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  exponential_t<float> func;
  func.scale = scale;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateExponentialDoubleEx(randutilGenerator_t generator,
                                                            double* outputPtr,
                                                            size_t n,
                                                            double scale)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  exponential_t<double> func;
  func.scale = scale;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region gumbel

#include "generator_gumbel.inl"

extern "C" rnd_status_t randutilGenerateGumbelEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gumbel_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateGumbelDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gumbel_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region laplace

#include "generator_laplace.inl"

extern "C" rnd_status_t randutilGenerateLaplaceEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  laplace_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateLaplaceDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  laplace_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region logistic

#include "generator_logistic.inl"

extern "C" rnd_status_t randutilGenerateLogisticEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  logistic_t<float> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateLogisticDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  logistic_t<double> func;
  func.mu   = mu;
  func.beta = beta;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region pareto

#include "generator_pareto.inl"

extern "C" rnd_status_t randutilGenerateParetoEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float xm, float alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  pareto_t<float> func;
  func.xm       = xm;
  func.invalpha = 1.0f / alpha;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateParetoDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double xm, double alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  pareto_t<double> func;
  func.xm       = xm;
  func.invalpha = 1.0 / alpha;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region power

#include "generator_power.inl"

extern "C" rnd_status_t randutilGeneratePowerEx(randutilGenerator_t generator,
                                                float* outputPtr,
                                                size_t n,
                                                float alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  power_t<float> func;
  func.invalpha = 1.0f / alpha;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGeneratePowerDoubleEx(randutilGenerator_t generator,
                                                      double* outputPtr,
                                                      size_t n,
                                                      double alpha)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  power_t<double> func;
  func.invalpha = 1.0 / alpha;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region rayleigh

#include "generator_rayleigh.inl"

extern "C" rnd_status_t randutilGenerateRayleighEx(randutilGenerator_t generator,
                                                   float* outputPtr,
                                                   size_t n,
                                                   float sigma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  rayleigh_t<float> func;
  func.sigma = sigma;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateRayleighDoubleEx(randutilGenerator_t generator,
                                                         double* outputPtr,
                                                         size_t n,
                                                         double sigma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  rayleigh_t<double> func;
  func.sigma = sigma;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region cauchy

#include "generator_cauchy.inl"

extern "C" rnd_status_t randutilGenerateCauchyEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float x0, float gamma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  cauchy_t<float> func;
  func.x0    = x0;
  func.gamma = gamma;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateCauchyDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double x0, double gamma)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  cauchy_t<double> func;
  func.x0    = x0;
  func.gamma = gamma;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region triangular

#include "generator_triangular.inl"

extern "C" rnd_status_t randutilGenerateTriangularEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b, float c)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  triangular_t<float> func;
  func.a = a;
  func.b = b;
  func.c = c;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateTriangularDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b, double c)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  triangular_t<double> func;
  func.a = a;
  func.b = b;
  func.c = c;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region weibull

#include "generator_weibull.inl"

extern "C" rnd_status_t randutilGenerateWeibullEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float lam, float k)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  weibull_t<float> func;
  func.lambda = lam;
  func.invk   = 1.0f / k;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateWeibullDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double lam, double k)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  weibull_t<double> func;
  func.lambda = lam;
  func.invk   = 1.0 / k;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion
