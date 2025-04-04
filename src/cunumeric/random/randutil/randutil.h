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

#include <cstdint>
// #include <curand.h>

#include "cunumeric/random/rnd_aliases.h"

typedef void* randutilGenerator_t;

/* generator */

// CUDA-ONLY API
#ifdef LEGATE_USE_CUDA
extern "C" rnd_status_t randutilCreateGenerator(randutilGenerator_t* generator,
                                                randRngType_t rng_type,
                                                uint64_t seed,
                                                uint64_t generatorID,
                                                uint32_t flags,
                                                stream_t stream);
#endif

extern "C" rnd_status_t randutilCreateGeneratorHost(randutilGenerator_t* generator,
                                                    randRngType_t rng_type,
                                                    uint64_t seed,
                                                    uint64_t generatorID,
                                                    uint32_t flags);
extern "C" rnd_status_t randutilDestroyGenerator(randutilGenerator_t generator);

/* curand distributions */

extern "C" rnd_status_t randutilGenerateIntegers16(randutilGenerator_t generator,
                                                   int16_t* outputPtr,
                                                   size_t num,
                                                   int16_t low /* inclusive */,
                                                   int16_t high /* exclusive */);

extern "C" rnd_status_t randutilGenerateIntegers32(randutilGenerator_t generator,
                                                   int32_t* outputPtr,
                                                   size_t num,
                                                   int32_t low /* inclusive */,
                                                   int32_t high /* exclusive */);
extern "C" rnd_status_t randutilGenerateIntegers64(randutilGenerator_t generator,
                                                   int64_t* outputPtr,
                                                   size_t num,
                                                   int64_t low /* inclusive */,
                                                   int64_t high /* exclusive */);
extern "C" rnd_status_t randutilGenerateRawUInt32(randutilGenerator_t generator,
                                                  uint32_t* outputPtr,
                                                  size_t num);

extern "C" rnd_status_t randutilGenerateUniformEx(randutilGenerator_t generator,
                                                  float* outputPtr,
                                                  size_t num,
                                                  float low  = 0.0f, /* inclusive */
                                                  float high = 1.0f /* exclusive */);

extern "C" rnd_status_t randutilGenerateUniformDoubleEx(randutilGenerator_t generator,
                                                        double* outputPtr,
                                                        size_t num,
                                                        double low  = 0.0, /* inclusive */
                                                        double high = 1.0 /* exclusive */);

extern "C" rnd_status_t randutilGenerateLogNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" rnd_status_t randutilGenerateLogNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" rnd_status_t randutilGenerateNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
extern "C" rnd_status_t randutilGenerateNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);

extern "C" rnd_status_t randutilGeneratePoissonEx(randutilGenerator_t generator,
                                                  uint32_t* outputPtr,
                                                  size_t n,
                                                  double lambda);

/* Straightforward Distributions */

extern "C" rnd_status_t randutilGenerateExponentialEx(randutilGenerator_t generator,
                                                      float* outputPtr,
                                                      size_t n,
                                                      float scale);
extern "C" rnd_status_t randutilGenerateExponentialDoubleEx(randutilGenerator_t generator,
                                                            double* outputPtr,
                                                            size_t n,
                                                            double scale);

extern "C" rnd_status_t randutilGenerateGumbelEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" rnd_status_t randutilGenerateGumbelDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" rnd_status_t randutilGenerateLaplaceEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" rnd_status_t randutilGenerateLaplaceDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" rnd_status_t randutilGenerateLogisticEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float beta);
extern "C" rnd_status_t randutilGenerateLogisticDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double beta);

extern "C" rnd_status_t randutilGenerateParetoEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float xm, float alpha);
extern "C" rnd_status_t randutilGenerateParetoDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double xm, double alpha);

extern "C" rnd_status_t randutilGeneratePowerEx(randutilGenerator_t generator,
                                                float* outputPtr,
                                                size_t n,
                                                float alpha);
extern "C" rnd_status_t randutilGeneratePowerDoubleEx(randutilGenerator_t generator,
                                                      double* outputPtr,
                                                      size_t n,
                                                      double alpha);

extern "C" rnd_status_t randutilGenerateRayleighEx(randutilGenerator_t generator,
                                                   float* outputPtr,
                                                   size_t n,
                                                   float sigma);
extern "C" rnd_status_t randutilGenerateRayleighDoubleEx(randutilGenerator_t generator,
                                                         double* outputPtr,
                                                         size_t n,
                                                         double sigma);

extern "C" rnd_status_t randutilGenerateCauchyEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float x0, float gamma);
extern "C" rnd_status_t randutilGenerateCauchyDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double x0, double gamma);

extern "C" rnd_status_t randutilGenerateTriangularEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b, float c);
extern "C" rnd_status_t randutilGenerateTriangularDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b, double c);

extern "C" rnd_status_t randutilGenerateWeibullEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float lam, float k);
extern "C" rnd_status_t randutilGenerateWeibullDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double lam, double k);

/* more advanced distributions */

extern "C" rnd_status_t randutilGenerateBetaEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float a, float b);
extern "C" rnd_status_t randutilGenerateBetaDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double a, double b);

extern "C" rnd_status_t randutilGenerateFisherSnedecorEx(
  randutilGenerator_t generator,
  float* outputPtr,
  size_t n,
  float dfnum,
  float dfden,
  float nonc = 0.0f);  // 0.0f is F distribution
extern "C" rnd_status_t randutilGenerateFisherSnedecorDoubleEx(
  randutilGenerator_t generator,
  double* outputPtr,
  size_t n,
  double dfnum,
  double dfden,
  double nonc = 0.0);  // 0.0 is F distribution

extern "C" rnd_status_t randutilGenerateLogSeriesEx(randutilGenerator_t generator,
                                                    uint32_t* outputPtr,
                                                    size_t n,
                                                    double p);

extern "C" rnd_status_t randutilGenerateChiSquareEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float df, float nonc = 0.0);
extern "C" rnd_status_t randutilGenerateChiSquareDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double df, double nonc = 0.0);

extern "C" rnd_status_t randutilGenerateGammaEx(
  randutilGenerator_t generator,
  float* outputPtr,
  size_t n,
  float shape,
  float scale = 1.0f);  // scale = 1.0 is standard_gamma
extern "C" rnd_status_t randutilGenerateGammaDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double shape, double scale = 1.0);

extern "C" rnd_status_t randutilGenerateStandardTDoubleEx(randutilGenerator_t generator,
                                                          double* outputPtr,
                                                          size_t n,
                                                          double df);
extern "C" rnd_status_t randutilGenerateStandardTEx(randutilGenerator_t generator,
                                                    float* outputPtr,
                                                    size_t n,
                                                    float df);
extern "C" rnd_status_t randutilGenerateHyperGeometricEx(randutilGenerator_t generator,
                                                         uint32_t* outputPtr,
                                                         size_t n,
                                                         int64_t ngood,
                                                         int64_t nbad,
                                                         int64_t nsample);
extern "C" rnd_status_t randutilGenerateVonMisesDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double kappa);
extern "C" rnd_status_t randutilGenerateVonMisesEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float kappa);
extern "C" rnd_status_t randutilGenerateZipfEx(randutilGenerator_t generator,
                                               uint32_t* outputPtr,
                                               size_t n,
                                               double a);
extern "C" rnd_status_t randutilGenerateGeometricEx(randutilGenerator_t generator,
                                                    uint32_t* outputPtr,
                                                    size_t n,
                                                    double p);
extern "C" rnd_status_t randutilGenerateWaldDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mu, double lambda);
extern "C" rnd_status_t randutilGenerateWaldEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mu, float lambda);

extern "C" rnd_status_t randutilGenerateBinomialEx(
  randutilGenerator_t generator, uint32_t* outputPtr, size_t n, uint32_t ntrials, double p);
extern "C" rnd_status_t randutilGenerateNegativeBinomialEx(
  randutilGenerator_t generator, uint32_t* outputPtr, size_t n, uint32_t ntrials, double p);
