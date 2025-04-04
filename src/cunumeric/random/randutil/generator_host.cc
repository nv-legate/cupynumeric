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
#include "generator_create.inl"
#include "cunumeric/random/rnd_aliases.h"

#if !defined(LEGATE_USE_CUDA)
// the host code of cuRAND try to extern these variables out of nowhere,
// so we need to define them somewhere.
const dim3 blockDim{};
const uint3 threadIdx{};
#endif

extern "C" rnd_status_t randutilCreateGeneratorHost(randutilGenerator_t* generator,
                                                    randRngType_t rng_type,
                                                    uint64_t seed,
                                                    uint64_t generatorID,
                                                    uint32_t flags)
{
  return inner_randutilCreateGenerator<randutilimpl::execlocation::HOST>(
    generator, rng_type, seed, generatorID, nullptr);
}

extern "C" rnd_status_t randutilDestroyGenerator(randutilGenerator_t generator)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  gen->destroy();
  delete gen;
  return RND_STATUS_SUCCESS;
}

#pragma region integers

#include "generator_integers.inl"

extern "C" rnd_status_t randutilGenerateIntegers16(
  randutilGenerator_t generator, int16_t* outputPtr, size_t n, int16_t low, int16_t high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  integers<int16_t> func;
  func.from = low;
  func.to   = high;
  return randutilimpl::dispatch<decltype(func), int16_t>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateIntegers32(
  randutilGenerator_t generator, int32_t* outputPtr, size_t n, int32_t low, int32_t high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  integers<int32_t> func;
  func.from = low;
  func.to   = high;
  return randutilimpl::dispatch<decltype(func), int32_t>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateIntegers64(
  randutilGenerator_t generator, int64_t* outputPtr, size_t n, int64_t low, int64_t high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  integers<int64_t> func;
  func.from = low;
  func.to   = high;
  return randutilimpl::dispatch<decltype(func), int64_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region lognormal

#include "generator_lognormal.inl"

extern "C" rnd_status_t randutilGenerateLogNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  lognormal_t<float> func;
  func.mean   = mean;
  func.stddev = stddev;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateLogNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  lognormal_t<double> func;
  func.mean   = mean;
  func.stddev = stddev;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region normal

#include "generator_normal.inl"

extern "C" rnd_status_t randutilGenerateNormalEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  normal_t<float> func;
  func.mean   = mean;
  func.stddev = stddev;
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateNormalDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  normal_t<double> func;
  func.mean   = mean;
  func.stddev = stddev;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region poisson

#include "generator_poisson.inl"

extern "C" rnd_status_t randutilGeneratePoissonEx(randutilGenerator_t generator,
                                                  uint32_t* outputPtr,
                                                  size_t n,
                                                  double lambda)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  poisson func;
  func.lambda = lambda;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region raw

#include "generator_raw.inl"

extern "C" rnd_status_t randutilGenerateRawUInt32(randutilGenerator_t generator,
                                                  uint32_t* outputPtr,
                                                  size_t n)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  raw<uint32_t> func;
  return randutilimpl::dispatch<decltype(func), uint32_t>(gen, func, n, outputPtr);
}

#pragma endregion

#pragma region uniform

#include "generator_uniform.inl"

extern "C" rnd_status_t randutilGenerateUniformEx(
  randutilGenerator_t generator, float* outputPtr, size_t n, float low, float high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  uniform_t<float> func;
  func.offset = high;
  func.mult = low - high;  // randutil_uniform is 0 exclusive and 1 inclusive. We want low inclusive
                           // and high exclusive
  return randutilimpl::dispatch<decltype(func), float>(gen, func, n, outputPtr);
}

extern "C" rnd_status_t randutilGenerateUniformDoubleEx(
  randutilGenerator_t generator, double* outputPtr, size_t n, double low, double high)
{
  randutilimpl::basegenerator* gen = (randutilimpl::basegenerator*)generator;
  uniform_t<double> func;
  func.offset = high;
  func.mult   = low - high;
  return randutilimpl::dispatch<decltype(func), double>(gen, func, n, outputPtr);
}

#pragma endregion
