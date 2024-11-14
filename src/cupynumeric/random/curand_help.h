/* Copyright 2024 NVIDIA Corporation
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

#include <curand.h>

#include <string_view>

#define CHECK_CURAND(...)                                  \
  do {                                                     \
    curandStatus_t __result__ = __VA_ARGS__;               \
    randutil_check_curand(__result__, __FILE__, __LINE__); \
  } while (false)

// necessary, b/c the STL variant (host only MacOS) uses non-curand abstractions,
// hence a different checker defined in bitgenerator.cc, while the curand checker
// gets undefined; however the device code still requires the curand checker and
// attempts to link against the definition in bitgenerator.cc, which is disabled
// in this situation; the checker below fulfills that purpose:
//
#define CHECK_CURAND_DEVICE(expr)                                 \
  do {                                                            \
    curandStatus_t __result__ = (expr);                           \
    randutil_check_curand_device(__result__, __FILE__, __LINE__); \
  } while (false)

namespace cupynumeric {
legate::Logger& randutil_log();
void randutil_check_curand(curandStatus_t error, std::string_view file, int line);

// required by CHECK_CURAND_DEVICE:
//
void randutil_check_curand_device(curandStatus_t error, const char* file, int line);

static inline curandRngType get_curandRngType(cupynumeric::BitGeneratorType kind)
{
  switch (kind) {
    case cupynumeric::BitGeneratorType::DEFAULT: return curandRngType::CURAND_RNG_PSEUDO_XORWOW;
    case cupynumeric::BitGeneratorType::XORWOW: return curandRngType::CURAND_RNG_PSEUDO_XORWOW;
    case cupynumeric::BitGeneratorType::MRG32K3A: return curandRngType::CURAND_RNG_PSEUDO_MRG32K3A;
    case cupynumeric::BitGeneratorType::MTGP32: return curandRngType::CURAND_RNG_PSEUDO_MTGP32;
    case cupynumeric::BitGeneratorType::MT19937: return curandRngType::CURAND_RNG_PSEUDO_MT19937;
    case cupynumeric::BitGeneratorType::PHILOX4_32_10:
      return curandRngType::CURAND_RNG_PSEUDO_PHILOX4_32_10;
    default: LEGATE_ABORT("Unknown generator type");
  }
  return curandRngType::CURAND_RNG_TEST;
}

}  // namespace cupynumeric
