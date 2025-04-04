/* Copyright 2023 NVIDIA Corporation
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

#include "cunumeric/random/rnd_aliases.h"

#ifdef USE_STL_RANDOM_ENGINE_

#define CHECK_RND_ENGINE(expr)                             \
  do {                                                     \
    rnd_status_t __result__ = (expr);                      \
    randutil_check_status(__result__, __FILE__, __LINE__); \
  } while (false)

// #define randutil_check_curand randutil_check_status

namespace cunumeric {
legate::Logger& randutil_log();

void randutil_check_status(rnd_status_t error, const char* file, int line);

static inline randRngType get_rndRngType(cunumeric::BitGeneratorType kind)
{
  // for now, all generator types rerouted to STL
  // would use the MT19937 generator; perhaps,
  // this might become more flexible in the future;
  //
  switch (kind) {
    case cunumeric::BitGeneratorType::DEFAULT: return randRngType::STL_MT_19937;
    case cunumeric::BitGeneratorType::XORWOW: return randRngType::STL_MT_19937;
    case cunumeric::BitGeneratorType::MRG32K3A: return randRngType::STL_MT_19937;
    case cunumeric::BitGeneratorType::MTGP32: return randRngType::STL_MT_19937;
    case cunumeric::BitGeneratorType::MT19937: return randRngType::STL_MT_19937;
    case cunumeric::BitGeneratorType::PHILOX4_32_10: return randRngType::STL_MT_19937;
    default: LEGATE_ABORT;
  }
  return randRngType::RND_RNG_TEST;
}

}  // namespace cunumeric

#else
#include "cunumeric/random/curand_help.h"

#define CHECK_RND_ENGINE(expr) CHECK_CURAND((expr))
#define randutil_check_status randutil_check_curand
#define get_rndRngType get_curandRngType

#endif
