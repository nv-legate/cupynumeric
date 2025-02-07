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

#include "cupynumeric/random/rnd_aliases.h"
#include <string_view>

#ifdef CUPYNUMERIC_USE_STL_RANDOM_ENGINE

#define CHECK_RND_ENGINE(expr)                             \
  do {                                                     \
    rnd_status_t __result__ = (expr);                      \
    randutil_check_status(__result__, __FILE__, __LINE__); \
  } while (false)

// #define randutil_check_curand randutil_check_status

namespace cupynumeric {
legate::Logger& randutil_log();

void randutil_check_status(rnd_status_t error, std::string_view, int line);

static inline randRngType get_rndRngType(cupynumeric::BitGeneratorType kind)
{
  // for now, all generator types rerouted to STL
  // would use the MT19937 generator; perhaps,
  // this might become more flexible in the future;
  //
  switch (kind) {
    case cupynumeric::BitGeneratorType::DEFAULT: return randRngType::STL_MT_19937;
    case cupynumeric::BitGeneratorType::XORWOW: return randRngType::STL_MT_19937;
    case cupynumeric::BitGeneratorType::MRG32K3A: return randRngType::STL_MT_19937;
    case cupynumeric::BitGeneratorType::MTGP32: return randRngType::STL_MT_19937;
    case cupynumeric::BitGeneratorType::MT19937: return randRngType::STL_MT_19937;
    case cupynumeric::BitGeneratorType::PHILOX4_32_10: return randRngType::STL_MT_19937;
    default: LEGATE_ABORT("Unsupported random generator.");
  }
  return randRngType::RND_RNG_TEST;
}

}  // namespace cupynumeric

#else
#include "cupynumeric/random/curand_help.h"

#define CHECK_RND_ENGINE(...) CHECK_CURAND(__VA_ARGS__)
#define get_rndRngType get_curandRngType

#endif
