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

#include "generator.h"

template <typename gen_t, randutilimpl::execlocation location>
rnd_status_t randutilGenerator(randutilGenerator_t* generator,
                               uint64_t seed,
                               uint64_t generatorID,
                               stream_t stream = nullptr)
{
  randutilimpl::inner_generator<gen_t, location>* result =
    new randutilimpl::inner_generator<gen_t, location>(seed, generatorID, stream);
  *generator = (randutilGenerator_t)result;
  return RND_STATUS_SUCCESS;
}

template <randutilimpl::execlocation location>
static rnd_status_t inner_randutilCreateGenerator(randutilGenerator_t* generator,
                                                  randRngType_t rng_type,
                                                  uint64_t seed,
                                                  uint64_t generatorID,
                                                  stream_t stream = nullptr)
{
  switch (static_cast<int>(rng_type)) {
    case RND_RNG_PSEUDO_XORWOW:
      return randutilGenerator<gen_XORWOW_t, location>(generator, seed, generatorID, stream);
    case RND_RNG_PSEUDO_PHILOX4_32_10:
      return randutilGenerator<gen_Philox4_32_10_t, location>(generator, seed, generatorID, stream);
    case RND_RNG_PSEUDO_MRG32K3A:
      return randutilGenerator<gen_MRG32k3a_t, location>(generator, seed, generatorID, stream);
    default: LEGATE_ABORT;
  }
  return RND_STATUS_TYPE_ERROR;
}
