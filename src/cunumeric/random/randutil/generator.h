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
#include <cassert>

#include "legate.h"
#include "randutil_curand.h"
#include "randutil_impl.h"

#include "cunumeric/random/rnd_aliases.h"

namespace randutilimpl {

struct basegenerator {
  virtual int generatorTypeId()   = 0;
  virtual execlocation location() = 0;
  virtual void destroy()          = 0;  // avoid exceptions in destructor
  virtual ~basegenerator() {}
};

template <typename gen_t>
struct generatorid;

template <>
struct generatorid<gen_XORWOW_t> {
  static constexpr int rng_type = RND_RNG_PSEUDO_XORWOW;
};

#ifndef USE_STL_RANDOM_ENGINE_
// Curand *different* specializations, not possible with only one generator
//
template <>
struct generatorid<gen_Philox4_32_10_t> {
  static constexpr int rng_type = RND_RNG_PSEUDO_PHILOX4_32_10;
};
template <>
struct generatorid<gen_MRG32k3a_t> {
  static constexpr int rng_type = RND_RNG_PSEUDO_MRG32K3A;
};
#endif

#if 0
// this seems dead code
//
template <int gen_id>
struct generatortype;

template <>
struct generatortype<RND_RNG_PSEUDO_XORWOW> {
  using rng_t = gen_XORWOW_t;
};
template <>
struct generatortype<RND_RNG_PSEUDO_PHILOX4_32_10> {
  using rng_t = gen_Philox4_32_10_t;
};
template <>
struct generatortype<RND_RNG_PSEUDO_MRG32K3A> {
  using rng_t = gen_MRG32k3a_t;
};
#endif

template <typename gen_t>
struct inner_generator<gen_t, randutilimpl::execlocation::HOST> : basegenerator {
  uint64_t seed;
  uint64_t generatorID;
  gen_t generator;

  inner_generator(uint64_t seed, uint64_t generatorID, stream_t ignored)
    : seed(seed),
      generatorID(generatorID)
#ifdef USE_STL_RANDOM_ENGINE_
      ,
      generator(seed)
#endif
  {
#ifdef USE_STL_RANDOM_ENGINE_
    std::srand(seed);
#else
    curand_init(seed, generatorID, 0, &generator);
#endif
  }

  virtual void destroy() override {}

  virtual execlocation location() override { return randutilimpl::execlocation::HOST; }

  virtual int generatorTypeId() override { return generatorid<gen_t>::rng_type; }

  virtual ~inner_generator() {}

  template <typename func_t, typename out_t>
  curandStatus_t draw(func_t func, size_t N, out_t* out)
  {
    for (size_t k = 0; k < N; ++k) { out[k] = func(generator); }
    return CURAND_STATUS_SUCCESS;
  }
};

template <randutilimpl::execlocation location, typename func_t, typename out_t>
rnd_status_t inner_dispatch_sample(basegenerator* gen, func_t func, size_t N, out_t* out)
{
  switch (gen->generatorTypeId()) {
    case RND_RNG_PSEUDO_XORWOW:
      return static_cast<inner_generator<gen_XORWOW_t, location>*>(gen)
        ->template draw<func_t, out_t>(func, N, out);
    case RND_RNG_PSEUDO_PHILOX4_32_10:
      return static_cast<inner_generator<gen_Philox4_32_10_t, location>*>(gen)
        ->template draw<func_t, out_t>(func, N, out);
    case RND_RNG_PSEUDO_MRG32K3A:
      return static_cast<inner_generator<gen_MRG32k3a_t, location>*>(gen)
        ->template draw<func_t, out_t>(func, N, out);
    default: LEGATE_ABORT;
  }
  return RND_STATUS_INTERNAL_ERROR;
}

// template funtion with HOST and DEVICE implementations
template <randutilimpl::execlocation location, typename func_t, typename out_t>
struct dispatcher {
  static rnd_status_t run(randutilimpl::basegenerator* generator,
                          func_t func,
                          size_t N,
                          out_t* out);
};

// HOST-side template instantiation of generator
template <typename func_t, typename out_t>
struct dispatcher<randutilimpl::execlocation::HOST, func_t, out_t> {
  static rnd_status_t run(randutilimpl::basegenerator* gen, func_t func, size_t N, out_t* out)
  {
    return inner_dispatch_sample<randutilimpl::execlocation::HOST, func_t, out_t>(
      gen, func, N, out);
  }
};

template <typename func_t, typename out_t>
rnd_status_t dispatch(randutilimpl::basegenerator* gen, func_t func, size_t N, out_t* out)
{
  switch (gen->location()) {
    case randutilimpl::execlocation::HOST:
      return dispatcher<randutilimpl::execlocation::HOST, func_t, out_t>::run(gen, func, N, out);
#ifdef LEGATE_USE_CUDA
    case randutilimpl::execlocation::DEVICE:
      return dispatcher<randutilimpl::execlocation::DEVICE, func_t, out_t>::run(gen, func, N, out);
#endif
    default: LEGATE_ABORT;
  }
  return RND_STATUS_INTERNAL_ERROR;
}

}  // namespace randutilimpl
