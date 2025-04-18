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

#include "generator.h"

#include "randomizer.h"

template <typename field_t>
struct weibull_t;

template <>
struct weibull_t<float> {
  float lambda, invk;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    float y = randutilimpl::engine_uniform<float>(gen);  // y cannot be 0
    // log(y) can be zero !
    auto lny = ::logf(y);
    if (lny == 0.0f) {
      return 0.0f;
    }
    return lambda * ::expf(::logf(-lny) * invk);
  }
};

template <>
struct weibull_t<double> {
  double lambda, invk;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    double y = randutilimpl::engine_uniform<double>(gen);  // y cannot be 0
    // log(y) can be zero !
    auto lny = ::log(y);
    if (lny == 0.0f) {
      return 0.0f;
    }
    return lambda * ::exp(::log(-lny) * invk);
  }
};
