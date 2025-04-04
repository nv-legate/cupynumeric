/* Copyright 2022 NVIDIA Corporation
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
#include "random_distributions.h"

#include "randomizer.h"

template <typename field_t>
struct wald_t;

template <>
struct wald_t<float> {
  float mu, lambda;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    float v = randutilimpl::engine_normal<float>(gen);
    float y = v * v;
    float x = mu + (mu * mu * y) / (2.0f * lambda) -
              (mu / (2.0f * lambda)) * ::sqrtf(mu * y * (4.0f * lambda + mu * y));
    float z = randutilimpl::engine_uniform<float>(gen);
    if (z <= (mu) / (mu + x))
      return x;
    else
      return (mu * mu) / x;
  }
};

template <>
struct wald_t<double> {
  double mu, lambda;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    double v = randutilimpl::engine_normal<double>(gen);
    double y = v * v;
    double x = mu + (mu * mu * y) / (2.0 * lambda) -
               (mu / (2.0 * lambda)) * ::sqrtf(mu * y * (4.0 * lambda + mu * y));
    double z = randutilimpl::engine_uniform<double>(gen);
    if (z <= (mu) / (mu + x))
      return x;
    else
      return (mu * mu) / x;
  }
};
