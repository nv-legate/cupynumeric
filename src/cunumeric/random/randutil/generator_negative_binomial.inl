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
struct negative_binomial_t;

template <>
struct negative_binomial_t<uint32_t> {
  uint32_t n;
  double p;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    double lambda = rk_standard_gamma(&gen, (double)n) * ((1 - p) / p);
    return static_cast<float>(randutilimpl::engine_poisson<uint32_t>(gen, lambda));
  }
};
