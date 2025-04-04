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

#include "randomizer.h"

template <typename field_t>
struct laplace_t;

template <>
struct laplace_t<float> {
  float mu, beta;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    float y = randutilimpl::engine_uniform<float>(gen);  // y cannot be zero
    if (y == 0.5f) return mu;
    if (y < 0.5f)
      return mu + beta * ::logf(2.0f * y);
    else
      return mu - beta * ::logf(2.0f * y - 1.0f);  // y can be 1.0 => revert y to avoid this
  }
};

template <>
struct laplace_t<double> {
  double mu, beta;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    double y = randutilimpl::engine_uniform<double>(gen);  // y cannot be zero
    if (y == 0.5) return mu;
    if (y < 0.5)
      return mu + beta * ::log(2.0 * y);
    else
      return mu - beta * ::log(2.0 * y - 1.0);  // y can be 1.0 => revert y to avoid this
  }
};
