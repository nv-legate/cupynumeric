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
struct cauchy_t;

template <>
struct cauchy_t<float> {
  static constexpr float pi = 3.1415926535897932384626433832795f;

  float x0, gamma;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    using value_t = float;
    value_t y     = randutilimpl::engine_uniform<value_t>(gen);  // y \in (0, 1]

    // must avoid tan(pi/2), hence scale the interval by (1-eps):
    //
    y = y * (value_t{1} - cuda::std::numeric_limits<value_t>::epsilon());
    return x0 + gamma * ::tanf(pi * (y - 0.5f));
  }
};

template <>
struct cauchy_t<double> {
  static constexpr double pi = 3.1415926535897932384626433832795;

  double x0, gamma;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    using value_t = double;
    value_t y     = randutilimpl::engine_uniform<value_t>(gen);  // y \in (0, 1]

    // must avoid tan(pi/2), hence scale the interval by (1-eps):
    //
    y = y * (value_t{1} - cuda::std::numeric_limits<value_t>::epsilon());
    return x0 + gamma * ::tan(pi * (y - 0.5));
  }
};
