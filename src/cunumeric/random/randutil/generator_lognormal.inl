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
struct lognormal_t;

template <>
struct lognormal_t<float> {
  float mean   = 0.0;
  float stddev = 1.0;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    return randutilimpl::engine_log_normal(gen, mean, stddev);
  }
};

template <>
struct lognormal_t<double> {
  double mean   = 0.0;
  double stddev = 1.0;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    return randutilimpl::engine_log_normal(gen, mean, stddev);
  }
};
