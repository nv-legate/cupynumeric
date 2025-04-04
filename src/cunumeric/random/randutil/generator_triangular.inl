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
struct triangular_t;

template <>
struct triangular_t<float> {
  float a, b, c;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    float y = randutilimpl::engine_uniform<float>(gen);  // y cannot be 0
    if (y <= ((c - a) / (b - a))) {
      float delta = (y * (b - a) * (c - a));
      if (delta < 0.0f) delta = 0.0f;
      return a + ::sqrtf(delta);
    } else {
      float delta = ((1.0f - y) * (b - a) * (b - c));
      if (delta < 0.0f) delta = 0.0f;
      return b - ::sqrtf(delta);
    }
  }
};

template <>
struct triangular_t<double> {
  double a, b, c;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    double y = randutilimpl::engine_uniform<double>(gen);  // y cannot be 0
    if (y <= ((c - a) / (b - a))) {
      double delta = (y * (b - a) * (c - a));
      if (delta < 0.0) delta = 0.0;
      return a + ::sqrt(delta);
    } else {
      double delta = ((1.0 - y) * (b - a) * (b - c));
      if (delta < 0.0) delta = 0.0;
      return b - ::sqrt(delta);
    }
  }
};
