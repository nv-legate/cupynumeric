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
#include "random_distributions.h"

template <typename field_t>
struct f_t;

template <>
struct f_t<float> {
  float dfnum, dfden;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    // TODO: fp32 implementation ?
    return (float)rk_f(&gen, (double)dfnum, (double)dfden);  // no float implementation
  }
};

template <typename field_t>
struct noncentralf_t;

template <>
struct noncentralf_t<float> {
  float dfnum, dfden, nonc;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    // TODO: fp32 implementation ?
    return (float)rk_noncentral_f(
      &gen, (double)dfnum, (double)dfden, (double)nonc);  // no float implementation
  }
};

template <>
struct f_t<double> {
  double dfnum, dfden;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS double operator()(gen_t& gen)
  {
    return rk_f(&gen, dfnum, dfden);
  }
};

template <>
struct noncentralf_t<double> {
  double dfnum, dfden, nonc;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS float operator()(gen_t& gen)
  {
    return rk_noncentral_f(&gen, dfnum, dfden, nonc);
  }
};
