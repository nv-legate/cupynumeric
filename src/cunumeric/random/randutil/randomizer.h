/* Copyright 2023 NVIDIA Corporation
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

#include <random>
#include <functional>
#include <type_traits>

#include <cstdlib>

namespace randutilimpl {

// trampoline functions for branching-off CURAND vs. STL
// implementations (STL used on platforms that don't support CUDA)
//
template <typename element_t, typename gen_t>
RANDUTIL_QUALIFIERS decltype(auto) engine_uniform(gen_t& gen)
{
#ifdef USE_STL_RANDOM_ENGINE_
  std::uniform_real_distribution<element_t> dis(0, 1);
  auto y = dis(gen);  // returns [0, 1);

  // bring to (0, 1]:
  return 1 - y;
#else
  if constexpr (std::is_same_v<element_t, float>)
    return curand_uniform(&gen);  // returns (0, 1];
  else
    return curand_uniform_double(&gen);  // returns (0, 1];
#endif
}

template <typename ret_t, typename gen_t>
RANDUTIL_QUALIFIERS decltype(auto) engine_poisson(gen_t& gen, double lambda)
{
#ifdef USE_STL_RANDOM_ENGINE_
  std::poisson_distribution<ret_t> dis(lambda);
  return dis(gen);
#else
  return curand_poisson(&gen, lambda);
#endif
}

template <typename element_t, typename gen_t>
RANDUTIL_QUALIFIERS decltype(auto) engine_normal(gen_t& gen)
{
#ifdef USE_STL_RANDOM_ENGINE_
  std::normal_distribution<element_t> dis(0, 1);
  return dis(gen);
#else
  if constexpr (std::is_same_v<element_t, float>)
    return curand_normal(&gen);
  else
    return curand_normal_double(&gen);
#endif
}

template <typename gen_t, typename element_t>
RANDUTIL_QUALIFIERS decltype(auto) engine_log_normal(gen_t& gen, element_t mean, element_t stddev)
{
#ifdef USE_STL_RANDOM_ENGINE_
  std::lognormal_distribution<element_t> dis{mean, stddev};
  return dis(gen);
#else
  if constexpr (std::is_same_v<element_t, float>)
    return curand_log_normal(&gen, mean, stddev);
  else
    return curand_log_normal_double(&gen, mean, stddev);
#endif
}

template <typename gen_t>
RANDUTIL_QUALIFIERS decltype(auto) engine_rand(gen_t& gen)
{
#ifdef USE_STL_RANDOM_ENGINE_
  return std::rand();
#else
  return curand(&gen);
#endif
}

#ifdef EXPERIMENTAL_STL_BRANCH_OFF_
enum class random_client_t : int { CURAND = 0, STL };

template <typename state_type, random_client_t client, typename ret_type = void>
struct randomizer_t;

// Curand randomizer maintains a state:
//
template <typename state_t, typename ret_t>
struct randomizer_t<state_t, random_client_t::CURAND, ret_t> {
  randomizer_t(state_t state, std::function<ret_t(state_t&)> f) : state_(state), f_(f) {}

  ret_t run(void) { return std::invoke(f_, state_); }

 private:
  state_t state_;
  std::function<ret_t(state_t&)> f_;
};

// STL randomizer uses Distribution-specific info (DSI); e.g., (low, high) for uniform:
//
template <typename dsi_data_t, typename ret_t>
struct randomizer_t<dsi_data_t, random_client_t::STL, ret_t> {
  randomizer_t(dsi_data_t const& data, std::function<ret_t(dsi_data_t const&)> f)
    : data_(data), f_(f)
  {
  }

  ret_t run(void) const { return std::invoke(f_, data_); }

 private:
  dsi_data_t data_;
  std::function<ret_t(dsi_data_t const&)> f_;
};
#endif

}  // namespace randutilimpl
