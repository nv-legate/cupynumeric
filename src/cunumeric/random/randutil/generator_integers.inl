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
#include <type_traits>

#include "randomizer.h"

template <typename field_t, typename = void>
struct integers;

template <typename field_t>
struct integers<
  field_t,
  std::enable_if_t<std::is_same_v<field_t, int16_t> || std::is_same_v<field_t, int32_t>>> {
  using ufield_t = std::conditional_t<std::is_same_v<field_t, int16_t>, uint16_t, uint32_t>;
  field_t from;
  field_t to;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS field_t operator()(gen_t& gen)
  {
    auto y = randutilimpl::engine_rand(gen);
    return (field_t)(y % (ufield_t)(to - from)) + from;
  }
};

template <typename field_t>
struct integers<field_t, std::enable_if_t<std::is_same_v<field_t, int64_t>>> {
  using ufield_t = uint64_t;
  field_t from;
  field_t to;

  template <typename gen_t>
  RANDUTIL_QUALIFIERS field_t operator()(gen_t& gen)
  {
    // take two draws to get a 64 bits value
    unsigned low  = randutilimpl::engine_rand(gen);
    unsigned high = randutilimpl::engine_rand(gen);

    return (field_t)((((ufield_t)high << 32) | (ufield_t)low) % (to - from)) + from;
  }
};
