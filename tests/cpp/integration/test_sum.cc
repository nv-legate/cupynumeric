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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "common_utils.h"

namespace {

TEST(Sum, EmptyArray)
{
  const auto a      = cupynumeric::mk_array<int32_t>({}, {0});
  const auto result = sum(a);

  const std::vector<int64_t> expect{0};
  const std::vector<uint64_t> shape{1};

  cupynumeric::check_array(result, expect, shape);
}

TEST(Sum, SingleElement)
{
  const auto a      = cupynumeric::mk_array<int32_t>({42});
  const auto result = sum(a);

  const std::vector<int64_t> expect{42};
  const std::vector<uint64_t> shape{1};

  cupynumeric::check_array(result, expect, shape);
}

TEST(Sum, 1DArray)
{
  const auto a      = cupynumeric::mk_array<int32_t>({1, 2, 3, 4});
  const auto result = sum(a);

  const std::vector<int64_t> expect{10};
  const std::vector<uint64_t> shape{1};

  cupynumeric::check_array(result, expect, shape);
}

TEST(Sum, 2DArray)
{
  const auto a      = cupynumeric::mk_array<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const auto result = sum(a);

  const std::vector<float> expect{21.0f};
  const std::vector<uint64_t> shape{1};

  cupynumeric::check_array_near(result, expect, shape);
}

}  // namespace
