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

TEST(ArrayEqual, EmptyArrays)
{
  const auto a = cupynumeric::mk_array<int32_t>({}, {0});
  const auto b = cupynumeric::mk_array<int32_t>({}, {0});

  EXPECT_TRUE(cupynumeric::array_equal(a, b));
}

TEST(ArrayEqual, DifferentShapes)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3});
  const auto b = cupynumeric::mk_array<int32_t>({1, 2});

  EXPECT_FALSE(cupynumeric::array_equal(a, b));
}

TEST(ArrayEqual, 1DArraysSameValues)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3});
  const auto b = cupynumeric::mk_array<int32_t>({1, 2, 3});

  EXPECT_TRUE(cupynumeric::array_equal(a, b));
}

TEST(ArrayEqual, 1DArraysDifferentValues)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3});
  const auto b = cupynumeric::mk_array<int32_t>({1, 2, 4});

  EXPECT_FALSE(cupynumeric::array_equal(a, b));
}

TEST(ArrayEqual, 2DArraysSameValues)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, {2, 2});
  const auto b = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, {2, 2});

  EXPECT_TRUE(cupynumeric::array_equal(a, b));
}

TEST(ArrayEqual, 2DArraysDifferentValues)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, {2, 2});
  const auto b = cupynumeric::mk_array<int32_t>({1, 2, 3, 5}, {2, 2});

  EXPECT_FALSE(cupynumeric::array_equal(a, b));
}

}  // namespace