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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <cstdint>
#include "legate.h"
#include "cupynumeric.h"
#include "common_utils.h"

namespace {

TEST(ArangeType, ImplicitInt64)
{
  int64_t start               = 1567891032456;
  std::optional<int64_t> stop = 1567891032465;
  std::array<int64_t, 9> exp  = {1567891032456,
                                 1567891032457,
                                 1567891032458,
                                 1567891032459,
                                 1567891032460,
                                 1567891032461,
                                 1567891032462,
                                 1567891032463,
                                 1567891032464};
  auto arr                    = cupynumeric::arange(start, stop);
  check_array_eq<int64_t, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeType, ImplicitInt32)
{
  int32_t stop                = 10;
  std::array<int32_t, 10> exp = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto arr                    = cupynumeric::arange(stop);
  check_array_eq<int32_t, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeType, ImplicitFloat64)
{
  double start              = 1.5;
  double stop               = 10.5;
  std::array<double, 9> exp = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
  auto arr                  = cupynumeric::arange(start, (std::optional<double>)stop);
  check_array_eq<double, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeType, ImplicitFloat32)
{
  float start              = 1.5;
  float stop               = 10.5;
  std::array<float, 9> exp = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
  auto arr                 = cupynumeric::arange(start, (std::optional<float>)stop);
  check_array_eq<float, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeType, ExplicitInt32)
{
  float start                = 1.5;
  float stop                 = 10.5;
  std::array<int32_t, 9> exp = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto arr                   = cupynumeric::arange<int32_t>(start, stop);
  check_array_eq<int32_t, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeScalar, Float32)
{
  float start              = 1.5;
  float stop               = 10.5;
  std::array<float, 9> exp = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
  auto arr                 = cupynumeric::arange(legate::Scalar(start), legate::Scalar(stop));
  check_array_eq<float, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeScalar, EmptyArray)
{
  constexpr float start          = 10.5;
  constexpr float stop           = 10.5;
  const std::array<float, 0> exp = {};
  const auto arr                 = cupynumeric::arange(legate::Scalar(start), legate::Scalar(stop));

  check_array_eq<float, 1>(arr, exp.data(), exp.size());
}

TEST(ArangeErrors, ScalarTypeMismatch)
{
  float start  = 1.5;
  int32_t stop = 10;
  EXPECT_THROW(cupynumeric::arange(legate::Scalar(start), legate::Scalar(stop)),
               std::invalid_argument);
}

TEST(ArangeErrorsFromNDArray, StartTypeMismatch)
{
  constexpr float start  = 1.5;
  constexpr int32_t stop = 10;
  constexpr int32_t step = 1;

  auto arr = cupynumeric::mk_array<int32_t>({1, 2, 3});

  ASSERT_THAT(
    [&] { arr.arange(legate::Scalar{start}, legate::Scalar{stop}, legate::Scalar{step}); },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("start/stop/step should have the same type as the array")));
}

}  // namespace