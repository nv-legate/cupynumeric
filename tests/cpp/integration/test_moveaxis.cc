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

#include <iomanip>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include "legate.h"
#include "cupynumeric.h"
#include "util.inl"

template <int32_t DIM>
static void moveaxis_int32_test(std::vector<int32_t> input,
                                std::vector<int32_t> exp,
                                std::vector<uint64_t> in_shape,
                                std::vector<uint64_t> out_shape,
                                std::vector<int32_t> source,
                                std::vector<int32_t> destination)
{
  auto a_input = cupynumeric::zeros(in_shape, legate::int32());
  assign_values_to_array<int32_t, DIM>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::moveaxis(a_input, source, destination);
  check_array_eq<int32_t, DIM>(a_output, exp.data(), exp.size());
  EXPECT_EQ(a_output.shape(), out_shape);
}

static void moveaxis_int32_test_2(std::vector<uint64_t> in_shape,
                                  std::vector<uint64_t> out_shape,
                                  std::vector<int32_t> source,
                                  std::vector<int32_t> destination)
{
  auto a_input  = cupynumeric::zeros(in_shape, legate::int32());
  auto a_output = cupynumeric::moveaxis(a_input, source, destination);
  EXPECT_EQ(a_output.shape(), out_shape);
}

TEST(MoveAxis, Normal)
{
  moveaxis_int32_test<2>({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {2, 3}, {2, 3}, {0}, {0});
  moveaxis_int32_test<2>({1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, {2, 3}, {3, 2}, {0}, {-1});
  moveaxis_int32_test<3>(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
    {0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23},
    {2, 3, 4},
    {3, 4, 2},
    {0},
    {-1});
}

TEST(MoveAxis, SpecialArrays)
{
  // test single element array
  {
    std::vector<int32_t> input{99};
    auto a = cupynumeric::zeros({1}, legate::int32());
    a.fill(legate::Scalar(input[0]));
    auto a_out = cupynumeric::moveaxis(a, {0}, {-1});
    check_array_eq<int32_t, 1>(a_out, input.data(), input.size());
    EXPECT_EQ(a_out.shape(), a.shape());
  }
  {
    std::vector<int32_t> input{-100};
    auto a = cupynumeric::zeros({1, 1}, legate::int32());
    a.fill(legate::Scalar(input[0]));
    auto a_out = cupynumeric::moveaxis(a, {0, 1}, {-1, -2});
    check_array_eq<int32_t, 2>(a_out, input.data(), input.size());
    EXPECT_EQ(a_out.shape(), a.shape());
  }

  // test empty array
  {
    auto a     = cupynumeric::zeros({0}, legate::int32());
    auto a_out = cupynumeric::moveaxis(a, {0}, {-1});
    EXPECT_EQ(a_out.shape(), a.shape());
  }
}

TEST(MoveAxis, Shape)
{
  moveaxis_int32_test_2({3, 4, 5}, {4, 5, 3}, {0}, {-1});
  moveaxis_int32_test_2({3, 4, 5}, {5, 3, 4}, {-1}, {0});
  moveaxis_int32_test_2({3, 4, 5}, {5, 4, 3}, {0, 1}, {-1, -2});
  moveaxis_int32_test_2({3, 4, 5}, {5, 4, 3}, {0, 1, 2}, {-1, -2, -3});
}

TEST(MoveAxis, Shape7D)
{
  moveaxis_int32_test_2({3, 2, 2, 2}, {2, 2, 2, 3}, {0}, {-1});

#if LEGATE_MAX_DIM >= 5
  moveaxis_int32_test_2({3, 2, 2, 2, 2}, {2, 2, 2, 2, 3}, {0}, {-1});
#endif

#if LEGATE_MAX_DIM >= 6
  moveaxis_int32_test_2({3, 4, 2, 2, 2, 2}, {2, 2, 2, 2, 4, 3}, {0, 1}, {-1, -2});
#endif

#if LEGATE_MAX_DIM >= 7
  moveaxis_int32_test_2({3, 4, 5, 2, 2, 2, 2}, {2, 2, 2, 2, 3, 4, 5}, {2, 1, 0}, {-1, -2, -3});
#endif
}

TEST(MoveAxis, EmptyShape)
{
  moveaxis_int32_test_2({0, 1, 2}, {1, 2, 0}, {0}, {-1});
  moveaxis_int32_test_2({1, 0, 7}, {7, 1, 0}, {-1}, {0});
  moveaxis_int32_test_2({4, 0, 9, 0}, {0, 4, 0, 9}, {2, 0}, {3, 1});
}

TEST(MoveAxis, With_empty_array)
{
  moveaxis_int32_test_2({3, 4}, {3, 4}, {}, {});
  moveaxis_int32_test_2({3, 4, 5}, {3, 4, 5}, {}, {});
}

TEST(MoveAxisErrors, Repeated_axis)
{
  auto x = cupynumeric::zeros({3, 4, 5}, legate::int32());
  EXPECT_THROW(cupynumeric::moveaxis(x, {0, 0}, {1, 0}), std::invalid_argument);
  EXPECT_THROW(cupynumeric::moveaxis(x, {0, 1}, {0, -3}), std::invalid_argument);
}

TEST(MoveAxisErrors, Axis_out_of_bound)
{
  auto x = cupynumeric::zeros({3, 4, 5}, legate::int32());
  EXPECT_THROW(cupynumeric::moveaxis(x, {0, 3}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(cupynumeric::moveaxis(x, {0, 1}, {0, -4}), std::invalid_argument);
  EXPECT_THROW(cupynumeric::moveaxis(x, {4}, {0}), std::invalid_argument);
  EXPECT_THROW(cupynumeric::moveaxis(x, {0}, {-4}), std::invalid_argument);
}

TEST(MoveAxisErrors, Axis_with_different_length)
{
  auto x = cupynumeric::zeros({3, 4, 5}, legate::int32());
  EXPECT_THROW(cupynumeric::moveaxis(x, {0}, {1, 0}), std::invalid_argument);
}
