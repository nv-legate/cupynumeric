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
#include "legate.h"
#include "cupynumeric.h"
#include "util.inl"

template <size_t SIZE, int32_t DIM>
void transpose_int32_test(std::array<int32_t, SIZE> input,
                          std::array<int32_t, SIZE> exp,
                          std::vector<uint64_t> in_shape,
                          std::vector<uint64_t> out_shape,
                          std::optional<std::vector<int32_t>> axes = std::nullopt)
{
  auto a_input = cupynumeric::zeros(in_shape, legate::int32());
  assign_values_to_array<int32_t, DIM>(a_input, input.data(), input.size());

  auto a_output = cupynumeric::array(out_shape, legate::int32());

  if (axes) {
    a_output = cupynumeric::transpose(a_input, axes.value());
  } else {
    a_output = cupynumeric::transpose(a_input);
  }
  check_array_eq<int32_t, DIM>(a_output, exp.data(), exp.size());
  EXPECT_EQ(a_output.shape(), out_shape);
}

TEST(Transpose, Dim)
{
  const size_t size               = 6;
  const int32_t dim               = 2;
  std::array<int32_t, size> input = {1, 2, 3, 4, 5, 6};
  std::array<int32_t, size> exp   = {1, 4, 2, 5, 3, 6};
  std::vector<uint64_t> in_shape  = {2, 3};
  std::vector<uint64_t> out_shape = {3, 2};
  auto axes                       = std::nullopt;

  transpose_int32_test<size, dim>(input, exp, in_shape, out_shape, axes);
}

TEST(Transpose, Axes)
{
  const size_t size               = 12;
  const int32_t dim               = 3;
  std::array<int32_t, size> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::array<int32_t, size> exp   = {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12};
  std::vector<uint64_t> in_shape  = {2, 2, 3};
  std::vector<uint64_t> out_shape = {3, 2, 2};
  auto axes                       = {2, 1, 0};

  transpose_int32_test<size, dim>(input, exp, in_shape, out_shape, axes);
}

TEST(Transpose, EmptyArray)
{
  const size_t size               = 0;
  const int32_t dim               = 1;
  std::array<int32_t, size> input = {};
  std::array<int32_t, size> exp   = input;
  std::vector<uint64_t> in_shape  = {0};
  std::vector<uint64_t> out_shape = in_shape;
  auto axes                       = std::nullopt;

  transpose_int32_test<size, dim>(input, exp, in_shape, out_shape, axes);
}

TEST(Transpose, SingletonAxes)
{
  const size_t size               = 6;
  const int32_t dim               = 1;
  std::array<int32_t, size> input = {1, 2, 3, 4, 5, 6};
  std::array<int32_t, size> exp   = input;
  std::vector<uint64_t> in_shape  = {6};
  std::vector<uint64_t> out_shape = in_shape;
  auto axes                       = {1};

  transpose_int32_test<size, dim>(input, exp, in_shape, out_shape, axes);
}

TEST(Transpose, Singleton)
{
  const size_t size               = 6;
  const int32_t dim               = 1;
  std::array<int32_t, size> input = {1, 2, 3, 4, 5, 6};
  std::array<int32_t, size> exp   = input;
  std::vector<uint64_t> in_shape  = {6};
  std::vector<uint64_t> out_shape = in_shape;
  auto axes                       = std::nullopt;

  transpose_int32_test<size, dim>(input, exp, in_shape, out_shape, axes);
}

TEST(Transpose, DefaultType)
{
  const size_t size               = 6;
  const int32_t dim               = 2;
  std::array<double, size> input  = {1.3, 2, 3.6, 4, 5, 6};
  std::array<double, size> exp    = {1.3, 4, 2, 5, 3.6, 6};
  std::vector<uint64_t> in_shape  = {2, 3};
  std::vector<uint64_t> out_shape = {3, 2};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, dim>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::transpose(a_input);
  check_array_eq<double, dim>(a_output, exp.data(), exp.size());
  EXPECT_EQ(a_output.shape(), out_shape);
}

TEST(TransposeErrors, InvalidAxes)
{
  const size_t size               = 6;
  const int32_t dim               = 2;
  std::array<double, size> input  = {1.3, 2, 3.6, 4, 5, 6};
  std::vector<uint64_t> in_shape  = {2, 3};
  std::vector<uint64_t> out_shape = {3, 2};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, dim>(a_input, input.data(), input.size());
  EXPECT_THROW(cupynumeric::transpose(a_input, (std::vector<int32_t>){0, 1, 2}),
               std::invalid_argument);
  EXPECT_THROW(cupynumeric::transpose(a_input, (std::vector<int32_t>){1}), std::invalid_argument);
  EXPECT_THROW(cupynumeric::transpose(a_input, (std::vector<int32_t>){3, 4}),
               std::invalid_argument);
}
