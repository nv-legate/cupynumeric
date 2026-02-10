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
#include "cupynumeric.h"
#include "util.inl"

template <size_t IN_SIZE, size_t IN_DIM, size_t EXP_SIZE, size_t EXP_DIM>
void diagonal_test(std::array<double, IN_SIZE> input,
                   std::array<double, EXP_SIZE> exp,
                   std::vector<uint64_t> in_shape,
                   int32_t offset = 0,
                   int32_t axis1  = 0,
                   int32_t axis2  = 1,
                   bool extract   = true)
{
  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, IN_DIM>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::diagonal(a_input, offset, axis1, axis2, extract);
  check_array_eq<double, EXP_DIM>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, Singleton)
{
  const size_t in_size              = 6;
  const size_t in_dim               = 1;
  const size_t exp_size             = 36;
  const size_t exp_dim              = 2;
  std::array<double, in_size> input = {1.3, 2, 3.6, 4, 5, 6};
  std::array<double, exp_size> exp  = {1.3, 0., 0.,  0., 0., 0., 0., 2., 0., 0., 0., 0.,
                                       0.,  0., 3.6, 0., 0., 0., 0., 0., 0., 4., 0., 0.,
                                       0.,  0., 0.,  0., 5., 0., 0., 0., 0., 0., 0., 6.};
  std::vector<uint64_t> in_shape    = {6};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::diagonal(a_input, 0, std::nullopt, std::nullopt, false);
  check_array_eq<double, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, SingletonPositiveOffset)
{
  constexpr size_t in_size                = 3;
  constexpr size_t in_dim                 = 1;
  constexpr size_t exp_size               = 16;
  constexpr size_t exp_dim                = 2;
  const std::array<double, in_size> input = {1.0, 2.0, 3.0};
  const std::array<double, exp_size> exp  = {
    0., 1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0.};
  const std::vector<uint64_t> in_shape = {3};

  auto a_input = cupynumeric::zeros(in_shape);

  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());

  const auto a_output = cupynumeric::diagonal(
    a_input, /*offset=*/1, /*axis1=*/std::nullopt, /*axis2=*/std::nullopt, /*extract=*/false);

  check_array_eq<double, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, NegativeOffsetExtract2D)
{
  constexpr size_t in_size                = 9;
  constexpr size_t in_dim                 = 2;
  constexpr size_t exp_size               = 2;
  constexpr size_t exp_dim                = 1;
  const std::array<double, in_size> input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  const std::array<double, exp_size> exp  = {4.0, 8.0};
  const std::vector<uint64_t> in_shape    = {3, 3};

  auto a_input = cupynumeric::zeros(in_shape);

  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());

  const auto a_output =
    cupynumeric::diagonal(a_input, /*offset=*/-1, /*axis1=*/0, /*axis2=*/1, /*extract=*/true);

  check_array_eq<double, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, SingletonExtract)
{
  std::vector<uint64_t> in_shape = {6};

  auto a_input = cupynumeric::zeros(in_shape);
  EXPECT_THROW(cupynumeric::diagonal(a_input, 0, std::nullopt, std::nullopt, true),
               std::invalid_argument);
}

TEST(Diagonal, SingletonAxes)
{
  std::vector<uint64_t> in_shape = {6};

  auto a_input = cupynumeric::zeros(in_shape);
  EXPECT_THROW(cupynumeric::diagonal(a_input, 0, 0, 1, false), std::invalid_argument);
}

TEST(Diagonal, Defaults)
{
  const size_t in_size              = 9;
  const size_t in_dim               = 2;
  const size_t exp_size             = 3;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5, 6};
  std::array<double, exp_size> exp  = {9, 2, 6};
  std::vector<uint64_t> in_shape    = {3, 3};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::diagonal(a_input);
  check_array_eq<double, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, EmptyArray)
{
  const size_t exp_size             = 0;
  const size_t exp_dim              = 2;
  std::array<int32_t, exp_size> exp = {};

  auto a_input  = cupynumeric::array({0}, legate::int32());
  auto a_output = cupynumeric::diagonal(a_input, 0, std::nullopt, std::nullopt, false);
  check_array_eq<int32_t, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, Simple)
{
  const size_t in_size              = 9;
  const size_t in_dim               = 2;
  const size_t exp_size             = 3;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5, 6};
  std::array<double, exp_size> exp  = {9, 2, 6};
  std::vector<uint64_t> in_shape    = {3, 3};

  diagonal_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape);
}

TEST(Diagonal, Offset)
{
  const size_t in_size              = 9;
  const size_t in_dim               = 3;
  const size_t exp_size             = 1;
  const size_t exp_dim              = 2;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5, 6};
  std::array<double, exp_size> exp  = {0.5};
  std::vector<uint64_t> in_shape    = {3, 3, 1};

  diagonal_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape, 2);
}

TEST(Diagonal, NegativeOffset)
{
  constexpr size_t in_size                = 18;
  constexpr size_t in_dim                 = 3;
  constexpr size_t exp_size               = 4;
  constexpr size_t exp_dim                = 2;
  const std::array<double, in_size> input = {
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  const std::array<double, exp_size> exp = {0.0, 0.0, 0.0, 0.0};
  const std::vector<uint64_t> in_shape   = {3, 3, 2};

  diagonal_test<in_size, in_dim, exp_size, exp_dim>(
    input, exp, in_shape, /*offset=*/-1, /*axis1=*/0, /*axis2=*/1);
}

TEST(Diagonal, Axes)
{
  const size_t in_size              = 6;
  const size_t in_dim               = 2;
  const size_t exp_size             = 2;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {1.3, 2, 3.6, 4, 5, 6};
  std::array<double, exp_size> exp  = {1.3, 5};
  std::vector<uint64_t> in_shape    = {2, 3};

  diagonal_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape, 0, 1, 0);
}

TEST(Diagonal, InvalidAxes)
{
  const size_t in_size              = 6;
  const size_t in_dim               = 2;
  std::array<double, in_size> input = {1.3, 2, 3.6, 4, 5, 6};
  std::vector<uint64_t> in_shape    = {2, 3};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  EXPECT_THROW(cupynumeric::diagonal(a_input, 0, 2, 6, true), std::invalid_argument);
  EXPECT_THROW(cupynumeric::diagonal(a_input, 0, 1, 1, true), std::invalid_argument);
}

TEST(Diagonal, InvalidOffset)
{
  const size_t in_size              = 6;
  const size_t in_dim               = 2;
  std::array<double, in_size> input = {1.3, 2, 3.6, 4, 5, 6};
  std::vector<uint64_t> in_shape    = {2, 3};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  EXPECT_THROW(cupynumeric::diagonal(a_input, 3), std::invalid_argument);
}

TEST(Diagonal, IntArray)
{
  const size_t in_size               = 6;
  const size_t in_dim                = 2;
  const size_t exp_size              = 2;
  const size_t exp_dim               = 1;
  std::array<int32_t, in_size> input = {1, 2, 3, 4, 5, 6};
  std::array<int32_t, exp_size> exp  = {1, 5};
  std::vector<uint64_t> in_shape     = {2, 3};

  auto a_input = cupynumeric::zeros(in_shape, legate::int32());
  assign_values_to_array<int32_t, in_dim>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::diagonal(a_input);
  check_array_eq<int32_t, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Diagonal, MaxDim)
{
  // Only test int type for max dim
  const size_t in_size              = 16;
  std::array<double, in_size> input = {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
#if LEGATE_MAX_DIM >= 4
  diagonal_test<in_size, 4, 8, 3>(input, {14, 6, 10, 11, 3, 1, 12, 15}, {2, 2, 1, 4});
#endif

#if LEGATE_MAX_DIM >= 5
  diagonal_test<in_size, 5, 8, 4>(input, {14, 10, 3, 12, 5, 13, 2, 4}, {1, 2, 2, 1, 4});
#endif

#if LEGATE_MAX_DIM >= 6
  diagonal_test<in_size, 6, 8, 5>(input, {14, 10, 3, 12, 5, 13, 2, 4}, {2, 1, 1, 2, 2, 2});
#endif

#if LEGATE_MAX_DIM >= 7
  diagonal_test<in_size, 7, 8, 6>(input, {14, 6, 10, 11, 3, 1, 12, 15}, {2, 2, 1, 1, 2, 1, 2});
#endif
}

template <size_t IN_SIZE, size_t IN_DIM, size_t EXP_SIZE, size_t EXP_DIM>
void trace_test(std::array<double, IN_SIZE> input,
                std::array<double, EXP_SIZE> exp,
                std::vector<uint64_t> in_shape,
                int32_t offset                          = 0,
                int32_t axis1                           = 0,
                int32_t axis2                           = 1,
                std::optional<legate::Type> type        = std::nullopt,
                std::optional<cupynumeric::NDArray> out = std::nullopt)
{
  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, IN_DIM>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::trace(a_input, offset, axis1, axis2, type, out);
  check_array_eq<double, EXP_DIM>(a_output, exp.data(), exp.size());
}

TEST(Trace, Simple2D)
{
  constexpr size_t in_size                = 8;
  constexpr size_t in_dim                 = 2;
  constexpr size_t exp_size               = 1;
  constexpr size_t exp_dim                = 1;
  const std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  const std::array<double, exp_size> exp  = {12.6};
  const std::vector<uint64_t> in_shape    = {2, 4};

  trace_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape);
}

TEST(Trace, Simple3D)
{
  constexpr size_t in_size                = 8;
  constexpr size_t in_dim                 = 3;
  constexpr size_t exp_size               = 4;
  constexpr size_t exp_dim                = 1;
  const std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  const std::array<double, exp_size> exp  = {9, 7, 0.5, 1.3};
  const std::vector<uint64_t> in_shape    = {2, 1, 4};

  trace_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape);
}

TEST(Trace, Offset)
{
  const size_t in_size              = 8;
  const size_t in_dim               = 3;
  const size_t exp_size             = 1;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  std::array<double, exp_size> exp  = {5.5};
  std::vector<uint64_t> in_shape    = {2, 4, 1};
  trace_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape, 2);
}

TEST(Trace, Axes)
{
  const size_t in_size              = 8;
  const size_t in_dim               = 3;
  const size_t exp_size             = 2;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  std::array<double, exp_size> exp  = {9, 2};
  std::vector<uint64_t> in_shape    = {2, 4, 1};
  trace_test<in_size, in_dim, exp_size, exp_dim>(input, exp, in_shape, 0, 2, 1);
}

TEST(Trace, IntArray)
{
  const size_t in_size               = 8;
  const size_t in_dim                = 3;
  const size_t exp_size              = 1;
  const size_t exp_dim               = 1;
  std::array<int32_t, in_size> input = {9, 7, 5, 3, 2, 6, 4, 1};
  std::array<int32_t, exp_size> exp  = {15};
  std::vector<uint64_t> in_shape     = {2, 4, 1};
  auto a_input                       = cupynumeric::zeros(in_shape, legate::int32());
  assign_values_to_array<int32_t, in_dim>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::trace(a_input, 0, 0, 1);
  check_array_eq<int32_t, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Trace, TypeInt)
{
  const size_t in_size              = 8;
  const size_t in_dim               = 3;
  const size_t exp_size             = 1;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  std::array<int32_t, exp_size> exp = {12};
  std::vector<uint64_t> in_shape    = {2, 4, 1};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  auto a_output = cupynumeric::trace(a_input, 0, 0, 1, legate::int32());
  check_array_eq<int32_t, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Trace, OutType)
{
  const size_t in_size              = 8;
  const size_t in_dim               = 3;
  const size_t exp_size             = 1;
  const size_t exp_dim              = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  std::array<int32_t, exp_size> exp = {12};
  std::vector<uint64_t> in_shape    = {2, 4, 1};
  std::vector<uint64_t> out_shape   = {1};

  auto a_input  = cupynumeric::zeros(in_shape);
  auto a_output = cupynumeric::zeros(out_shape, legate::int32());
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  cupynumeric::trace(a_input, 0, 0, 1, std::nullopt, a_output);
  check_array_eq<int32_t, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Trace, DifferentOutType)
{
  constexpr size_t in_size                = 8;
  constexpr size_t in_dim                 = 3;
  constexpr size_t exp_size               = 1;
  constexpr size_t exp_dim                = 1;
  const std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  const std::array<int32_t, exp_size> exp = {12};
  const std::vector<uint64_t> in_shape    = {2, 4, 1};
  const std::vector<uint64_t> out_shape   = {1};

  auto a_input        = cupynumeric::zeros(in_shape);
  const auto a_output = cupynumeric::zeros(out_shape, legate::float64());

  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  cupynumeric::trace(
    a_input, /*offset=*/0, /*axis1=*/0, /*axis2=*/1, /*type=*/legate::int32(), /*out=*/a_output);
  check_array_eq<int32_t, exp_dim>(a_output, exp.data(), exp.size());
}

TEST(Trace, InvalidArray)
{
  const size_t in_size              = 8;
  const size_t in_dim               = 1;
  std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  std::vector<uint64_t> in_shape    = {8};

  auto a_input = cupynumeric::zeros(in_shape);
  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());
  EXPECT_THROW(cupynumeric::trace(a_input), std::invalid_argument);
}

TEST(Trace, InvalidOutputShape)
{
  constexpr size_t in_size                = 8;
  constexpr size_t in_dim                 = 3;
  constexpr size_t exp_size               = 1;
  constexpr size_t exp_dim                = 1;
  const std::array<double, in_size> input = {9, 7, 0.5, 1.3, 2, 3.6, 4, 5};
  const std::vector<uint64_t> in_shape    = {2, 4, 1};
  const std::vector<uint64_t> out_shape   = {2, 2};

  auto a_input = cupynumeric::zeros(in_shape);

  assign_values_to_array<double, in_dim>(a_input, input.data(), input.size());

  auto a_output = cupynumeric::zeros(out_shape);

  ASSERT_THAT(
    [&] {
      cupynumeric::trace(
        a_input, /*offset=*/0, /*axis1=*/0, /*axis2=*/1, /*type=*/std::nullopt, /*out=*/a_output);
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("output array has the wrong shape")));
}
