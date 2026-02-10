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
#include "common_utils.h"

namespace {

TEST(Operators, PlusArray1D)
{
  const auto a = cupynumeric::mk_array<double>({1.0, 2.0, 3.0, 4.0});
  const auto b = cupynumeric::mk_array<double>({5.0, 6.0, 7.0, 8.0});

  const std::vector<double> expect{6.0, 8.0, 10.0, 12.0};

  const auto result = a + b;

  cupynumeric::check_array_near(result, expect);
}

TEST(Operators, PlusArray2D)
{
  const std::vector<uint64_t> shape{2, 3};
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4, 5, 6}, shape);
  const auto b = cupynumeric::mk_array<int32_t>({10, 20, 30, 40, 50, 60}, shape);

  const std::vector<int32_t> expect{11, 22, 33, 44, 55, 66};

  const auto result = a + b;

  cupynumeric::check_array(result, expect, shape);
}

TEST(Operators, PlusScalar)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4});
  const auto b = legate::Scalar(int32_t(10));

  const std::vector<int32_t> expect{11, 12, 13, 14};

  const auto result = a + b;

  cupynumeric::check_array(result, expect);
}

TEST(Operators, PlusEmptyArray)
{
  const std::vector<uint64_t> shape{0};
  const auto a = cupynumeric::mk_array<double>({}, shape);
  const auto b = cupynumeric::mk_array<double>({}, shape);

  const std::vector<double> expect{};

  const auto result = a + b;

  cupynumeric::check_array_near(result, expect, shape);
}

TEST(Operators, PlusDifferentTypes)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4});
  const auto b = cupynumeric::mk_array<double>({5.0, 6.0, 7.0, 8.0});

  ASSERT_THAT([&] { a + b; },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Operands must have the same type")));
}

TEST(Operators, PlusEqualsArray1D)
{
  auto a       = cupynumeric::mk_array<double>({1.0, 2.0, 3.0, 4.0});
  const auto b = cupynumeric::mk_array<double>({5.0, 6.0, 7.0, 8.0});

  const std::vector<double> expect{6.0, 8.0, 10.0, 12.0};

  a += b;

  cupynumeric::check_array_near(a, expect);
}

TEST(Operators, PlusEqualsArray2D)
{
  const std::vector<uint64_t> shape{2, 2};
  auto a       = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, shape);
  const auto b = cupynumeric::mk_array<int32_t>({10, 20, 30, 40}, shape);

  const std::vector<int32_t> expect{11, 22, 33, 44};

  a += b;

  cupynumeric::check_array(a, expect, shape);
}

TEST(Operators, MultiplyArray1D)
{
  auto a       = cupynumeric::mk_array<double>({2.0, 3.0, 4.0, 5.0});
  const auto b = cupynumeric::mk_array<double>({10.0, 10.0, 10.0, 10.0});

  const std::vector<double> expect{20.0, 30.0, 40.0, 50.0};

  const auto result = a * b;

  cupynumeric::check_array_near(result, expect);
}

TEST(Operators, MultiplyArray2D)
{
  const std::vector<uint64_t> shape{2, 2};
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, shape);
  const auto b = cupynumeric::mk_array<int32_t>({2, 2, 2, 2}, shape);

  const std::vector<int32_t> expect{2, 4, 6, 8};

  const auto result = a * b;

  cupynumeric::check_array(result, expect, shape);
}

TEST(Operators, MultiplyScalar)
{
  const auto a = cupynumeric::mk_array<int32_t>({2, 3, 4});
  const auto b = legate::Scalar(int32_t(5));

  const std::vector<int32_t> expect{10, 15, 20};

  const auto result = a * b;

  cupynumeric::check_array(result, expect);
}

TEST(Operators, MultiplyEmptyArray)
{
  const std::vector<uint64_t> shape{0};
  const auto a = cupynumeric::mk_array<int32_t>({}, shape);
  const auto b = cupynumeric::mk_array<int32_t>({}, shape);

  const std::vector<int32_t> expect{};

  const auto result = a * b;

  cupynumeric::check_array(result, expect, shape);
}

TEST(Operators, MultiplyEqualsArray1D)
{
  auto a       = cupynumeric::mk_array<double>({2.0, 3.0, 4.0, 5.0});
  const auto b = cupynumeric::mk_array<double>({2.0, 2.0, 2.0, 2.0});

  const std::vector<double> expect{4.0, 6.0, 8.0, 10.0};

  a *= b;

  cupynumeric::check_array_near(a, expect);
}

TEST(Operators, MultiplyEqualsArray2D)
{
  const std::vector<uint64_t> shape{2, 2};
  auto a       = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, shape);
  const auto b = cupynumeric::mk_array<int32_t>({10, 10, 10, 10}, shape);

  const std::vector<int32_t> expect{10, 20, 30, 40};

  a *= b;

  cupynumeric::check_array(a, expect, shape);
}

TEST(Operators, DivideArray1D)
{
  const auto a = cupynumeric::mk_array<double>({10.0, 20.0, 30.0, 40.0});
  const auto b = cupynumeric::mk_array<double>({2.0, 4.0, 5.0, 8.0});

  const std::vector<double> expect{5.0, 5.0, 6.0, 5.0};

  const auto result = a / b;

  cupynumeric::check_array_near(result, expect);
}

TEST(Operators, DivideArray2D)
{
  const std::vector<uint64_t> shape{2, 2};
  const auto a = cupynumeric::mk_array<double>({10.0, 20.0, 30.0, 40.0}, shape);
  const auto b = cupynumeric::mk_array<double>({2.0, 2.0, 3.0, 4.0}, shape);

  const std::vector<double> expect{5.0, 10.0, 10.0, 10.0};

  const auto result = a / b;

  cupynumeric::check_array_near(result, expect, shape);
}

TEST(Operators, DivideScalar)
{
  const auto a = cupynumeric::mk_array<float>({10.0f, 20.0f, 30.0f, 40.0f});
  const auto b = legate::Scalar(2.0f);

  const std::vector<float> expect{5.0f, 10.0f, 15.0f, 20.0f};

  const auto result = a / b;

  cupynumeric::check_array_near(result, expect);
}

TEST(Operators, DivideEmptyArray)
{
  const std::vector<uint64_t> shape{0};
  const auto a = cupynumeric::mk_array<float>({}, shape);
  const auto b = cupynumeric::mk_array<float>({}, shape);

  const std::vector<float> expect{};

  const auto result = a / b;

  cupynumeric::check_array_near(result, expect, shape);
}

TEST(Operators, SubscriptArray1D)
{
  const auto a = cupynumeric::mk_array<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  const std::vector<int32_t> expect{2, 3, 4, 5, 6};

  const auto sliced = a[{cupynumeric::slice(2, 7)}];

  cupynumeric::check_array(sliced, expect);
}

TEST(Operators, SubscriptArray1DFullSlice)
{
  const auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4, 5});

  const std::vector<int32_t> expect{1, 2, 3, 4, 5};

  const auto sliced = a[{cupynumeric::slice()}];

  cupynumeric::check_array(sliced, expect);
}

TEST(Operators, SubscriptArray2D)
{
  const std::vector<uint64_t> shape{3, 4};
  const auto a = cupynumeric::mk_array<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, shape);

  const auto sliced = a[{cupynumeric::slice(0, 2), cupynumeric::slice(1, 3)}];

  const std::vector<int32_t> expect{1, 2, 5, 6};
  const std::vector<uint64_t> sliced_shape{2, 2};

  cupynumeric::check_array(sliced, expect, sliced_shape);
}

TEST(Operators, SubscriptArray2DColumns)
{
  const std::vector<uint64_t> shape{2, 5};
  const auto a = cupynumeric::mk_array<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, shape);

  const auto sliced = a[{cupynumeric::slice(), cupynumeric::slice(1, 4)}];

  const std::vector<int32_t> expect{1, 2, 3, 6, 7, 8};
  const std::vector<uint64_t> sliced_shape{2, 3};

  cupynumeric::check_array(sliced, expect, sliced_shape);
}

TEST(Operators, SubscriptTooManySlices)
{
  auto a = cupynumeric::mk_array<int32_t>({1, 2, 3, 4}, {2, 2});

  ASSERT_THAT(
    [&] {
      (a[{cupynumeric::slice(), cupynumeric::slice(), cupynumeric::slice()}]);
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Can't slice a 2-D ndarray with 3 slices")));
}

TEST(Operators, BoolScalarTrue)
{
  auto a = cupynumeric::mk_array<bool>({true});

  EXPECT_TRUE(a);
}

TEST(Operators, BoolScalarFalse)
{
  auto a = cupynumeric::mk_array<int32_t>({0});

  EXPECT_FALSE(a);
}

}  // namespace
