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

template <typename T>
auto get_eye_expect_result_3_2()
{
  std::map<int32_t, std::array<T, 6>> expect_result = {{-30, {0, 0, 0, 0, 0, 0}},
                                                       {-3, {0, 0, 0, 0, 0, 0}},
                                                       {-2, {0, 0, 0, 0, 1, 0}},
                                                       {-1, {0, 0, 1, 0, 0, 1}},
                                                       {0, {1, 0, 0, 1, 0, 0}},
                                                       {1, {0, 1, 0, 0, 0, 0}},
                                                       {2, {0, 0, 0, 0, 0, 0}},
                                                       {3, {0, 0, 0, 0, 0, 0}},
                                                       {30, {0, 0, 0, 0, 0, 0}}};
  return expect_result;
}

template <typename T>
auto get_eye_expect_result_3_3()
{
  std::map<int32_t, std::array<T, 9>> expect_result = {{-30, {0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                                       {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                                       {-2, {0, 0, 0, 0, 0, 0, 1, 0, 0}},
                                                       {-1, {0, 0, 0, 1, 0, 0, 0, 1, 0}},
                                                       {0, {1, 0, 0, 0, 1, 0, 0, 0, 1}},
                                                       {1, {0, 1, 0, 0, 0, 1, 0, 0, 0}},
                                                       {2, {0, 0, 1, 0, 0, 0, 0, 0, 0}},
                                                       {3, {0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                                       {30, {0, 0, 0, 0, 0, 0, 0, 0, 0}}};
  return expect_result;
}

template <typename T>
auto get_eye_expect_result_3_4()
{
  std::map<int32_t, std::array<T, 12>> expect_result = {
    {-30, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {-2, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
    {-1, {0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
    {0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
    {1, {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
    {2, {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
    {3, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
    {30, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
  };
  return expect_result;
}

template <typename T>
auto test_eye_3_2(std::vector<int32_t>& k_vals, std::optional<legate::Type> type = std::nullopt)
{
  auto expect_result                 = get_eye_expect_result_3_2<T>();
  std::vector<uint64_t> expect_shape = {3, 2};
  for (auto k : k_vals) {
    if (type.has_value()) {
      auto result = cupynumeric::eye(3, 2, k, type.value());
      EXPECT_EQ(result.type(), type.value());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[k];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    } else {
      auto result = cupynumeric::eye(3, 2, k);
      EXPECT_EQ(result.type(), legate::float64());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[k];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    }
  }
}

template <typename T>
auto test_eye_3_3(std::vector<int32_t>& k_vals, std::optional<legate::Type> type = std::nullopt)
{
  auto expect_result                 = get_eye_expect_result_3_3<T>();
  std::vector<uint64_t> expect_shape = {3, 3};
  for (auto k : k_vals) {
    if (type.has_value()) {
      auto result = cupynumeric::eye(3, 3, k, type.value());
      EXPECT_EQ(result.type(), type.value());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[k];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    } else {
      auto result = cupynumeric::eye(3, 3, k);
      EXPECT_EQ(result.type(), legate::float64());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[k];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    }
  }
}

template <typename T>
auto test_eye_3_4(std::vector<int32_t>& k_vals, std::optional<legate::Type> type = std::nullopt)
{
  auto expect_result                 = get_eye_expect_result_3_4<T>();
  std::vector<uint64_t> expect_shape = {3, 4};
  for (auto k : k_vals) {
    if (type.has_value()) {
      auto result = cupynumeric::eye(3, 4, k, type.value());
      EXPECT_EQ(result.type(), type.value());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[k];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    } else {
      auto result = cupynumeric::eye(3, 4, k);
      EXPECT_EQ(result.type(), legate::float64());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[k];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    }
  }
}

template <typename T>
auto test_eye_square_3(std::optional<std::vector<int32_t>> k_vals = std::nullopt,
                       std::optional<legate::Type> type           = std::nullopt)
{
  auto expect_result                 = get_eye_expect_result_3_3<T>();
  std::vector<uint64_t> expect_shape = {3, 3};
  if (k_vals.has_value()) {
    for (auto k : k_vals.value()) {
      if (type.has_value()) {
        auto result = cupynumeric::eye(3, std::nullopt, k, type.value());
        EXPECT_EQ(result.type(), type.value());
        EXPECT_EQ(result.shape(), expect_shape);
        auto expect = expect_result[k];
        check_array_eq<T, 2>(result, expect.data(), expect.size());
      } else {
        auto result = cupynumeric::eye(3, std::nullopt, k);
        EXPECT_EQ(result.type(), legate::float64());
        auto expect = expect_result[k];
        check_array_eq<T, 2>(result, expect.data(), expect.size());
      }
    }
  } else {
    if (type.has_value()) {
      auto result = cupynumeric::eye(3, std::nullopt, 0, type.value());
      EXPECT_EQ(result.type(), type.value());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[0];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    } else {
      auto result = cupynumeric::eye(3);
      EXPECT_EQ(result.type(), legate::float64());
      EXPECT_EQ(result.shape(), expect_shape);
      auto expect = expect_result[0];
      check_array_eq<T, 2>(result, expect.data(), expect.size());
    }
  }
}

void eye_basic()
{
  std::vector<int32_t> k_vals = {-30, -3, -2, -1, 0, 1, 2, 3, 30};

  // Test default data type
  test_eye_3_2<double>(k_vals);
  test_eye_3_3<double>(k_vals);
  test_eye_3_4<double>(k_vals);

  // Test int type
  test_eye_3_2<int32_t>(k_vals, legate::int32());
  test_eye_3_3<int32_t>(k_vals, legate::int32());
  test_eye_3_4<int32_t>(k_vals, legate::int32());

  // Test complex type
  test_eye_3_2<legate::Complex<float>>(k_vals, legate::complex64());
  test_eye_3_3<legate::Complex<float>>(k_vals, legate::complex64());
  test_eye_3_4<legate::Complex<float>>(k_vals, legate::complex64());
}

void eye_square()
{
  std::vector<int32_t> k_vals = {-30, -3, -2, -1, 0, 1, 2, 3, 30};

  // Test default parameter
  test_eye_square_3<double>();

  // Test with k input
  test_eye_square_3<double>(k_vals);

  // Test with datatype input
  test_eye_square_3<int32_t>(std::nullopt, legate::int32());

  // Test with k and datatype input
  test_eye_square_3<legate::Complex<float>>(k_vals, legate::complex64());
}

void eye_input_zero()
{
  // Test n=0
  auto result1                        = cupynumeric::eye(0);
  std::vector<uint64_t> expect_shape1 = {0, 0};
  EXPECT_EQ(result1.type(), legate::float64());
  EXPECT_EQ(result1.size(), 0);
  EXPECT_EQ(result1.shape(), expect_shape1);

  // Test m=0
  auto result2                        = cupynumeric::eye(3, 0);
  std::vector<uint64_t> expect_shape2 = {3, 0};
  EXPECT_EQ(result2.type(), legate::float64());
  EXPECT_EQ(result2.size(), 0);
  EXPECT_EQ(result2.shape(), expect_shape2);
}

void eye_large_array()
{
  const size_t n_or_m = 1000;

  // Test 1000 * 1000 array
  auto result1                        = cupynumeric::eye(n_or_m);
  std::vector<uint64_t> expect_shape1 = {n_or_m, n_or_m};
  std::array<double, n_or_m * n_or_m> expect_result1;
  expect_result1.fill(0);
  for (size_t i = 0; i < n_or_m; i++) {
    expect_result1[i * n_or_m + i] = 1;
  }
  EXPECT_EQ(result1.type(), legate::float64());
  EXPECT_EQ(result1.shape(), expect_shape1);
  check_array_eq<double, 2>(result1, expect_result1.data(), expect_result1.size());

  // Test 3 * 1000 array
  const size_t n                      = 3;
  auto result2                        = cupynumeric::eye(n, n_or_m, 0, legate::int32());
  std::vector<uint64_t> expect_shape2 = {n, n_or_m};
  std::array<int32_t, n * n_or_m> expect_result2;
  expect_result2.fill(0);
  for (size_t i = 0; i < n; i++) {
    expect_result2[i * n_or_m + i] = 1;
  }
  EXPECT_EQ(result2.type(), legate::int32());
  EXPECT_EQ(result2.shape(), expect_shape2);
  check_array_eq<int32_t, 2>(result2, expect_result2.data(), expect_result2.size());

  // Test 1000 * 3 array
  const size_t m                      = 3;
  auto result3                        = cupynumeric::eye(n_or_m, m, 0, legate::complex64());
  std::vector<uint64_t> expect_shape3 = {n_or_m, m};
  std::array<legate::Complex<float>, n_or_m * m> expect_result3;
  expect_result3.fill(0);
  for (size_t i = 0; i < n_or_m; i++) {
    if (i < m) {
      expect_result3[i * m + i] = 1;
    }
  }
  EXPECT_EQ(result3.type(), legate::complex64());
  EXPECT_EQ(result3.shape(), expect_shape3);
  check_array_eq<legate::Complex<float>, 2>(result3, expect_result3.data(), expect_result3.size());
}

void eye_negative()
{
  // Test bad n
  EXPECT_THROW(cupynumeric::eye(-1), std::invalid_argument);
  EXPECT_THROW(cupynumeric::eye(-1, 3), std::invalid_argument);

  // Test bad m
  EXPECT_THROW(cupynumeric::eye(3, -1), std::invalid_argument);
  EXPECT_THROW(cupynumeric::eye(-1, -1), std::invalid_argument);

  // Test bad dtype
  EXPECT_THROW(cupynumeric::eye(3, std::nullopt, 0, legate::binary_type(2)), std::invalid_argument);
  EXPECT_THROW(cupynumeric::eye(3, std::nullopt, 0, legate::point_type(2)), std::invalid_argument);
}

// void cpp_test()
TEST(Eye, Basic) { eye_basic(); }
TEST(Eye, Square) { eye_square(); }
TEST(Eye, InputZero) { eye_input_zero(); }
TEST(Eye, LargeArray) { eye_large_array(); }
TEST(Eye, Negative) { eye_negative(); }
