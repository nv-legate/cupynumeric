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

auto get_msort_expect_result_int()
{
  std::vector<std::array<int32_t, 12>> expect_result = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                                        {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1},
                                                        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                                        {2, 3, 8, 1, 7, 4, 11, 5, 10, 6, 12, 9},
                                                        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                                        {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1},
                                                        {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1},
                                                        {8, 3, 7, 5, 2, 1, 10, 9, 12, 6, 11, 4}};
  return expect_result;
}

auto get_msort_expect_result_int_4d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {14, 8, 3, 7, 5, 11, 1, 4, 16, 10, 9, 12, 6, 13, 2, 15}};
  return expect_result;
}

auto get_msort_expect_result_int_5d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15}};
  return expect_result;
}

auto get_msort_expect_result_int_6d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {14, 8, 3, 7, 5, 11, 1, 4, 16, 10, 9, 12, 6, 13, 2, 15}};
  return expect_result;
}

auto get_msort_expect_result_int_7d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {5, 8, 1, 4, 6, 10, 2, 7, 14, 11, 3, 12, 16, 13, 9, 15},
    {14, 8, 3, 7, 5, 11, 1, 4, 16, 10, 9, 12, 6, 13, 2, 15}};
  return expect_result;
}

auto get_msort_expect_result_double()
{
  std::vector<std::array<double, 12>> expect_result = {
    {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12},
    {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4},
    {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12},
    {1.5, 3.66, 6, 4, 2.2, 10.5, 8, 5.98, 7.9, 12, 9, 11},
    {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12},
    {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4},
    {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4},
    {1.5, 3.66, 6, 5.98, 2.2, 4, 8, 11, 7.9, 12, 9, 10.5}};
  return expect_result;
}

auto get_msort_expect_result_complex()
{
  std::vector<std::array<complex<float>, 12>> expect_result = {{complex<float>(1.5, 3.66),
                                                                complex<float>(2, 4),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(6, 4),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(7, 6),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(8, 9),
                                                                complex<float>(8, 11),
                                                                complex<float>(10, 3),
                                                                complex<float>(11, 1),
                                                                complex<float>(12, 5)},
                                                               {complex<float>(10, 3),
                                                                complex<float>(12, 5),
                                                                complex<float>(2, 4),
                                                                complex<float>(8, 9),
                                                                complex<float>(7, 6),
                                                                complex<float>(11, 1),
                                                                complex<float>(1.5, 3.66),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(8, 11),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(6, 4)},
                                                               {complex<float>(1.5, 3.66),
                                                                complex<float>(2, 4),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(6, 4),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(7, 6),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(8, 9),
                                                                complex<float>(8, 11),
                                                                complex<float>(10, 3),
                                                                complex<float>(11, 1),
                                                                complex<float>(12, 5)},
                                                               {complex<float>(2.2, 10.5),
                                                                complex<float>(8, 11),
                                                                complex<float>(1.5, 3.66),
                                                                complex<float>(6, 4),
                                                                complex<float>(7, 6),
                                                                complex<float>(11, 1),
                                                                complex<float>(2, 4),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(10, 3),
                                                                complex<float>(12, 5),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(8, 9)},
                                                               {complex<float>(1.5, 3.66),
                                                                complex<float>(2, 4),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(6, 4),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(7, 6),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(8, 9),
                                                                complex<float>(8, 11),
                                                                complex<float>(10, 3),
                                                                complex<float>(11, 1),
                                                                complex<float>(12, 5)},
                                                               {complex<float>(10, 3),
                                                                complex<float>(12, 5),
                                                                complex<float>(2, 4),
                                                                complex<float>(8, 9),
                                                                complex<float>(7, 6),
                                                                complex<float>(11, 1),
                                                                complex<float>(1.5, 3.66),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(8, 11),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(6, 4)},
                                                               {complex<float>(10, 3),
                                                                complex<float>(12, 5),
                                                                complex<float>(2, 4),
                                                                complex<float>(8, 9),
                                                                complex<float>(7, 6),
                                                                complex<float>(11, 1),
                                                                complex<float>(1.5, 3.66),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(8, 11),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(6, 4)},
                                                               {complex<float>(1.5, 3.66),
                                                                complex<float>(6, 5.98),
                                                                complex<float>(2, 4),
                                                                complex<float>(8, 9),
                                                                complex<float>(7, 6),
                                                                complex<float>(6, 4),
                                                                complex<float>(10, 3),
                                                                complex<float>(12, 5),
                                                                complex<float>(2.2, 10.5),
                                                                complex<float>(8, 11),
                                                                complex<float>(7.9, 12),
                                                                complex<float>(11, 1)}};
  return expect_result;
}

template <typename T, int32_t SIZE, int32_t DIM>
void test_msort(std::array<T, SIZE>& in_array,
                std::array<T, SIZE>& expect,
                legate::Type leg_type,
                std::vector<uint64_t> shape)
{
  auto A1 = cupynumeric::zeros(shape, leg_type);
  if (in_array.size() != 0) {
    if (in_array.size() == 1) {
      A1.fill(legate::Scalar(in_array[0]));
    } else {
      assign_values_to_array<T, DIM>(A1, in_array.data(), in_array.size());
    }
  }

  auto B1 = cupynumeric::msort(A1);
  check_array_eq<T, DIM>(B1, expect.data(), expect.size());
}

template <typename T, int32_t SIZE>
void msort_basic_impl(std::vector<std::vector<uint64_t>>& test_shapes,
                      std::array<T, SIZE> in_array,
                      std::vector<std::array<T, SIZE>>& expect_result,
                      legate::Type leg_type)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    auto expect_val = expect_result[i];
    if (dim == 1) {
      test_msort<T, SIZE, 1>(in_array, expect_val, leg_type, test_shape);
    } else if (dim == 2) {
      test_msort<T, SIZE, 2>(in_array, expect_val, leg_type, test_shape);
    } else if (dim == 3) {
      test_msort<T, SIZE, 3>(in_array, expect_val, leg_type, test_shape);
    } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
      test_msort<T, SIZE, 4>(in_array, expect_val, leg_type, test_shape);
#endif
    } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
      test_msort<T, SIZE, 5>(in_array, expect_val, leg_type, test_shape);
#endif
    } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
      test_msort<T, SIZE, 6>(in_array, expect_val, leg_type, test_shape);
#endif
    } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
      test_msort<T, SIZE, 7>(in_array, expect_val, leg_type, test_shape);
#endif
    }
  }
}

void msort_basic()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};

  // Test int type
  std::array<int32_t, 12> in_array1 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  auto expect_result1               = get_msort_expect_result_int();
  msort_basic_impl<int32_t, 12>(test_shapes, in_array1, expect_result1, legate::int32());

  // Test float type
  std::array<double, 12> in_array2 = {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4};
  auto expect_result2              = get_msort_expect_result_double();
  msort_basic_impl<double, 12>(test_shapes, in_array2, expect_result2, legate::float64());

  // Test complex type
  std::array<complex<float>, 12> in_array3 = {complex<float>(10, 3),
                                              complex<float>(12, 5),
                                              complex<float>(2, 4),
                                              complex<float>(8, 9),
                                              complex<float>(7, 6),
                                              complex<float>(11, 1),
                                              complex<float>(1.5, 3.66),
                                              complex<float>(6, 5.98),
                                              complex<float>(2.2, 10.5),
                                              complex<float>(8, 11),
                                              complex<float>(7.9, 12),
                                              complex<float>(6, 4)};
  auto expect_result3                      = get_msort_expect_result_complex();
  msort_basic_impl<complex<float>, 12>(test_shapes, in_array3, expect_result3, legate::complex64());
}

void msort_basic_max_dim()
{
  // Only test int type for max dim
  std::array<int32_t, 16> in_array = {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
#if LEGATE_MAX_DIM >= 4
  std::vector<std::vector<uint64_t>> test_shapes_4d = {{1, 1, 1, 16}, {16, 1, 1, 1}, {2, 2, 1, 4}};
  auto expect_result_4d                             = get_msort_expect_result_int_4d();
  msort_basic_impl<int32_t, 16>(test_shapes_4d, in_array, expect_result_4d, legate::int32());
#endif

#if LEGATE_MAX_DIM >= 5
  std::vector<std::vector<uint64_t>> test_shapes_5d = {
    {16, 1, 1, 1, 1}, {1, 16, 1, 1, 1}, {1, 2, 2, 1, 4}};
  auto expect_result_5d = get_msort_expect_result_int_5d();
  msort_basic_impl<int32_t, 16>(test_shapes_5d, in_array, expect_result_5d, legate::int32());
#endif

#if LEGATE_MAX_DIM >= 6
  std::vector<std::vector<uint64_t>> test_shapes_6d = {
    {16, 1, 1, 1, 1, 1}, {1, 1, 16, 1, 1, 1}, {2, 1, 1, 2, 2, 2}};
  auto expect_result_6d = get_msort_expect_result_int_6d();
  msort_basic_impl<int32_t, 16>(test_shapes_6d, in_array, expect_result_6d, legate::int32());
#endif

#if LEGATE_MAX_DIM >= 7
  std::vector<std::vector<uint64_t>> test_shapes_7d = {
    {1, 16, 1, 1, 1, 1, 1}, {4, 1, 2, 2, 1, 1, 1}, {2, 2, 1, 1, 2, 1, 2}};
  auto expect_result_7d = get_msort_expect_result_int_7d();
  msort_basic_impl<int32_t, 16>(test_shapes_7d, in_array, expect_result_7d, legate::int32());
#endif
}

void msort_large_array()
{
  const int32_t count                            = 10000;
  std::vector<std::vector<uint64_t>> test_shapes = {{count}};

  // Test int type for large array
  std::array<int32_t, count> in_array1;
  for (int32_t i = 0; i < count; i++) {
    in_array1[i] = count - i;
  }
  std::array<int32_t, count> expect_val1;
  for (int32_t j = 0; j < count; j++) {
    expect_val1[j] = j + 1;
  }
  std::vector<std::array<int32_t, count>> expect_result1 = {expect_val1};
  msort_basic_impl<int32_t, count>(test_shapes, in_array1, expect_result1, legate::int32());

  // Test float type
  std::array<double, count> in_array2;
  for (int32_t i = 0; i < count; i++) {
    in_array2[i] = count * 1.0 - i;
  }
  std::array<double, count> expect_val2;
  for (int32_t j = 0; j < count; j++) {
    expect_val2[j] = (j + 1) * 1.0;
  }
  std::vector<std::array<double, count>> expect_result2 = {expect_val2};
  msort_basic_impl<double, count>(test_shapes, in_array2, expect_result2, legate::float64());

  // Test complex type
  std::array<complex<float>, count> in_array3;
  for (int32_t i = 0; i < count; i++) {
    in_array3[i] = complex<float>(count - i, count - i);
  }
  std::array<complex<float>, count> expect_val3;
  for (int32_t j = 0; j < count; j++) {
    expect_val3[j] = complex<float>(j + 1, j + 1);
  }
  std::vector<std::array<complex<float>, count>> expect_result3 = {expect_val3};
  msort_basic_impl<complex<float>, count>(
    test_shapes, in_array3, expect_result3, legate::complex64());
}

void msort_empty_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {0}, {0, 1}, {1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}};

  std::array<int32_t, 0> in_array = {};
  size_t test_shape_size          = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    if (dim == 1) {
      test_msort<int32_t, 0, 1>(in_array, in_array, legate::int32(), test_shape);
    } else if (dim == 2) {
      test_msort<int32_t, 0, 2>(in_array, in_array, legate::int32(), test_shape);
    } else {
      test_msort<int32_t, 0, 3>(in_array, in_array, legate::int32(), test_shape);
    }
  }
}

void msort_single_item_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {{1}, {1, 1}, {1, 1, 1}};

  std::array<int32_t, 1> in_array = {12};
  size_t test_shape_size          = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    if (dim == 1) {
      test_msort<int32_t, 1, 1>(in_array, in_array, legate::int32(), test_shape);
    } else if (dim == 2) {
      test_msort<int32_t, 1, 2>(in_array, in_array, legate::int32(), test_shape);
    } else {
      test_msort<int32_t, 1, 3>(in_array, in_array, legate::int32(), test_shape);
    }
  }
}

// void cpp_test()
TEST(Msort, Basic) { msort_basic(); }
TEST(Msort, BasicMaxDim) { msort_basic_max_dim(); }
TEST(Msort, LargeArray) { msort_large_array(); }
TEST(Msort, EmptyArray) { msort_empty_array(); }
TEST(Msort, SingleItemArray) { msort_single_item_array(); }
