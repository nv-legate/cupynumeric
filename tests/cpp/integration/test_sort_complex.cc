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
auto get_sort_complex_expect_result()
{
  std::vector<std::array<legate::Complex<T>, 12>> expect_result = {{legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(6, 4),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(8, 11),
                                                                    legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(12, 5)},
                                                                   {legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(6, 4),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(8, 11),
                                                                    legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(12, 5)},
                                                                   {legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(12, 5),
                                                                    legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(8, 11),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(6, 4)},
                                                                   {legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(12, 5),
                                                                    legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(6, 4),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(8, 11)},
                                                                   {legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(12, 5),
                                                                    legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(8, 11),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(6, 4)},
                                                                   {legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(12, 5),
                                                                    legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(8, 11),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(6, 4)},
                                                                   {legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(6, 4),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(8, 11),
                                                                    legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(12, 5)},
                                                                   {legate::Complex<T>(2, 4),
                                                                    legate::Complex<T>(10, 3),
                                                                    legate::Complex<T>(12, 5),
                                                                    legate::Complex<T>(7, 6),
                                                                    legate::Complex<T>(8, 9),
                                                                    legate::Complex<T>(11, 1),
                                                                    legate::Complex<T>(1.5, 3.66),
                                                                    legate::Complex<T>(2.2, 10.5),
                                                                    legate::Complex<T>(6, 5.98),
                                                                    legate::Complex<T>(6, 4),
                                                                    legate::Complex<T>(7.9, 12),
                                                                    legate::Complex<T>(8, 11)}};
  return expect_result;
}

template <typename T, int32_t SIZE>
auto change_int_to_complex(const std::vector<std::array<int32_t, SIZE>>& input)
{
  std::vector<std::array<legate::Complex<T>, SIZE>> results;
  for (size_t i = 0; i < input.size(); i++) {
    std::array<legate::Complex<T>, SIZE> result;
    for (size_t j = 0; j < input[i].size(); j++) {
      result[j] = legate::Complex<T>(input[i][j], 0);
    }
    results.push_back(result);
  }
  return results;
}

template <typename T>
auto get_sort_complex_expect_result_from_int()
{
  std::vector<std::array<int32_t, 12>> expect_result = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                                        {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1},
                                                        {3, 5, 10, 12, 2, 4, 8, 9, 1, 6, 7, 11},
                                                        {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1},
                                                        {3, 10, 12, 2, 4, 5, 7, 8, 9, 1, 6, 11}};

  return change_int_to_complex<T, 12>(expect_result);
}

auto get_sort_complex_expect_result_4d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {3, 10, 12, 14, 2, 4, 5, 13, 7, 8, 9, 16, 1, 6, 11, 15}};

  return change_int_to_complex<float, 16>(expect_result);
}

auto get_sort_complex_expect_result_5d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {3, 10, 12, 14, 2, 4, 5, 13, 7, 8, 9, 16, 1, 6, 11, 15}};

  return change_int_to_complex<float, 16>(expect_result);
}

auto get_sort_complex_expect_result_6d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {10, 14, 3, 12, 5, 13, 2, 4, 8, 16, 7, 9, 6, 11, 1, 15}};

  return change_int_to_complex<float, 16>(expect_result);
}

auto get_sort_complex_expect_result_7d()
{
  std::vector<std::array<int32_t, 16>> expect_result = {
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15},
    {10, 14, 3, 12, 5, 13, 2, 4, 8, 16, 7, 9, 6, 11, 1, 15}};

  return change_int_to_complex<float, 16>(expect_result);
}

template <typename T_IN, typename T_OUT, int32_t SIZE, int32_t DIM>
void test_sort_complex(std::array<T_IN, SIZE>& in_array,
                       std::array<T_OUT, SIZE>& expect,
                       legate::Type leg_type,
                       std::vector<uint64_t> shape)
{
  auto A1 = cupynumeric::zeros(shape, leg_type);
  if (in_array.size() != 0) {
    if (in_array.size() == 1) {
      A1.fill(legate::Scalar(in_array[0]));
    } else {
      assign_values_to_array<T_IN, DIM>(A1, in_array.data(), in_array.size());
    }
  }
  auto B1 = cupynumeric::sort_complex(A1);
  if (in_array.size() != 0) {
    check_array_eq<T_OUT, DIM>(B1, expect.data(), expect.size());
  }
}

template <typename T_IN, typename T_OUT, int32_t SIZE>
void sort_complex_basic_impl(std::vector<std::vector<uint64_t>>& test_shapes,
                             std::array<T_IN, SIZE> in_array,
                             std::vector<std::array<T_OUT, SIZE>>& expect_result,
                             legate::Type leg_type)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    auto expect_val = expect_result[i];
    if (dim == 1) {
      test_sort_complex<T_IN, T_OUT, SIZE, 1>(in_array, expect_val, leg_type, test_shape);
    } else if (dim == 2) {
      test_sort_complex<T_IN, T_OUT, SIZE, 2>(in_array, expect_val, leg_type, test_shape);
    } else if (dim == 3) {
      test_sort_complex<T_IN, T_OUT, SIZE, 3>(in_array, expect_val, leg_type, test_shape);
    } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
      test_sort_complex<T_IN, T_OUT, SIZE, 4>(in_array, expect_val, leg_type, test_shape);
#endif
    } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
      test_sort_complex<T_IN, T_OUT, SIZE, 5>(in_array, expect_val, leg_type, test_shape);
#endif
    } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
      test_sort_complex<T_IN, T_OUT, SIZE, 6>(in_array, expect_val, leg_type, test_shape);
#endif
    } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
      test_sort_complex<T_IN, T_OUT, SIZE, 7>(in_array, expect_val, leg_type, test_shape);
#endif
    }
  }
}

void sort_complex_basic()
{
  // Test int8 type
  std::vector<std::vector<uint64_t>> test_shapes_int = {
    {12}, {12, 1}, {3, 4}, {12, 1, 1}, {2, 2, 3}};
  std::array<int8_t, 12> in_array1 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  auto expect_result1              = get_sort_complex_expect_result_from_int<float>();
  sort_complex_basic_impl<int8_t, legate::Complex<float>, 12>(
    test_shapes_int, in_array1, expect_result1, legate::int8());

  // Test int16 type
  std::array<int16_t, 12> in_array2 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  auto expect_result2               = get_sort_complex_expect_result_from_int<float>();
  sort_complex_basic_impl<int16_t, legate::Complex<float>, 12>(
    test_shapes_int, in_array2, expect_result2, legate::int16());

  // Test int32 type
  std::array<int32_t, 12> int_array3 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  auto expect_result3                = get_sort_complex_expect_result_from_int<double>();
  sort_complex_basic_impl<int32_t, legate::Complex<double>, 12>(
    test_shapes_int, int_array3, expect_result3, legate::int32());

  // Test complex type
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};

  std::array<legate::Complex<float>, 12> in_array4 = {legate::Complex<float>(10, 3),
                                                      legate::Complex<float>(12, 5),
                                                      legate::Complex<float>(2, 4),
                                                      legate::Complex<float>(8, 9),
                                                      legate::Complex<float>(7, 6),
                                                      legate::Complex<float>(11, 1),
                                                      legate::Complex<float>(1.5, 3.66),
                                                      legate::Complex<float>(6, 5.98),
                                                      legate::Complex<float>(2.2, 10.5),
                                                      legate::Complex<float>(8, 11),
                                                      legate::Complex<float>(7.9, 12),
                                                      legate::Complex<float>(6, 4)};
  auto expect_result4                              = get_sort_complex_expect_result<float>();
  sort_complex_basic_impl<legate::Complex<float>, legate::Complex<float>, 12>(
    test_shapes, in_array4, expect_result4, legate::complex64());

  std::array<legate::Complex<double>, 12> in_array5 = {legate::Complex<double>(10, 3),
                                                       legate::Complex<double>(12, 5),
                                                       legate::Complex<double>(2, 4),
                                                       legate::Complex<double>(8, 9),
                                                       legate::Complex<double>(7, 6),
                                                       legate::Complex<double>(11, 1),
                                                       legate::Complex<double>(1.5, 3.66),
                                                       legate::Complex<double>(6, 5.98),
                                                       legate::Complex<double>(2.2, 10.5),
                                                       legate::Complex<double>(8, 11),
                                                       legate::Complex<double>(7.9, 12),
                                                       legate::Complex<double>(6, 4)};
  auto expect_result5                               = get_sort_complex_expect_result<double>();
  sort_complex_basic_impl<legate::Complex<double>, legate::Complex<double>, 12>(
    test_shapes, in_array5, expect_result5, legate::complex128());
}

void sort_complex_basic_max_dim()
{
  // Only test int type for max dim
  std::array<int16_t, 16> in_array = {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
#if LEGATE_MAX_DIM >= 4
  std::vector<std::vector<uint64_t>> test_shapes_4d = {{1, 1, 1, 16}, {16, 1, 1, 1}, {2, 2, 1, 4}};
  auto expect_result_4d                             = get_sort_complex_expect_result_4d();
  sort_complex_basic_impl<int16_t, legate::Complex<float>, 16>(
    test_shapes_4d, in_array, expect_result_4d, legate::int16());
#endif

#if LEGATE_MAX_DIM >= 5
  std::vector<std::vector<uint64_t>> test_shapes_5d = {
    {1, 1, 1, 1, 16}, {1, 16, 1, 1, 1}, {1, 2, 2, 1, 4}};
  auto expect_result_5d = get_sort_complex_expect_result_5d();
  sort_complex_basic_impl<int16_t, legate::Complex<float>, 16>(
    test_shapes_5d, in_array, expect_result_5d, legate::int16());
#endif

#if LEGATE_MAX_DIM >= 6
  std::vector<std::vector<uint64_t>> test_shapes_6d = {
    {1, 1, 1, 1, 1, 16}, {1, 1, 16, 1, 1, 1}, {2, 1, 1, 2, 2, 2}};
  auto expect_result_6d = get_sort_complex_expect_result_6d();
  sort_complex_basic_impl<int16_t, legate::Complex<float>, 16>(
    test_shapes_6d, in_array, expect_result_6d, legate::int16());
#endif

#if LEGATE_MAX_DIM >= 7
  std::vector<std::vector<uint64_t>> test_shapes_7d = {
    {1, 16, 1, 1, 1, 1, 1}, {4, 1, 2, 2, 1, 1, 1}, {2, 2, 1, 1, 2, 1, 2}};
  auto expect_result_7d = get_sort_complex_expect_result_7d();
  sort_complex_basic_impl<int16_t, legate::Complex<float>, 16>(
    test_shapes_7d, in_array, expect_result_7d, legate::int16());
#endif
}

void sort_complex_large_array()
{
  const int32_t count                            = 10000;
  std::vector<std::vector<uint64_t>> test_shapes = {{count}};

  // Test int16 type for large array
  std::array<int16_t, count> in_array1;
  for (int16_t i = 0; i < count; i++) {
    in_array1[i] = count - i;
  }
  std::array<legate::Complex<float>, count> expect_val1;
  for (int32_t j = 0; j < count; j++) {
    expect_val1[j] = legate::Complex<float>(j + 1, 0);
  }
  std::vector<std::array<legate::Complex<float>, count>> expect_result1 = {expect_val1};
  sort_complex_basic_impl<int16_t, legate::Complex<float>, count>(
    test_shapes, in_array1, expect_result1, legate::int16());

  // Test int32 type for large array
  std::array<int32_t, count> in_array2;
  for (int32_t i = 0; i < count; i++) {
    in_array2[i] = count - i;
  }
  std::array<legate::Complex<double>, count> expect_val2;
  for (int32_t j = 0; j < count; j++) {
    expect_val2[j] = legate::Complex<double>(j + 1, 0);
  }
  std::vector<std::array<legate::Complex<double>, count>> expect_result2 = {expect_val2};
  sort_complex_basic_impl<int32_t, legate::Complex<double>, count>(
    test_shapes, in_array2, expect_result2, legate::int32());

  // Test complex type
  std::array<legate::Complex<float>, count> in_array3;
  for (int32_t i = 0; i < count; i++) {
    in_array3[i] = legate::Complex<float>(count - i, count - i);
  }
  std::array<legate::Complex<float>, count> expect_val3;
  for (int32_t j = 0; j < count; j++) {
    expect_val3[j] = legate::Complex<float>(j + 1, j + 1);
  }
  std::vector<std::array<legate::Complex<float>, count>> expect_result3 = {expect_val3};
  sort_complex_basic_impl<legate::Complex<float>, legate::Complex<float>, count>(
    test_shapes, in_array3, expect_result3, legate::complex64());
}

void sort_complex_empty_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {0}, {0, 1}, {1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}};

  std::array<legate::Complex<float>, 0> in_array = {};
  size_t test_shape_size                         = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    if (dim == 1) {
      test_sort_complex<legate::Complex<float>, legate::Complex<float>, 0, 1>(
        in_array, in_array, legate::complex64(), test_shape);
    } else if (dim == 2) {
      test_sort_complex<legate::Complex<float>, legate::Complex<float>, 0, 2>(
        in_array, in_array, legate::complex64(), test_shape);
    } else {
      test_sort_complex<legate::Complex<float>, legate::Complex<float>, 0, 3>(
        in_array, in_array, legate::complex64(), test_shape);
    }
  }
}

void sort_complex_single_item_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {{1}, {1, 1}, {1, 1, 1}};

  std::array<double, 1> in_array                       = {12};
  std::array<legate::Complex<double>, 1> expect_result = {legate::Complex<double>(12, 0)};
  size_t test_shape_size                               = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    if (dim == 1) {
      test_sort_complex<double, legate::Complex<double>, 1, 1>(
        in_array, expect_result, legate::float64(), test_shape);
    } else if (dim == 2) {
      test_sort_complex<double, legate::Complex<double>, 1, 2>(
        in_array, expect_result, legate::float64(), test_shape);
    } else {
      test_sort_complex<double, legate::Complex<double>, 1, 3>(
        in_array, expect_result, legate::float64(), test_shape);
    }
  }
}

// void cpp_test()
TEST(SortComplex, Basic) { sort_complex_basic(); }
TEST(SortComplex, BasicMaxDim) { sort_complex_basic_max_dim(); }
TEST(SortComplex, LargeArray) { sort_complex_large_array(); }
TEST(SortComplex, EmptyArray) { sort_complex_empty_array(); }
TEST(SortComplex, SingleItemArray) { sort_complex_single_item_array(); }
