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

auto get_flip_expect_result_int()
{
  std::vector<std::map<int32_t, std::array<int32_t, 12>>> expect_result = {
    {{0, {1, 11, 6, 7, 9, 8, 4, 2, 5, 12, 3, 10}}},
    {{0, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}, {1, {1, 11, 6, 7, 9, 8, 4, 2, 5, 12, 3, 10}}},
    {{0, {1, 11, 6, 7, 9, 8, 4, 2, 5, 12, 3, 10}}, {1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}},
    {{0, {7, 6, 11, 1, 2, 4, 8, 9, 10, 3, 12, 5}}, {1, {5, 12, 3, 10, 9, 8, 4, 2, 1, 11, 6, 7}}},
    {{0, {1, 11, 6, 7, 9, 8, 4, 2, 5, 12, 3, 10}},
     {1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {2, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}},
    {{0, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {1, {1, 11, 6, 7, 9, 8, 4, 2, 5, 12, 3, 10}},
     {2, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}},
    {{0, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {2, {1, 11, 6, 7, 9, 8, 4, 2, 5, 12, 3, 10}}},
    {{0, {8, 9, 7, 6, 11, 1, 10, 3, 12, 5, 2, 4}},
     {1, {5, 2, 4, 10, 3, 12, 6, 11, 1, 8, 9, 7}},
     {2, {12, 3, 10, 4, 2, 5, 7, 9, 8, 1, 11, 6}}}};
  return expect_result;
}

auto get_flip_expect_result_double()
{
  std::vector<std::map<int32_t, std::array<double, 12>>> expect_result = {
    {{0, {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5}}},
    {{0, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {1, {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5}}},
    {{0, {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5}},
     {1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}}},
    {{0, {7.9, 12, 9, 4, 2.2, 10.5, 8, 11, 1.5, 3.66, 6, 5.98}},
     {1, {5.98, 6, 3.66, 1.5, 11, 8, 10.5, 2.2, 4, 9, 12, 7.9}}},
    {{0, {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5}},
     {1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {2, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}}},
    {{0, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {1, {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5}},
     {2, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}}},
    {{0, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {2, {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5}}},
    {{0, {8, 11, 7.9, 12, 9, 4, 1.5, 3.66, 6, 5.98, 2.2, 10.5}},
     {1, {5.98, 2.2, 10.5, 1.5, 3.66, 6, 12, 9, 4, 8, 11, 7.9}},
     {2, {6, 3.66, 1.5, 10.5, 2.2, 5.98, 7.9, 11, 8, 4, 9, 12}}}};
  return expect_result;
}

auto get_flip_expect_result_complex()
{
  std::vector<std::map<int32_t, std::array<complex<float>, 12>>> expect_result = {
    {{0,
      {complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3)}}},
    {{0,
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
       complex<float>(6, 4)}},
     {1,
      {complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3)}}},
    {{0,
      {complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3)}},
     {1,
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
       complex<float>(6, 4)}}},
    {{0,
      {complex<float>(2.2, 10.5),
       complex<float>(8, 11),
       complex<float>(7.9, 12),
       complex<float>(6, 4),
       complex<float>(7, 6),
       complex<float>(11, 1),
       complex<float>(1.5, 3.66),
       complex<float>(6, 5.98),
       complex<float>(10, 3),
       complex<float>(12, 5),
       complex<float>(2, 4),
       complex<float>(8, 9)}},
     {1,
      {complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5)}}},
    {{0,
      {complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3)}},
     {1,
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
       complex<float>(6, 4)}},
     {2,
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
       complex<float>(6, 4)}}},
    {{0,
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
       complex<float>(6, 4)}},
     {1,
      {complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3)}},
     {2,
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
       complex<float>(6, 4)}}},
    {{0,
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
       complex<float>(6, 4)}},
     {1,
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
       complex<float>(6, 4)}},
     {2,
      {complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3)}}},
    {{0,
      {complex<float>(1.5, 3.66),
       complex<float>(6, 5.98),
       complex<float>(2.2, 10.5),
       complex<float>(8, 11),
       complex<float>(7.9, 12),
       complex<float>(6, 4),
       complex<float>(10, 3),
       complex<float>(12, 5),
       complex<float>(2, 4),
       complex<float>(8, 9),
       complex<float>(7, 6),
       complex<float>(11, 1)}},
     {1,
      {complex<float>(8, 9),
       complex<float>(7, 6),
       complex<float>(11, 1),
       complex<float>(10, 3),
       complex<float>(12, 5),
       complex<float>(2, 4),
       complex<float>(8, 11),
       complex<float>(7.9, 12),
       complex<float>(6, 4),
       complex<float>(1.5, 3.66),
       complex<float>(6, 5.98),
       complex<float>(2.2, 10.5)}},
     {2,
      {complex<float>(2, 4),
       complex<float>(12, 5),
       complex<float>(10, 3),
       complex<float>(11, 1),
       complex<float>(7, 6),
       complex<float>(8, 9),
       complex<float>(2.2, 10.5),
       complex<float>(6, 5.98),
       complex<float>(1.5, 3.66),
       complex<float>(6, 4),
       complex<float>(7.9, 12),
       complex<float>(8, 11)}}}};
  return expect_result;
}

template <typename T, int32_t SIZE, int32_t DIM>
void test_flip(std::array<T, SIZE>& in_array,
               std::array<T, SIZE>& expect,
               legate::Type leg_type,
               std::vector<uint64_t> shape,
               std::optional<std::vector<int32_t>> axis = std::nullopt)
{
  auto A1 = cupynumeric::zeros(shape, leg_type);
  if (in_array.size() != 0) {
    if (in_array.size() == 1) {
      A1.fill(legate::Scalar(in_array[0]));
    } else {
      assign_values_to_array<T, DIM>(A1, in_array.data(), in_array.size());
    }
  }

  auto B1 = cupynumeric::flip(A1, axis);
  check_array_eq<T, DIM>(B1, expect.data(), expect.size());
}

template <typename T, int32_t SIZE>
void test_flip_none_axis(std::vector<std::vector<uint64_t>>& test_shapes,
                         std::array<T, SIZE>& in_array,
                         std::array<T, SIZE>& expect_result,
                         legate::Type leg_type)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    if (dim == 1) {
      test_flip<T, SIZE, 1>(in_array, expect_result, leg_type, test_shape);
    } else if (dim == 2) {
      test_flip<T, SIZE, 2>(in_array, expect_result, leg_type, test_shape);
    } else if (dim == 3) {
      test_flip<T, SIZE, 3>(in_array, expect_result, leg_type, test_shape);
    } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
      test_flip<T, SIZE, 4>(in_array, expect_result, leg_type, test_shape);
#endif
    } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
      test_flip<T, SIZE, 5>(in_array, expect_result, leg_type, test_shape);
#endif
    } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
      test_flip<T, SIZE, 2>(in_array, expect_result, leg_type, test_shape);
#endif
    } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
      test_flip<T, SIZE, 2>(in_array, expect_result, leg_type, test_shape);
#endif
    }
  }
}

template <typename T, int32_t SIZE>
void test_flip_each_axis(std::vector<std::vector<uint64_t>>& test_shapes,
                         std::array<T, SIZE>& in_array,
                         std::vector<std::map<int32_t, std::array<T, SIZE>>>& expect_result,
                         legate::Type leg_type)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      auto index      = axis < 0 ? axis + dim : axis;
      auto expect_val = expect_result[i][index];
      auto axes       = {axis};
      if (dim == 1) {
        test_flip<T, SIZE, 1>(in_array, expect_val, leg_type, test_shape, axes);
      } else if (dim == 2) {
        test_flip<T, SIZE, 2>(in_array, expect_val, leg_type, test_shape, axes);
      } else if (dim == 3) {
        test_flip<T, SIZE, 3>(in_array, expect_val, leg_type, test_shape, axes);
      } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
        test_flip<T, SIZE, 4>(in_array, expect_val, leg_type, test_shape, axes);
#endif
      } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
        test_flip<T, SIZE, 5>(in_array, expect_val, leg_type, test_shape, axes);
#endif
      } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
        test_flip<T, SIZE, 2>(in_array, expect_val, leg_type, test_shape, axes);
#endif
      } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
        test_flip<T, SIZE, 2>(in_array, expect_val, leg_type, test_shape, axes);
#endif
      }
    }
  }
}

void flip_basic()
{
  // If no axis is input, the expect result would equal reverse result of the input array, no matter
  // what's the array shape is.
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};

  // Test int type
  std::array<int32_t, 12> in_array1 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  std::array<int32_t, 12> expect_result1;
  std::reverse_copy(in_array1.begin(), in_array1.end(), expect_result1.begin());
  test_flip_none_axis<int32_t, 12>(test_shapes, in_array1, expect_result1, legate::int32());

  // Test float type
  std::array<double, 12> int_array2 = {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4};
  std::array<double, 12> expect_result2;
  std::reverse_copy(int_array2.begin(), int_array2.end(), expect_result2.begin());
  test_flip_none_axis<double, 12>(test_shapes, int_array2, expect_result2, legate::float64());

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
  std::array<complex<float>, 12> expect_result3;
  std::reverse_copy(in_array3.begin(), in_array3.end(), expect_result3.begin());
  test_flip_none_axis<complex<float>, 12>(
    test_shapes, in_array3, expect_result3, legate::complex64());
}

void flip_single_axis()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};

  // Test int type
  std::array<int32_t, 12> in_array1 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  auto expect_result1               = get_flip_expect_result_int();
  test_flip_each_axis<int32_t, 12>(test_shapes, in_array1, expect_result1, legate::int32());

  // Test float type
  std::array<double, 12> int_array2 = {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4};
  auto expect_result2               = get_flip_expect_result_double();
  test_flip_each_axis<double, 12>(test_shapes, int_array2, expect_result2, legate::float64());

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
  auto expect_result3                      = get_flip_expect_result_complex();
  test_flip_each_axis<complex<float>, 12>(
    test_shapes, in_array3, expect_result3, legate::complex64());
}

void flip_multi_axis()
{
  // Test float type
  std::vector<uint64_t> test_shape = {2, 2, 3};
  std::array<double, 12> in_array  = {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4};

  auto axes1                            = {-1, 0};
  std::array<double, 12> expect_result1 = {7.9, 11, 8, 4, 9, 12, 6, 3.66, 1.5, 10.5, 2.2, 5.98};
  test_flip<double, 12, 3>(in_array, expect_result1, legate::float64(), test_shape, axes1);

  auto axes2                            = {-1, 1};
  std::array<double, 12> expect_result2 = {10.5, 2.2, 5.98, 6, 3.66, 1.5, 4, 9, 12, 7.9, 11, 8};
  test_flip<double, 12, 3>(in_array, expect_result2, legate::float64(), test_shape, axes2);

  auto axes3                            = {0, 1};
  std::array<double, 12> expect_result3 = {12, 9, 4, 8, 11, 7.9, 5.98, 2.2, 10.5, 1.5, 3.66, 6};
  test_flip<double, 12, 3>(in_array, expect_result3, legate::float64(), test_shape, axes3);

  auto axes4                            = {-1, 0, 1};
  std::array<double, 12> expect_result4 = {4, 9, 12, 7.9, 11, 8, 10.5, 2.2, 5.98, 6, 3.66, 1.5};
  test_flip<double, 12, 3>(in_array, expect_result4, legate::float64(), test_shape, axes4);
}

void flip_max_dim()
{
  std::array<int32_t, 16> in_array      = {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
  std::array<int32_t, 16> expect_result = {15, 1, 11, 6, 7, 9, 8, 16, 4, 2, 13, 5, 12, 3, 10, 14};
#if LEGATE_MAX_DIM >= 4
  std::vector<uint64_t> test_shape_4d = {2, 2, 2, 2};
  // Flip with none axis
  test_flip<int32_t, 16, 4>(in_array, expect_result, legate::int32(), test_shape_4d);
  // Flip with axis
  auto axes_4d                             = {2, 1, 3};
  std::array<int32_t, 16> expect_result_4d = {
    4, 2, 13, 5, 12, 3, 10, 14, 15, 1, 11, 6, 7, 9, 8, 16};
  test_flip<int32_t, 16, 4>(in_array, expect_result_4d, legate::int32(), test_shape_4d, axes_4d);
#endif

#if LEGATE_MAX_DIM >= 5
  std::vector<uint64_t> test_shape_5d = {1, 2, 2, 1, 4};
  // Flip with none axis
  test_flip<int32_t, 16, 5>(in_array, expect_result, legate::int32(), test_shape_5d);
  // Flip with axis
  auto axes_5d                             = {4};
  std::array<int32_t, 16> expect_result_5d = {
    12, 3, 10, 14, 4, 2, 13, 5, 7, 9, 8, 16, 15, 1, 11, 6};
  test_flip<int32_t, 16, 5>(in_array, expect_result_5d, legate::int32(), test_shape_5d, axes_5d);
#endif

#if LEGATE_MAX_DIM >= 6
  std::vector<uint64_t> test_shape_6d = {2, 1, 1, 2, 2, 2};
  // Flip with none axis
  test_flip<int32_t, 16, 6>(in_array, expect_result, legate::int32(), test_shape_6d);
  // Flip with axis
  auto axes_6d                             = {-1, -3, 0, 1};
  std::array<int32_t, 16> expect_result_6d = {
    11, 6, 15, 1, 8, 16, 7, 9, 13, 5, 4, 2, 10, 14, 12, 3};
  test_flip<int32_t, 16, 6>(in_array, expect_result_6d, legate::int32(), test_shape_6d, axes_6d);
#endif

#if LEGATE_MAX_DIM >= 7
  std::vector<uint64_t> test_shape_7d = {1, 16, 1, 1, 1, 1, 1};
  // Flip with none axis
  test_flip<int32_t, 16, 7>(in_array, expect_result, legate::int32(), test_shape_7d);
  // Flip with axis
  auto axes_7d                             = {0, 2, 3, 4, 5, 6};
  std::array<int32_t, 16> expect_result_7d = {
    14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
  test_flip<int32_t, 16, 7>(in_array, expect_result_7d, legate::int32(), test_shape_7d, axes_7d);
#endif
}

void flip_large_array()
{
  const int32_t count              = 10000;
  std::vector<uint64_t> test_shape = {count};

  // Test int type for large array
  std::array<int32_t, count> in_array1;
  for (int32_t i = 0; i < count; i++) {
    in_array1[i] = count - i;
  }
  std::array<int32_t, count> expect_val1;
  for (int32_t j = 0; j < count; j++) {
    expect_val1[j] = j + 1;
  }
  test_flip<int32_t, count, 1>(in_array1, expect_val1, legate::int32(), test_shape);

  // Test float type
  std::array<double, count> in_array2;
  for (int32_t i = 0; i < count; i++) {
    in_array2[i] = count * 1.0 - i;
  }
  std::array<double, count> expect_val2;
  for (int32_t j = 0; j < count; j++) {
    expect_val2[j] = (j + 1) * 1.0;
  }
  test_flip<double, count, 1>(in_array2, expect_val2, legate::float64(), test_shape);

  // Test complex type
  std::array<complex<float>, count> in_array3;
  for (int32_t i = 0; i < count; i++) {
    in_array3[i] = complex<float>(count - i, count - i);
  }
  std::array<complex<float>, count> expect_val3;
  for (int32_t j = 0; j < count; j++) {
    expect_val3[j] = complex<float>(j + 1, j + 1);
  }
  test_flip<complex<float>, count, 1>(in_array3, expect_val3, legate::complex64(), test_shape);
}

void flip_empty_array()
{
  std::array<int32_t, 0> in_array  = {};
  std::vector<uint64_t> test_shape = {0};

  // Without axis input
  test_flip<int32_t, 0, 1>(in_array, in_array, legate::int32(), test_shape);

  // With axis input
  auto axes = {0};
  test_flip<int32_t, 0, 1>(in_array, in_array, legate::int32(), test_shape, axes);
}

void flip_single_item_array()
{
  std::vector<uint64_t> test_shape = {1, 1, 1};
  std::array<int32_t, 1> in_array  = {12};

  // Without axis input
  test_flip<int32_t, 1, 3>(in_array, in_array, legate::int32(), test_shape);

  // With axis input
  auto axes1 = {1};
  test_flip<int32_t, 1, 3>(in_array, in_array, legate::int32(), test_shape, axes1);

  auto axes2 = {-1, 1};
  test_flip<int32_t, 1, 3>(in_array, in_array, legate::int32(), test_shape, axes2);

  auto axes3 = {-1, 0, 1};
  test_flip<int32_t, 1, 3>(in_array, in_array, legate::int32(), test_shape, axes3);
}

void flip_negative_test()
{
  auto in_array = cupynumeric::zeros({2, 3}, legate::int32());

  // Test axis out-of-bound
  auto axes1 = {12};
  EXPECT_THROW(cupynumeric::flip(in_array, axes1), std::invalid_argument);

  // Test axis out-of-bound negative
  auto axes2 = {-12};
  EXPECT_THROW(cupynumeric::flip(in_array, axes2), std::invalid_argument);

  // Test axis repeated axis
  auto axes3 = {1, 1};
  EXPECT_THROW(cupynumeric::flip(in_array, axes3), std::invalid_argument);

  // Test axis out-of-bound multiple
  auto axes4 = {1, 2};
  EXPECT_THROW(cupynumeric::flip(in_array, axes4), std::invalid_argument);
}

// void cpp_test()
TEST(Flip, Basic) { flip_basic(); }
TEST(Flip, Single_Axis) { flip_single_axis(); }
TEST(Flip, Multi_Axis) { flip_multi_axis(); }
TEST(Flip, MaxDim) { flip_max_dim(); }
TEST(Flip, LargeArray) { flip_large_array(); }
TEST(Flip, EmptyArray) { flip_empty_array(); }
TEST(Flip, SingleItemArray) { flip_single_item_array(); }
TEST(Flip, Negative) { flip_negative_test(); }
