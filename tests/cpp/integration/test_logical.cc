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

template <typename T,
          typename OUT_T,
          int32_t IN_SIZE,
          int32_t OUT_SIZE,
          int32_t IN_DIM,
          int32_t OUT_DIM>
void test_all(std::array<T, IN_SIZE>& in_array,
              std::array<OUT_T, OUT_SIZE>& expect_result,
              legate::Type leg_type,
              std::vector<uint64_t> shape,
              std::vector<int32_t> axis                 = {},
              std::optional<cupynumeric::NDArray> out   = std::nullopt,
              bool keepdims                             = false,
              std::optional<cupynumeric::NDArray> where = std::nullopt)
{
  auto A1 = cupynumeric::zeros(shape, leg_type);
  if (in_array.size() != 0) {
    if (in_array.size() == 1) {
      A1.fill(legate::Scalar(in_array[0]));
    } else {
      assign_values_to_array<T, IN_DIM>(A1, in_array.data(), in_array.size());
    }
  }

  if (!out.has_value()) {
    auto B1 = cupynumeric::all(A1, axis, std::nullopt, keepdims, where);
    check_array_eq<OUT_T, OUT_DIM>(B1, expect_result.data(), expect_result.size());
  } else {
    cupynumeric::all(A1, axis, out, keepdims, where);
    check_array_eq<OUT_T, OUT_DIM>(out.value(), expect_result.data(), expect_result.size());
  }
}

template <typename T, int32_t IN_SIZE, int32_t OUT_SIZE>
void test_all_each_axis(std::array<T, IN_SIZE>& in_array,
                        std::map<int32_t, std::array<bool, OUT_SIZE>>& expect_result,
                        legate::Type leg_type,
                        std::vector<uint64_t> shape,
                        bool keepdims = false)
{
  int32_t dim = shape.size();
  for (int32_t axis = -dim + 1; axis < dim; ++axis) {
    auto index      = axis < 0 ? axis + dim : axis;
    auto expect_val = expect_result[index];
    auto axes       = {axis};
    if (keepdims) {
      if (dim == 1) {
        test_all<T, bool, IN_SIZE, OUT_SIZE, 1, 1>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
      } else if (dim == 2) {
        test_all<T, bool, IN_SIZE, OUT_SIZE, 2, 2>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
      } else if (dim == 3) {
        test_all<T, bool, IN_SIZE, OUT_SIZE, 3, 3>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
      } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
        test_all<T, bool, IN_SIZE, OUT_SIZE, 4, 4>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
        test_all<T, bool, IN_SIZE, OUT_SIZE, 5, 5>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
        test_all<T, bool, IN_SIZE, OUT_SIZE, 6, 6>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
        test_all<T, bool, IN_SIZE, OUT_SIZE, 7, 7>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      }
    } else {
      if (dim == 1) {
        test_all<T, bool, IN_SIZE, OUT_SIZE, 1, 1>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
      } else if (dim == 2) {
        test_all<T, bool, IN_SIZE, OUT_SIZE, 2, 1>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
      } else if (dim == 3) {
        test_all<T, bool, IN_SIZE, OUT_SIZE, 3, 2>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
      } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
        test_all<T, bool, IN_SIZE, OUT_SIZE, 4, 3>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
        test_all<T, bool, IN_SIZE, OUT_SIZE, 5, 4>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
        test_all<T, bool, IN_SIZE, OUT_SIZE, 6, 5>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
        test_all<T, bool, IN_SIZE, OUT_SIZE, 7, 6>(
          in_array, expect_val, leg_type, shape, axes, std::nullopt, keepdims);
#endif
      }
    }
  }
}

void test_all_basic()
{
  // Test int type
  std::array<int32_t, 3> in_array1                      = {-1, 4, 5};
  std::vector<uint64_t> shape1                          = {3};
  std::map<int32_t, std::array<bool, 1>> expect_result1 = {{0, {true}}};
  test_all_each_axis<int32_t, 3, 1>(in_array1, expect_result1, legate::int32(), shape1);
  test_all_each_axis<int32_t, 3, 1>(in_array1, expect_result1, legate::int32(), shape1, true);

  std::array<int32_t, 4> in_array2                      = {5, 10, 0, 100};
  std::map<int32_t, std::array<bool, 1>> expect_result2 = {{0, {false}}};
  std::vector<uint64_t> shape2                          = {4};
  test_all_each_axis<int32_t, 4, 1>(in_array2, expect_result2, legate::int32(), shape2);
  test_all_each_axis<int32_t, 4, 1>(in_array2, expect_result2, legate::int32(), shape2, true);

  std::array<int32_t, 4> in_array3                      = {0, 0, 0, 0};
  std::map<int32_t, std::array<bool, 2>> expect_result3 = {{0, {false, false}},
                                                           {1, {false, false}}};
  std::vector<uint64_t> shape3                          = {2, 2};
  test_all_each_axis<int32_t, 4, 2>(in_array3, expect_result3, legate::int32(), shape3);
  test_all_each_axis<int32_t, 4, 2>(in_array3, expect_result3, legate::int32(), shape3, true);

  std::array<int32_t, 8> in_array4                      = {0, 1, 2, 3, 4, 0, 6, 7};
  std::map<int32_t, std::array<bool, 4>> expect_result4 = {{0, {false, false, true, true}},
                                                           {1, {false, true, true, false}},
                                                           {2, {false, true, false, true}}};
  std::vector<uint64_t> shape4                          = {2, 2, 2};
  test_all_each_axis<int32_t, 8, 4>(in_array4, expect_result4, legate::int32(), shape4);
  test_all_each_axis<int32_t, 8, 4>(in_array4, expect_result4, legate::int32(), shape4, true);

  // Test bool type
  std::array<bool, 9> in_array5 = {true, true, false, true, true, true, true, true, false};
  std::map<int32_t, std::array<bool, 3>> expect_result5 = {{0, {true, true, false}},
                                                           {1, {false, true, false}}};
  std::vector<uint64_t> shape5                          = {3, 3};
  test_all_each_axis<bool, 9, 3>(in_array5, expect_result5, legate::bool_(), shape5);
  test_all_each_axis<bool, 9, 3>(in_array5, expect_result5, legate::bool_(), shape5, true);

  // Test float type
  std::array<double, 9> in_array6 = {0.0, 1.0, 0.0, 5.0, 2.0, 1.0, 1.0, 2.0, 3.0};
  std::map<int32_t, std::array<bool, 3>> expect_result6 = {{0, {false, true, false}},
                                                           {1, {false, true, true}}};
  std::vector<uint64_t> shape6                          = {3, 3};
  test_all_each_axis<double, 9, 3>(in_array6, expect_result6, legate::float64(), shape6);
  test_all_each_axis<double, 9, 3>(in_array6, expect_result6, legate::float64(), shape6, true);

  // Test complex type
  std::array<complex<float>, 4> in_array7 = {
    complex<float>(0, 1), complex<float>(1, 1), complex<float>(1, 0), complex<float>(0, 0)};
  std::map<int32_t, std::array<bool, 2>> expect_result7 = {{0, {true, false}}, {1, {true, false}}};
  std::vector<uint64_t> shape7                          = {2, 2};
  test_all_each_axis<complex<float>, 4, 2>(in_array7, expect_result7, legate::complex64(), shape7);
  test_all_each_axis<complex<float>, 4, 2>(
    in_array7, expect_result7, legate::complex64(), shape7, true);

  std::array<complex<double>, 1> in_array8              = {complex<double>(0, 1)};
  std::map<int32_t, std::array<bool, 1>> expect_result8 = {{0, {true}}};
  std::vector<uint64_t> shape8                          = {1};
  test_all_each_axis<complex<double>, 1, 1>(
    in_array8, expect_result8, legate::complex128(), shape8);
  test_all_each_axis<complex<double>, 1, 1>(
    in_array8, expect_result8, legate::complex128(), shape8, true);
}

void test_all_axis_input()
{
  std::array<int32_t, 4> in_array = {5, 10, 0, 100};
  std::vector<uint64_t> shape     = {1, 2, 2};

  std::vector<int32_t> axis1      = {0};
  std::array<bool, 4> expect_val1 = {true, true, false, true};
  test_all<int32_t, bool, 4, 4, 3, 2>(in_array, expect_val1, legate::int32(), shape, axis1);

  std::vector<int32_t> axis2      = {1, 2};
  std::array<bool, 1> expect_val2 = {false};
  test_all<int32_t, bool, 4, 1, 3, 1>(in_array, expect_val2, legate::int32(), shape, axis2);

  std::vector<int32_t> axis3      = {-1, 0, 1};
  std::array<bool, 1> expect_val3 = {false};
  test_all<int32_t, bool, 4, 1, 3, 1>(in_array, expect_val3, legate::int32(), shape, axis3);
}

void test_all_where_input()
{
  std::array<bool, 4> in_array = {true, false, true, true};
  std::vector<uint64_t> shape  = {2, 2};

  // Test where with multiple bool values
  std::array<bool, 2> where_in1 = {true, false};
  auto where_array1             = cupynumeric::zeros({2}, legate::bool_());
  assign_values_to_array<bool, 1>(where_array1, where_in1.data(), where_in1.size());

  std::array<bool, 1> expect_val1 = {true};
  test_all<bool, bool, 4, 1, 2, 1>(
    in_array, expect_val1, legate::bool_(), shape, {}, std::nullopt, false, where_array1);

  // Test where with single bool value
  std::array<bool, 1> where_in2 = {true};
  auto where_array2             = cupynumeric::zeros({1}, legate::bool_());
  assign_values_to_array<bool, 1>(where_array2, where_in2.data(), where_in2.size());

  std::array<bool, 1> expect_val2 = {false};
  test_all<bool, bool, 4, 1, 2, 1>(
    in_array, expect_val2, legate::bool_(), shape, {}, std::nullopt, false, where_array2);

  std::array<bool, 1> where_in3 = {false};
  auto where_array3             = cupynumeric::zeros({1}, legate::bool_());
  assign_values_to_array<bool, 1>(where_array3, where_in3.data(), where_in3.size());

  std::array<bool, 1> expect_val3 = {true};
  test_all<bool, bool, 4, 1, 2, 1>(
    in_array, expect_val3, legate::bool_(), shape, {}, std::nullopt, false, where_array3);
}

void test_all_out_input()
{
  std::array<int32_t, 8> in_array = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> shape     = {2, 2, 2};
  std::vector<uint64_t> out_shape = {2, 2};
  std::vector<int32_t> axis       = {0};

  auto out1                          = cupynumeric::zeros(out_shape, legate::int32());
  std::array<int32_t, 4> expect_val1 = {0, 1, 1, 1};
  test_all<int32_t, int32_t, 8, 4, 3, 2>(in_array, expect_val1, legate::int32(), shape, axis, out1);

  auto out2                         = cupynumeric::zeros(out_shape, legate::float64());
  std::array<double, 4> expect_val2 = {0.0, 1.0, 1.0, 1.0};
  test_all<int32_t, double, 8, 4, 3, 2>(in_array, expect_val2, legate::int32(), shape, axis, out2);

  auto out3                                 = cupynumeric::zeros(out_shape, legate::complex64());
  std::array<complex<float>, 4> expect_val3 = {
    complex<float>(0, 0), complex<float>(1, 0), complex<float>(1, 0), complex<float>(1, 0)};
  test_all<int32_t, complex<float>, 8, 4, 3, 2>(
    in_array, expect_val3, legate::int32(), shape, axis, out3);

  auto out4                       = cupynumeric::zeros(out_shape, legate::bool_());
  std::array<bool, 4> expect_val4 = {false, true, true, true};
  test_all<int32_t, bool, 8, 4, 3, 2>(in_array, expect_val4, legate::int32(), shape, axis, out4);
}

template <int32_t IN_SIZE, int32_t OUT_SIZE>
void test_all_max_dim(int32_t dim)
{
  std::array<int32_t, IN_SIZE> in_array;
  for (int32_t i = 0; i < IN_SIZE; i++) {
    in_array[i] = i;
  }

  int32_t count = IN_SIZE / OUT_SIZE;
  std::vector<uint64_t> shapes;
  for (int32_t i = 0; i < dim; i++) {
    shapes.push_back(count);
  }

  std::array<bool, OUT_SIZE> expect_val;
  expect_val[0] = false;
  for (int32_t i = 1; i < OUT_SIZE; i++) {
    expect_val[i] = true;
  }

  std::map<int32_t, std::array<bool, OUT_SIZE>> expect_result;
  for (int32_t i = 0; i < dim; i++) {
    expect_result[i] = expect_val;
  }

  test_all_each_axis<int32_t, IN_SIZE, OUT_SIZE>(in_array, expect_result, legate::int32(), shapes);
  test_all_each_axis<int32_t, IN_SIZE, OUT_SIZE>(
    in_array, expect_result, legate::int32(), shapes, true);
}

void test_all_max_dim()
{
#if LEGATE_MAX_DIM >= 4
  const int32_t count_4d        = 81;
  const int32_t count_expect_4d = 27;
  const int32_t dim_4d          = 4;
  test_all_max_dim<count_4d, count_expect_4d>(dim_4d);
#endif

#if LEGATE_MAX_DIM >= 5
  const int32_t count_5d        = 243;
  const int32_t count_expect_5d = 81;
  const int32_t dim_5d          = 5;
  test_all_max_dim<count_5d, count_expect_5d>(dim_5d);
#endif

#if LEGATE_MAX_DIM >= 6
  const int32_t count_6d        = 729;
  const int32_t count_expect_6d = 243;
  const int32_t dim_6d          = 6;
  test_all_max_dim<count_6d, count_expect_6d>(dim_6d);
#endif

#if LEGATE_MAX_DIM >= 7
  const int32_t count_7d        = 2187;
  const int32_t count_expect_7d = 729;
  const int32_t dim_7d          = 7;
  test_all_max_dim<count_7d, count_expect_7d>(dim_7d);
#endif
}

void test_all_empty_array()
{
  std::array<int32_t, 0> in_array = {};
  std::vector<uint64_t> shape     = {0};
  std::array<bool, 1> expect_val  = {true};

  test_all<int32_t, bool, 0, 1, 1, 1>(in_array, expect_val, legate::int32(), shape);
}

void test_all_large_array()
{
  const int32_t count            = 100000;
  std::vector<uint64_t> shape    = {count};
  std::array<bool, 1> expect_val = {true};

  // Test int type for large array
  std::array<int32_t, count> in_array1;
  for (int32_t i = 0; i < count; i++) {
    in_array1[i] = i + 1;
  }
  test_all<int32_t, bool, count, 1, 1, 1>(in_array1, expect_val, legate::int32(), shape);

  // Test float type
  std::array<double, count> in_array2;
  for (int32_t i = 0; i < count; i++) {
    in_array2[i] = i + 1.1;
  }
  test_all<double, bool, count, 1, 1, 1>(in_array2, expect_val, legate::float64(), shape);

  // Test complex type
  std::array<complex<float>, count> in_array3;
  for (int32_t i = 0; i < count; i++) {
    in_array3[i] = complex<float>(i + 1, i + 1);
  }
  test_all<complex<float>, bool, count, 1, 1, 1>(in_array3, expect_val, legate::complex64(), shape);
}

void test_all_invalid_axis()
{
  std::array<int32_t, 4> in_array = {5, 10, 0, 100};
  std::vector<uint64_t> shape     = {1, 2, 2};
  auto array                      = cupynumeric::zeros(shape, legate::int32());
  assign_values_to_array<int32_t, 3>(array, in_array.data(), in_array.size());

  // Test out-of-bound
  std::vector<int32_t> axis1 = {-4, 3};
  EXPECT_THROW(cupynumeric::all(array, axis1), std::invalid_argument);

  std::vector<int32_t> axis2 = {0, 3};
  EXPECT_THROW(cupynumeric::all(array, axis2), std::invalid_argument);

  // Test repeated axes
  std::vector<int32_t> axis3 = {1, 1};
  EXPECT_THROW(cupynumeric::all(array, axis3), std::invalid_argument);

  std::vector<int32_t> axis4 = {-1, 2};
  EXPECT_THROW(cupynumeric::all(array, axis4), std::invalid_argument);
}

void test_all_invalid_shape()
{
  std::array<int32_t, 4> in_array = {5, 10, 0, 100};
  std::vector<uint64_t> shape     = {1, 2, 2};
  auto array                      = cupynumeric::zeros(shape, legate::int32());
  assign_values_to_array<int32_t, 3>(array, in_array.data(), in_array.size());

  std::vector<uint64_t> out_shape1 = {1};
  auto out1                        = cupynumeric::zeros(out_shape1, legate::int32());
  EXPECT_THROW(cupynumeric::all(array, {}, out1), std::invalid_argument);

  std::vector<uint64_t> out_shape2 = {2};
  std::vector<int32_t> axis2       = {1};
  auto out2                        = cupynumeric::zeros(out_shape2, legate::int32());
  EXPECT_THROW(cupynumeric::all(array, axis2, out2), std::invalid_argument);

  std::vector<uint64_t> out_shape3 = {2, 2};
  std::vector<int32_t> axis3       = {1};
  auto out3                        = cupynumeric::zeros(out_shape3, legate::int32());
  EXPECT_THROW(cupynumeric::all(array, axis3, out3), std::invalid_argument);
}

void test_all_invalid_where()
{
  std::array<int32_t, 4> in_array = {5, 10, 0, 100};
  std::vector<uint64_t> shape     = {1, 2, 2};
  auto array                      = cupynumeric::zeros(shape, legate::int32());
  assign_values_to_array<int32_t, 3>(array, in_array.data(), in_array.size());

  // Test where with invalid type
  std::array<int32_t, 4> in_where1 = {0, 1, 0, 1};
  auto where1                      = cupynumeric::zeros(shape, legate::int32());
  assign_values_to_array<int32_t, 3>(where1, in_where1.data(), in_where1.size());
  EXPECT_THROW(cupynumeric::all(array, {}, std::nullopt, false, where1), std::invalid_argument);

  // Test where with invalid shape
  std::vector<uint64_t> where_shape = {2, 2, 1};
  std::array<bool, 4> in_where2     = {false, true, false, true};
  auto where2                       = cupynumeric::zeros(where_shape, legate::bool_());
  assign_values_to_array<bool, 3>(where2, in_where2.data(), in_where2.size());
  EXPECT_THROW(cupynumeric::all(array, {}, std::nullopt, false, where2), std::exception);
}

// void cpp_test()
TEST(Logical, AllBasicTest) { test_all_basic(); }
TEST(Logical, AllAxisInput) { test_all_axis_input(); }
TEST(Logical, AllOutInput) { test_all_out_input(); }
// TODO - after where is supported
// TEST(Logical, AllWhereInput) {  test_all_where_input(); }
TEST(Logical, AllEmptyArray) { test_all_empty_array(); }
TEST(Logical, AllLargeArray) { test_all_large_array(); }
TEST(Logical, AllMaxDim) { test_all_max_dim(); }
TEST(Logical, AllInvalidAxis) { test_all_invalid_axis(); }
TEST(Logical, AllInvalidShape) { test_all_invalid_shape(); }
TEST(Logical, AllInvalidWhere) { test_all_invalid_where(); }
