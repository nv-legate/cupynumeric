/* Copyright 2023 NVIDIA Corporation
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
#include "common_utils.h"

template <typename T, typename OUT_T>
void test_amin(const std::vector<T>& in_array,
               const std::vector<size_t>& shape,
               const std::vector<OUT_T>& expect_result,
               const std::vector<size_t>& expect_shape,
               std::vector<int32_t> axis                 = {},
               std::optional<legate::Type> dtype         = std::nullopt,
               std::optional<cupynumeric::NDArray> out   = std::nullopt,
               bool keepdims                             = false,
               std::optional<legate::Scalar> initial     = std::nullopt,
               std::optional<cupynumeric::NDArray> where = std::nullopt)
{
  auto array = cupynumeric::mk_array<T>(in_array, shape);

  if (!out.has_value()) {
    auto result = cupynumeric::amin(array, axis, dtype, std::nullopt, keepdims, initial, where);
    cupynumeric::check_array<OUT_T>(result, expect_result, expect_shape);
  } else {
    cupynumeric::amin(array, axis, dtype, out, keepdims, initial, where);
    cupynumeric::check_array<OUT_T>(out.value(), expect_result, expect_shape);
  }
}

template <typename T, typename OUT_T>
void test_amin_each_axis(const std::vector<T>& arr,
                         const std::vector<size_t>& shape,
                         std::map<int32_t, std::vector<OUT_T>>& expect_results,
                         std::map<int32_t, std::vector<size_t>>& expect_shapes,
                         bool keepdims                         = false,
                         std::optional<legate::Scalar> initial = std::nullopt)
{
  int32_t dim = shape.size();
  auto df     = std::nullopt;
  for (int32_t axis = -dim + 1; axis < dim; ++axis) {
    auto index     = axis < 0 ? axis + dim : axis;
    auto exp       = expect_results[index];
    auto exp_shape = expect_shapes[index];
    auto axes      = {axis};
    test_amin<T, OUT_T>(arr, shape, exp, exp_shape, axes, df, df, keepdims, initial, df);
  }
}

void test_amin_basic()
{
  typedef std::map<int32_t, std::vector<int32_t>> IntResult;
  typedef std::map<int32_t, std::vector<double>> DoubleResult;
  typedef std::map<int32_t, std::vector<size_t>> ShapeResult;

  // Test int type - dim=1
  std::vector<int32_t> arr1  = {-1, 4, 5, 2, 0};
  std::vector<size_t> shape1 = {5};
  ShapeResult exp_shape1     = {{0, {}}};
  ShapeResult exp_shape1_k   = {{0, {1}}};
  IntResult exp1             = {{0, {-1}}};
  test_amin_each_axis<int32_t, int32_t>(arr1, shape1, exp1, exp_shape1);
  test_amin_each_axis<int32_t, int32_t>(arr1, shape1, exp1, exp_shape1_k, true);

  // Test int type - dim=2
  std::vector<int32_t> arr2  = {1, 0, 0, 5, 3, 2};
  std::vector<size_t> shape2 = {3, 2};
  ShapeResult exp_shape2     = {{0, {2}}, {1, {3}}};
  ShapeResult exp_shape2_k   = {{0, {1, 2}}, {1, {3, 1}}};
  IntResult exp2             = {{0, {0, 0}}, {1, {0, 0, 2}}};
  test_amin_each_axis<int32_t, int32_t>(arr2, shape2, exp2, exp_shape2);
  test_amin_each_axis<int32_t, int32_t>(arr2, shape2, exp2, exp_shape2_k, true);

  // Test int type - dim=3
  std::vector<int32_t> arr3  = {0, 11, 2, 3, -4, 0, -6, 7};
  std::vector<size_t> shape3 = {2, 2, 2};
  ShapeResult exp_shape3     = {{0, {2, 2}}, {1, {2, 2}}, {2, {2, 2}}};
  ShapeResult exp_shape3_k   = {{0, {1, 2, 2}}, {1, {2, 1, 2}}, {2, {2, 2, 1}}};
  IntResult exp3             = {{0, {-4, 0, -6, 3}}, {1, {0, 3, -6, 0}}, {2, {0, 2, -4, -6}}};
  test_amin_each_axis<int32_t, int32_t>(arr3, shape3, exp3, exp_shape3);
  test_amin_each_axis<int32_t, int32_t>(arr3, shape3, exp3, exp_shape3_k, true);

  // Test float type - dim=3
  std::vector<double> arr4   = {0.0, -0.99, 10.0, -5.0, 2.999, 1.51, -1.0, 2.99, 3.0};
  std::vector<size_t> shape4 = {3, 1, 3};
  ShapeResult exp_shape4     = {{0, {1, 3}}, {1, {3, 3}}, {2, {3, 1}}};
  ShapeResult exp_shape4_k   = {{0, {1, 1, 3}}, {1, {3, 1, 3}}, {2, {3, 1, 1}}};
  DoubleResult exp4          = {{0, {-5.0, -0.99, 1.51}}, {1, arr4}, {2, {-0.99, -5.0, -1.0}}};
  test_amin_each_axis<double, double>(arr4, shape4, exp4, exp_shape4);
  test_amin_each_axis<double, double>(arr4, shape4, exp4, exp_shape4_k, true);
}

void test_amin_initial_input()
{
  typedef std::map<int32_t, std::vector<int32_t>> IntResult;
  typedef std::map<int32_t, std::vector<double>> DoubleResult;
  typedef std::map<int32_t, std::vector<size_t>> ShapeResult;

  std::vector<int32_t> arr1  = {0, 11, 2, 3, -4, 0, -6, 7};
  std::vector<size_t> shape1 = {2, 2, 2};
  ShapeResult exp_shape1     = {{0, {2, 2}}, {1, {2, 2}}, {2, {2, 2}}};
  ShapeResult exp_shape1_k   = {{0, {1, 2, 2}}, {1, {2, 1, 2}}, {2, {2, 2, 1}}};
  // use initial in each axis
  auto initial1  = legate::Scalar(-1);
  IntResult exp1 = {{0, {-4, -1, -6, -1}}, {1, {-1, -1, -6, -1}}, {2, {-1, -1, -4, -6}}};
  test_amin_each_axis<int32_t, int32_t>(arr1, shape1, exp1, exp_shape1, false, initial1);
  test_amin_each_axis<int32_t, int32_t>(arr1, shape1, exp1, exp_shape1_k, true, initial1);

  std::vector<double> arr2   = {0.0, -0.99, 10.0, -5.0, 2.999, 1.51, -1.0, 2.99, 3.0};
  std::vector<size_t> shape2 = {3, 3};
  ShapeResult exp_shape2     = {{0, {3}}, {1, {3}}};
  ShapeResult exp_shape2_k   = {{0, {1, 3}}, {1, {3, 1}}};
  auto initial2              = legate::Scalar(0.0);
  DoubleResult exp2          = {{0, {-5.0, -0.99, 0.0}}, {1, {-0.99, -5.0, -1.0}}};
  test_amin_each_axis<double, double>(arr2, shape2, exp2, exp_shape2, false, initial2);
  test_amin_each_axis<double, double>(arr2, shape2, exp2, exp_shape2_k, true, initial2);
}

void test_amin_dtype_input()
{
  // int to float
  std::vector<int32_t> arr1      = {-1, 4, 5, 2, 0};
  std::vector<size_t> shape1     = {5};
  std::vector<size_t> exp_shape1 = {};
  auto dtype1                    = legate::float64();
  std::vector<double> exp1       = {-1.0};
  test_amin<int32_t, double>(arr1, shape1, exp1, exp_shape1, {}, dtype1);

  // float to int
  std::vector<double> arr2       = {0.0, -0.99, 10.1, -5.6, 2.999, 1.51};
  std::vector<size_t> shape2     = {3, 2};
  std::vector<size_t> exp_shape2 = {};
  auto dtype2                    = legate::int32();
  std::vector<int32_t> exp2      = {-5};
  test_amin<double, int32_t>(arr2, shape2, exp2, exp_shape2, {}, dtype2);
}

void test_amin_axis_input()
{
  std::vector<double> arr   = {0.0, -0.99, 10.0, -5.0, 2.999, 1.51, -1.0, 2.99, 3.0};
  std::vector<size_t> shape = {3, 1, 3};

  std::vector<int32_t> axis     = {-1, 0, 1};
  std::vector<size_t> exp_shape = {};
  std::vector<double> exp       = {-5.0};
  test_amin<double, double>(arr, shape, exp, exp_shape, axis);
}

void test_amin_out_input()
{
  // Test out input with dim-1 and different datatype
  std::vector<int32_t> arr       = {-1, 4, 5, 2, 0, 3};
  std::vector<size_t> shape1     = {6};
  std::vector<size_t> exp_shape1 = {};
  auto df                        = std::nullopt;
  auto out1                      = cupynumeric::zeros(exp_shape1, legate::int32());
  auto out1_1                    = cupynumeric::zeros(exp_shape1, legate::float64());
  std::vector<int32_t> exp1      = {-1};
  std::vector<double> exp1_1     = {-1.0};
  test_amin<int32_t, int32_t>(arr, shape1, exp1, exp_shape1, {}, df, out1);
  test_amin<int32_t, double>(arr, shape1, exp1_1, exp_shape1, {}, df, out1_1);

  // Test out input with axis, keepdims and initial params
  std::vector<size_t> shape2       = {2, 3};
  std::vector<size_t> exp_shape2   = {2};
  std::vector<size_t> exp_shape2_k = {2, 1};
  auto out2                        = cupynumeric::zeros(exp_shape2, legate::int32());
  auto out2_k                      = cupynumeric::zeros(exp_shape2_k, legate::int32());
  std::vector<int32_t> axis        = {-1};
  auto ini                         = legate::Scalar(2);
  std::vector<int32_t> exp2        = {-1, 0};
  test_amin<int32_t, int32_t>(arr, shape2, exp2, exp_shape2, axis, df, out2);
  test_amin<int32_t, int32_t>(arr, shape2, exp2, exp_shape2_k, axis, df, out2_k, true);

  test_amin<int32_t, int32_t>(arr, shape2, exp2, exp_shape2, axis, df, out2, false, ini);
  test_amin<int32_t, int32_t>(arr, shape2, exp2, exp_shape2_k, axis, df, out2_k, true, ini);
}

void test_amin_max_dim()
{
  std::vector<int32_t> arr  = {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
  std::vector<int32_t> axis = {-1};
#if LEGATE_MAX_DIM >= 4
  std::vector<size_t> shape_4d     = {2, 2, 2, 2};
  std::vector<size_t> exp_shape_4d = {2, 2, 2};
  std::vector<int32_t> exp_4d      = {10, 3, 5, 2, 8, 7, 6, 1};
  test_amin<int32_t, int32_t>(arr, shape_4d, exp_4d, exp_shape_4d, axis);
#endif

#if LEGATE_MAX_DIM >= 5
  std::vector<size_t> shape_5d     = {1, 2, 2, 1, 4};
  std::vector<size_t> exp_shape_5d = {1, 2, 2, 1};
  std::vector<int32_t> exp_5d      = {3, 2, 7, 1};
  test_amin<int32_t, int32_t>(arr, shape_5d, exp_5d, exp_shape_5d, axis);
#endif

#if LEGATE_MAX_DIM >= 6
  std::vector<size_t> shape_6d     = {2, 1, 1, 2, 2, 2};
  std::vector<size_t> exp_shape_6d = {2, 1, 1, 2, 2};
  std::vector<int32_t> exp_6d      = {10, 3, 5, 2, 8, 7, 6, 1};
  test_amin<int32_t, int32_t>(arr, shape_6d, exp_6d, exp_shape_6d, axis);
#endif

#if LEGATE_MAX_DIM >= 7
  std::vector<size_t> shape_7d     = {2, 1, 1, 2, 1, 1, 4};
  std::vector<size_t> exp_shape_7d = {2, 1, 1, 2, 1, 1};
  std::vector<int32_t> exp_7d      = {3, 2, 7, 1};
  test_amin<int32_t, int32_t>(arr, shape_7d, exp_7d, exp_shape_7d, axis);
#endif
}

void test_amin_large_array()
{
  const int32_t count           = 100000;
  std::vector<size_t> shape     = {count};
  std::vector<size_t> exp_shape = {};

  // Test int type for large array
  std::vector<int32_t> arr1(count);
  for (int32_t i = 0; i < count; i++) {
    arr1[i] = i + 1;
  }
  std::vector<int32_t> exp1 = {1};
  test_amin<int32_t, int32_t>(arr1, shape, exp1, exp_shape);

  // Test float type
  std::vector<double> arr2(count);
  for (int32_t i = 0; i < count; i++) {
    arr2[i] = i + 1.1;
  }
  std::vector<double> exp2 = {1.1};
  test_amin<double, double>(arr2, shape, exp2, exp_shape);
}

void test_amin_scalar_array()
{
  std::vector<int32_t> arr  = {10};
  std::vector<size_t> shape = {};
  std::vector<int32_t> exp  = {10};
  auto out                  = cupynumeric::zeros(shape, legate::int32());
  auto df                   = std::nullopt;
  test_amin<int32_t, int32_t>(arr, shape, exp, shape);
  test_amin<int32_t, int32_t>(arr, shape, exp, shape, {}, df, out);

  // Test with initial
  auto initial              = legate::Scalar(9);
  std::vector<int32_t> exp1 = {9};
  test_amin<int32_t, int32_t>(arr, shape, exp1, shape, {}, df, df, false, initial);
}

void test_amin_invalid_array()
{
  // Test zero size array
  std::vector<int32_t> arr1  = {};
  std::vector<size_t> shape1 = {0};
  auto arr_emp               = cupynumeric::mk_array<int32_t>(arr1, shape1);
  EXPECT_THROW(cupynumeric::amin(arr_emp), std::invalid_argument);

  // Test complex array (not supported now)
  std::vector<complex<float>> arr2 = {complex<float>(0, 1), complex<float>(1, 1)};
  std::vector<size_t> shape2       = {2};
  auto arr_comp                    = cupynumeric::mk_array<complex<float>>(arr2, shape2);
  EXPECT_THROW(cupynumeric::amin(arr_comp), std::runtime_error);
}

void test_amin_invalid_axis()
{
  std::vector<int32_t> arr  = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> shape = {1, 3, 2};
  auto array                = cupynumeric::mk_array<int32_t>(arr, shape);

  // Test out-of-bound
  std::vector<int32_t> axis1 = {-4, 3};
  std::vector<int32_t> axis2 = {0, 3};
  EXPECT_THROW(cupynumeric::amin(array, axis1), std::invalid_argument);
  EXPECT_THROW(cupynumeric::amin(array, axis2), std::invalid_argument);

  // Test repeated axes
  std::vector<int32_t> axis3 = {1, 1};
  std::vector<int32_t> axis4 = {-1, 2};
  EXPECT_THROW(cupynumeric::amin(array, axis3), std::invalid_argument);
  EXPECT_THROW(cupynumeric::amin(array, axis4), std::invalid_argument);

  // Not reduce to one value (valid but not supported now)
  std::vector<int32_t> axis5 = {0, 1};
  EXPECT_THROW(cupynumeric::amin(array, axis5), std::runtime_error);
}

void test_amin_invalid_shape()
{
  std::vector<int32_t> arr  = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> shape = {1, 3, 2};
  auto array                = cupynumeric::mk_array<int32_t>(arr, shape);
  auto df                   = std::nullopt;

  std::vector<size_t> out_shape1 = {1};
  auto out1                      = cupynumeric::zeros(out_shape1, legate::int32());
  EXPECT_THROW(cupynumeric::amin(array, {}, df, out1), std::invalid_argument);

  std::vector<size_t> out_shape2 = {2};
  std::vector<int32_t> axis2     = {1};
  auto out2                      = cupynumeric::zeros(out_shape2, legate::int32());
  EXPECT_THROW(cupynumeric::amin(array, axis2, df, out2), std::invalid_argument);
}

void test_amin_invalid_dtype()
{
  std::vector<int32_t> arr  = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> shape = {1, 3, 2};
  auto array                = cupynumeric::mk_array<int32_t>(arr, shape);

  // Test invalid dtype
  auto dtype = legate::point_type(2);
  EXPECT_THROW(cupynumeric::amin(array, {}, dtype), std::invalid_argument);
}

// void cpp_test()
TEST(Amin, BasicTest) { test_amin_basic(); }
TEST(Amin, InitialInput) { test_amin_initial_input(); }
TEST(Amin, DtypeInput) { test_amin_dtype_input(); }
TEST(Amin, AxisInput) { test_amin_axis_input(); }
TEST(Amin, OutInput) { test_amin_out_input(); }
TEST(Amin, MaxDim) { test_amin_max_dim(); }
TEST(Amin, LargeArray) { test_amin_large_array(); }
TEST(Amin, ScalarArray) { test_amin_scalar_array(); }
TEST(Amin, InvalidArray) { test_amin_invalid_array(); }
TEST(Amin, InvalidAxis) { test_amin_invalid_axis(); }
TEST(Amin, InvalidShape) { test_amin_invalid_shape(); }
TEST(Amin, InvalidDtype) { test_amin_invalid_dtype(); }
