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
#include "common_utils.h"

auto get_nonzero_expect_result()
{
  std::vector<std::vector<std::vector<int64_t>>> expect_result = {
    {{1, 2, 4, 5, 6, 8, 10}},
    {{0, 0, 0, 0, 0, 0, 0}, {1, 2, 4, 5, 6, 8, 10}},
    {{1, 2, 4, 5, 6, 8, 10}, {0, 0, 0, 0, 0, 0, 0}},
    {{0, 0, 1, 1, 1, 2, 2}, {1, 2, 0, 1, 2, 0, 2}},
    {{1, 2, 4, 5, 6, 8, 10}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}},
    {{0, 0, 0, 0, 0, 0, 0}, {1, 2, 4, 5, 6, 8, 10}, {0, 0, 0, 0, 0, 0, 0}},
    {{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {1, 2, 4, 5, 6, 8, 10}},
    {{0, 0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 2, 0, 2, 1}}};
  return expect_result;
}

auto get_nonzero_expect_result_4d()
{
  std::vector<std::vector<std::vector<int64_t>>> expect_result = {{{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 5, 7, 9, 11, 14}},
                                                                  {{0, 2, 3, 5, 7, 9, 11, 14},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 1, 1, 1},
                                                                   {0, 0, 0, 1, 1, 0, 0, 1},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 1, 3, 1, 3, 2}}};
  return expect_result;
}

auto get_nonzero_expect_result_5d()
{
  std::vector<std::vector<std::vector<int64_t>>> expect_result = {{{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 5, 7, 9, 11, 14},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 5, 7, 9, 11, 14},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 1, 1, 1},
                                                                   {0, 0, 0, 1, 1, 0, 0, 1},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 1, 3, 1, 3, 2}}};
  return expect_result;
}

auto get_nonzero_expect_result_6d()
{
  std::vector<std::vector<std::vector<int64_t>>> expect_result = {{{0, 2, 3, 5, 7, 9, 11, 14},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 5, 7, 9, 11, 14},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 1, 1, 1},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 1, 1, 0, 0, 1},
                                                                   {0, 1, 1, 0, 1, 0, 1, 1},
                                                                   {0, 0, 1, 1, 1, 1, 1, 0}}};
  return expect_result;
}

auto get_nonzero_expect_result_7d()
{
  std::vector<std::vector<std::vector<int64_t>>> expect_result = {{{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 5, 7, 9, 11, 14},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 1, 1, 1},
                                                                   {0, 0, 0, 1, 1, 0, 0, 1},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 2, 3, 1, 3, 1, 3, 2},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0}},
                                                                  {{0, 0, 0, 0, 0, 1, 1, 1},
                                                                   {0, 0, 0, 1, 1, 0, 0, 1},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 1, 1, 0, 1, 0, 1, 1},
                                                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                                                   {0, 0, 1, 1, 1, 1, 1, 0}}};
  return expect_result;
}

template <typename T>
void test_nonzero(const std::vector<T>& in_array,
                  const std::vector<std::vector<int64_t>>& expect,
                  const std::vector<uint64_t>& shape)
{
  auto array         = cupynumeric::mk_array<T>(in_array, shape);
  auto result_vec    = cupynumeric::nonzero(array);
  size_t result_size = result_vec.size();
  ASSERT_EQ(result_size, expect.size());
  std::vector<uint64_t> expect_shape = {};
  if (shape.size() > 0) {
    if (result_vec[0].size() == 0) {
      expect_shape.push_back(0);
    } else if (result_vec[0].size() == 1) {
      expect_shape.push_back(1);
    }
  }
  for (size_t i = 0; i < result_size; ++i) {
    cupynumeric::check_array<int64_t>(result_vec[i], expect[i], expect_shape);
  }
}

template <typename T>
void nonzero_basic_impl(const std::vector<std::vector<uint64_t>>& test_shapes,
                        const std::vector<T>& in_array,
                        const std::vector<std::vector<std::vector<int64_t>>>& expect_result)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    test_nonzero<T>(in_array, expect_result[i], test_shapes[i]);
  }
}

void nonzero_basic()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};
  auto expect_result = get_nonzero_expect_result();

  // Test int type
  std::vector<int32_t> in_array1 = {0, 3, 12, 0, 2, 4, 8, 0, 7, 0, 11, 0};
  nonzero_basic_impl<int32_t>(test_shapes, in_array1, expect_result);

  // Test float type
  std::vector<double> in_array2 = {0.0, 3.5, 11.0, 0, 2.2, 6.5, 8, 0.0, 7.9, 0.0, 0.0011, 0};
  nonzero_basic_impl<double>(test_shapes, in_array2, expect_result);

  // Test complex type
  std::vector<legate::Complex<float>> in_array3 = {legate::Complex<float>(0, 0),
                                                   legate::Complex<float>(2.2, 0),
                                                   legate::Complex<float>(12, 5),
                                                   legate::Complex<float>(0),
                                                   legate::Complex<float>(2, 4),
                                                   legate::Complex<float>(6, 4),
                                                   legate::Complex<float>(8, 9),
                                                   legate::Complex<float>(0, 0),
                                                   legate::Complex<float>(7.9, 12),
                                                   legate::Complex<float>(0),
                                                   legate::Complex<float>(0, 0.001),
                                                   legate::Complex<float>(0, 0)};
  nonzero_basic_impl<legate::Complex<float>>(test_shapes, in_array3, expect_result);
}

void nonzero_basic_max_dim()
{
  // Only test int type for max dim
  std::vector<int32_t> in_array = {14, 0, 3, 12, 0, 13, 0, 4, 0, 8, 0, 7, 0, 0, 1, 0};

#if LEGATE_MAX_DIM >= 4
  std::vector<std::vector<uint64_t>> test_shapes_4d = {{1, 1, 1, 16}, {16, 1, 1, 1}, {2, 2, 1, 4}};
  auto expect_result_4d                             = get_nonzero_expect_result_4d();
  nonzero_basic_impl<int32_t>(test_shapes_4d, in_array, expect_result_4d);
#endif

#if LEGATE_MAX_DIM >= 5
  std::vector<std::vector<uint64_t>> test_shapes_5d = {
    {1, 1, 1, 16, 1}, {1, 16, 1, 1, 1}, {1, 2, 2, 1, 4}};
  auto expect_result_5d = get_nonzero_expect_result_5d();
  nonzero_basic_impl<int32_t>(test_shapes_5d, in_array, expect_result_5d);
#endif

#if LEGATE_MAX_DIM >= 6
  std::vector<std::vector<uint64_t>> test_shapes_6d = {
    {16, 1, 1, 1, 1, 1}, {1, 1, 16, 1, 1, 1}, {1, 2, 1, 2, 2, 2}};
  auto expect_result_6d = get_nonzero_expect_result_6d();
  nonzero_basic_impl<int32_t>(test_shapes_6d, in_array, expect_result_6d);
#endif

#if LEGATE_MAX_DIM >= 7
  std::vector<std::vector<uint64_t>> test_shapes_7d = {
    {1, 16, 1, 1, 1, 1, 1}, {1, 1, 2, 2, 1, 4, 1}, {2, 2, 1, 1, 2, 1, 2}};
  auto expect_result_7d = get_nonzero_expect_result_7d();
  nonzero_basic_impl<int32_t>(test_shapes_7d, in_array, expect_result_7d);
#endif
}

void nonzero_large_array()
{
  const int32_t count                             = 10000;
  std::vector<uint64_t> test_shape                = {count};
  std::vector<std::vector<int64_t>> expect_result = {{0, 9999}};

  // Test int type for large array
  std::vector<int32_t> in_array1(count);
  in_array1.assign(count, 0);
  in_array1[0]    = 1;
  in_array1[9999] = 1;
  test_nonzero<int32_t>(in_array1, expect_result, test_shape);

  // Test float type for large array
  std::vector<double> in_array2(count);
  in_array2.assign(count, 0.0);
  in_array2[0]    = 0.0001;
  in_array2[9999] = 0.0001;
  test_nonzero<double>(in_array2, expect_result, test_shape);

  // Test complex type for large array
  std::vector<legate::Complex<float>> in_array3(count);
  in_array3.assign(count, legate::Complex<float>(0.0));
  in_array3[0]    = legate::Complex<float>(0.0001, 0.0);
  in_array3[9999] = legate::Complex<float>(0.0, 0.0001);
  test_nonzero<legate::Complex<float>>(in_array3, expect_result, test_shape);
}

void nonzero_empty_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {0}, {0, 1}, {1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}};

  std::vector<int32_t> in_array                                = {};
  std::vector<std::vector<std::vector<int64_t>>> expect_result = {
    {{}}, {{}, {}}, {{}, {}}, {{}, {}, {}}, {{}, {}, {}}, {{}, {}, {}}};

  nonzero_basic_impl<int32_t>(test_shapes, in_array, expect_result);
}

void single_item_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {{1}, {1, 1}, {1, 1, 1}};

  std::vector<int32_t> in_array1                                = {1};
  std::vector<std::vector<std::vector<int64_t>>> expect_result1 = {
    {{0}}, {{0}, {0}}, {{0}, {0}, {0}}};
  nonzero_basic_impl<int32_t>(test_shapes, in_array1, expect_result1);

  std::vector<int32_t> in_array2                                = {0};
  std::vector<std::vector<std::vector<int64_t>>> expect_result2 = {{{}}, {{}, {}}, {{}, {}, {}}};
  nonzero_basic_impl<int32_t>(test_shapes, in_array2, expect_result2);
}

// void cpp_test()
TEST(Nonzero, Basic) { nonzero_basic(); }
TEST(Nonzero, BasicMaxDim) { nonzero_basic_max_dim(); }
TEST(Nonzero, LargeArray) { nonzero_large_array(); }
TEST(Nonzero, EmptyArray) { nonzero_empty_array(); }
TEST(Nonzero, SingleItemArray) { single_item_array(); }
