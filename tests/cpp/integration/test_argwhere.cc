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
#include "common_utils.h"

using namespace cupynumeric;

namespace {
std::vector<std::vector<uint64_t>> get_in_shapes_basic()
{
  std::vector<std::vector<uint64_t>> in_shapes = {{12},
                                                  {4, 3},
                                                  {2, 2, 3},
                                                  {2, 1, 2, 3},
                                                  {2, 1, 2, 1, 3},
                                                  {2, 1, 2, 1, 3, 1},
                                                  {1, 2, 1, 2, 1, 3, 1}};
  return in_shapes;
}

std::vector<std::vector<uint64_t>> get_exp_shapes_basic()
{
  std::vector<std::vector<uint64_t>> exp_shapes = {
    {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7}};
  return exp_shapes;
}

std::vector<std::vector<int64_t>> get_exp_vectors_basic()
{
  std::vector<std::vector<int64_t>> exp_vectors = {
    {0, 2, 5, 6, 9, 11},
    {0, 0, 0, 2, 1, 2, 2, 0, 3, 0, 3, 2},
    {0, 0, 0, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 1, 0, 1, 1, 2},
    {0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0,
     1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 2, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0}};
  return exp_vectors;
}

template <typename T>
void test_argwhere(std::vector<T>& in_vec,
                   std::vector<int64_t>& exp_vec,
                   const std::vector<uint64_t>& in_shape,
                   const std::vector<uint64_t>& exp_shape)
{
  auto a = mk_array(in_vec, in_shape);
  auto x = argwhere(a);
  check_array(x, exp_vec, exp_shape);
}

template <typename T>
void test_argwhere_basic(std::vector<T>& in_vec, uint32_t dim)
{
  auto in_shapes   = get_in_shapes_basic();
  auto exp_shapes  = get_exp_shapes_basic();
  auto exp_vectors = get_exp_vectors_basic();

  test_argwhere<T>(in_vec, exp_vectors[dim - 1], in_shapes[dim - 1], exp_shapes[dim - 1]);
}

template <typename T>
void test_argwhere_basic_for_all_dims(std::vector<T>& in_vec)
{
  test_argwhere_basic<T>(in_vec, 1);
  test_argwhere_basic<T>(in_vec, 2);
  test_argwhere_basic<T>(in_vec, 3);

#if LEGATE_MAX_DIM >= 4
  test_argwhere_basic<T>(in_vec, 4);
#endif

#if LEGATE_MAX_DIM >= 5
  test_argwhere_basic<T>(in_vec, 5);
#endif

#if LEGATE_MAX_DIM >= 6
  test_argwhere_basic<T>(in_vec, 6);
#endif

#if LEGATE_MAX_DIM >= 7
  test_argwhere_basic<T>(in_vec, 7);
#endif
}

void argwhere_int()
{
  std::vector<int32_t> in_vec = {-1, 0, 4, +0, -0, 45, 5, 0, 0, 9, 0, 4};
  test_argwhere_basic_for_all_dims<int32_t>(in_vec);
}

void argwhere_double()
{
  std::vector<double> in_vec = {0.01, 0, 4.0, -0.00, 0.00, 0.1, -5, +0.0, 0, 9, 0.0, 4};
  test_argwhere_basic_for_all_dims<double>(in_vec);
}

void argwhere_complex()
{
  std::vector<legate::Complex<float>> in_vec = {legate::Complex<float>(1.0, 0),
                                                legate::Complex<float>(0.0, 0.0),
                                                54,
                                                0,
                                                0.0,
                                                legate::Complex<float>(0, 1.0),
                                                45,
                                                0,
                                                0.0,
                                                9,
                                                -0.00,
                                                4};
  test_argwhere_basic_for_all_dims<legate::Complex<float>>(in_vec);
}

void argwhere_bool()
{
  std::vector<bool> in_vec = {
    true, false, true, false, false, true, true, false, false, true, false, true};
  test_argwhere_basic_for_all_dims<bool>(in_vec);
}

void test_argwhere_empty_array(legate::Type leg_type,
                               std::vector<uint64_t> in_shape,
                               std::vector<uint64_t> exp_shape)
{
  auto a = zeros(in_shape, leg_type);
  auto x = argwhere(a);
  EXPECT_EQ(x.size(), 0);
  EXPECT_EQ(x.type(), legate::int64());
  EXPECT_EQ(x.shape(), exp_shape);
}

template <typename T>
std::vector<T> init_large_vector(size_t size)
{
  std::vector<T> vec = {};
  for (uint i = 0; i < size; i++) {
    T element = (i % 2 == 0) ? 1 : 0;
    vec.push_back(element);
  }
  return vec;
}

template <typename T>
std::vector<int64_t> argwhere_result(const std::vector<T>& in_vec,
                                     const std::vector<uint64_t>& in_shape)
{
  std::vector<int64_t> a(in_shape.size(), 0);
  std::vector<int64_t> result;
  for (uint32_t i = 0; i < in_vec.size(); i++) {
    if (in_vec[i] != 0) {
      for (auto aa : a) {
        result.push_back(aa);
      }
    }
    int32_t j = a.size() - 1;
    while (j >= 0) {
      if (++a[j] >= in_shape[j]) {
        a[j] = 0;
        j--;
      } else {
        break;
      }
    }
  }
  return result;
}

std::vector<uint64_t> gen_shape(uint32_t dim, size_t in_size)
{
  std::vector<uint64_t> shape(dim, 1);
  size_t value = 2;
  size_t prod  = 1;
  for (int i = 0; i < dim - 1; i++) {
    shape[i] = value;
    prod *= value;
    value++;
  }
  shape[dim - 1] = in_size / prod;
  return shape;
}

void argwhere_large_array(uint32_t dim)
{
  size_t in_size = 2 * 3 * 4 * 5 * 6 * 7;
  auto in_vec    = init_large_vector<int32_t>(in_size);
  // for dim = 1, in_shape is {5040}
  // for dim = 2, in_shape is {2, 2520}
  // for dim = 3, in_shape is {2, 3, 840}
  // for dim = 7, in_shape is {2, 3, 4, 5, 6, 7}
  auto in_shape = gen_shape(dim, in_size);

  auto a                          = mk_array(in_vec, in_shape);
  auto x                          = argwhere(a);
  auto x_comp                     = argwhere_result<int32_t>(in_vec, in_shape);
  std::vector<uint64_t> exp_shape = {x_comp.size() / in_shape.size(), dim};
  check_array(x, x_comp, exp_shape);
}

TEST(Argwhere, Basic)
{
  argwhere_int();
  argwhere_double();
  argwhere_complex();
  argwhere_bool();
}

TEST(Argwhere, LargeArray)
{
  argwhere_large_array(1);
  argwhere_large_array(2);
  argwhere_large_array(3);

#if LEGATE_MAX_DIM >= 4
  argwhere_large_array(4);
#endif

#if LEGATE_MAX_DIM >= 5
  argwhere_large_array(5);
#endif

#if LEGATE_MAX_DIM >= 6
  argwhere_large_array(6);
#endif

#if LEGATE_MAX_DIM >= 7
  argwhere_large_array(7);
#endif
}

TEST(Argwhere, EmptyArray)
{
  std::vector<std::vector<uint64_t>> in_shapes = {{
                                                    0,
                                                  },
                                                  {0, 1},
                                                  {1, 0},
                                                  {1, 0, 0},
                                                  {1, 1, 0},
                                                  {1, 0, 1}};

  std::vector<std::vector<uint64_t>> exp_shapes = {
    // {0, 1}, {0, 2}, {0, 2}, {0, 3}, {0, 3}, {0, 3}}；//This is shape of numpy output array.
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0}  // This is shape of cupynumeric output array
  };

  assert(in_shapes.size() == exp_shapes.size());
  for (size_t i = 0; i < in_shapes.size(); i++) {
    test_argwhere_empty_array(legate::int32(), in_shapes[i], exp_shapes[i]);
  }
}

TEST(Argwhere, Scalar)
{
  std::vector<uint64_t> exp_shape1 = {0, 0};
  auto A1                          = zeros({}, legate::int32());
  auto B1                          = argwhere(A1);
  EXPECT_EQ(B1.size(), 0);
  EXPECT_EQ(B1.type(), legate::int64());
  EXPECT_EQ(B1.shape(), exp_shape1);

  std::vector<uint64_t> exp_shape2 = {1, 0};
  auto A2                          = zeros({}, legate::float64());
  A2.fill(legate::Scalar(static_cast<double>(1)));
  auto B2 = cupynumeric::argwhere(A2);
  EXPECT_EQ(B2.size(), 0);
  EXPECT_EQ(B2.type(), legate::int64());
  EXPECT_EQ(B2.shape(), exp_shape2);
}

}  // namespace
