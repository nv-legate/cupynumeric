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

#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include "legate.h"
#include "cupynumeric.h"
#include "common_utils.h"

std::vector<uint64_t> calc_c_shape_scalar(const std::vector<uint64_t>& a_shape,
                                          const std::vector<uint64_t>& b_shape)
{
  assert(a_shape.size() == 0 || b_shape.size() == 0);

  std::vector<uint64_t> c_shape;
  if (a_shape.size() == 0 && b_shape.size() == 0) {
    c_shape = {};
  } else {
    c_shape = a_shape.size() == 0 ? b_shape : a_shape;
  }
  return c_shape;
}

template <typename T>
std::vector<T> calc_result_scalar(const std::vector<T>& vec_a,
                                  const std::vector<uint64_t>& a_shape,
                                  const std::vector<T>& vec_b,
                                  const std::vector<uint64_t>& b_shape,
                                  const std::vector<uint64_t>& c_shape)
{
  if (a_shape.size() == 0) {
    assert(vec_a.size() == 1);
  }
  if (b_shape.size() == 0) {
    assert(vec_b.size() == 1);
  }

  std::vector<T> vec_c;
  if (a_shape.size() == 0 && b_shape.size() == 0) {
    vec_c.push_back(vec_a[0] * vec_b[0]);
    return vec_c;
  }

  auto vec_scalar     = a_shape.size() == 0 ? vec_a : vec_b;
  auto vec_non_scalar = a_shape.size() == 0 ? vec_b : vec_a;
  for (int i = 0; i < vec_non_scalar.size(); i++) {
    vec_c.push_back(vec_non_scalar[i] * vec_scalar[0]);
  }
  return vec_c;
}

std::vector<uint64_t> calc_c_shape_b_is_vector(const std::vector<uint64_t>& a_shape,
                                               const std::vector<uint64_t>& b_shape)
{
  assert(a_shape[a_shape.size() - 1] == b_shape[b_shape.size() - 1]);
  std::vector<uint64_t> c_shape;
  int a_size_in_c = 1;
  for (int i = 0; i < a_shape.size() - 1; i++) {
    c_shape.push_back(a_shape[i]);
    a_size_in_c *= a_shape[i];
  }
  return c_shape;
}

template <typename T>
std::vector<T> calc_result_b_is_vector(const std::vector<T>& vec_a,
                                       const std::vector<uint64_t>& a_shape,
                                       const std::vector<T>& vec_b,
                                       const std::vector<uint64_t>& b_shape,
                                       const std::vector<uint64_t>& c_shape)
{
  int a_size_in_c = 1;
  for (int i = 0; i < a_shape.size() - 1; i++) {
    a_size_in_c *= a_shape[i];
  }

  std::vector<T> vec_c;
  auto x       = a_shape[a_shape.size() - 1];
  int offset_a = 0;
  for (int i_a = 0; i_a < a_size_in_c; i_a++) {
    T sum = 0;
    for (int j = 0; j < x; j++) {
      sum += vec_a[offset_a + j] * vec_b[j];
    }
    vec_c.push_back(sum);
    offset_a += x;
  }
  return vec_c;
}

std::vector<uint64_t> calc_c_shape_contract(const std::vector<uint64_t>& a_shape,
                                            const std::vector<uint64_t>& b_shape)
{
  std::vector<uint64_t> c_shape = {};
  for (int i = 0; i < a_shape.size() - 1; i++) {
    c_shape.push_back(a_shape[i]);
  }
  for (int i = 0; i < b_shape.size() - 2; i++) {
    c_shape.push_back(b_shape[i]);
  }
  c_shape.push_back(b_shape[b_shape.size() - 1]);
  return c_shape;
}

template <typename T>
std::vector<T> calc_result_contract(const std::vector<T>& vec_a,
                                    const std::vector<uint64_t>& a_shape,
                                    const std::vector<T>& vec_b,
                                    const std::vector<uint64_t>& b_shape,
                                    const std::vector<uint64_t>& c_shape)
{
  int a_size_in_c = 1, b_size_in_c = 1;
  for (int i = 0; i < a_shape.size() - 1; i++) {
    a_size_in_c *= a_shape[i];
  }
  for (int i = 0; i < b_shape.size() - 2; i++) {
    b_size_in_c *= b_shape[i];
  }
  b_size_in_c *= b_shape[b_shape.size() - 1];

  std::vector<T> vec_c;
  assert(a_shape[a_shape.size() - 1] == b_shape[b_shape.size() - 2]);

  auto x       = a_shape[a_shape.size() - 1];
  auto m       = b_shape[b_shape.size() - 1];
  int offset_a = 0;
  for (int i_a = 0; i_a < a_size_in_c; i_a++) {
    int offset_b = 0, b_i = 0;
    for (int i_b = 0; i_b < b_size_in_c; i_b++) {
      T sum = 0;
      for (int j = 0; j < x; j++) {
        sum += vec_a[offset_a + j] * vec_b[offset_b + j * m];
      }
      vec_c.push_back(sum);
      if (++b_i >= m) {
        offset_b = offset_b + m * x - m + 1;
        b_i      = 0;
      } else {
        offset_b += 1;
      }
    }
    offset_a += x;
  }
  return vec_c;
}

template <typename T>
void verify_dot_output(cupynumeric::NDArray A, cupynumeric::NDArray B, cupynumeric::NDArray C)
{
  auto vec_a = cupynumeric::to_vector<T>(A);
  auto vec_b = cupynumeric::to_vector<T>(B);
  std::vector<T> vec_c;
  auto a_shape                      = A.shape();
  auto b_shape                      = B.shape();
  std::vector<uint64_t> vec_c_shape = {};

  if (A.dim() == 0 || B.dim() == 0) {
    vec_c_shape = calc_c_shape_scalar(a_shape, b_shape);
    vec_c       = calc_result_scalar<T>(vec_a, a_shape, vec_b, b_shape, vec_c_shape);
  } else if (B.dim() == 1 && A.dim() >= 1) {
    vec_c_shape = calc_c_shape_b_is_vector(a_shape, b_shape);
    vec_c       = calc_result_b_is_vector<T>(vec_a, a_shape, vec_b, b_shape, vec_c_shape);
  } else {
    vec_c_shape = calc_c_shape_contract(a_shape, b_shape);
    vec_c       = calc_result_contract<T>(vec_a, a_shape, vec_b, b_shape, vec_c_shape);
  }

  auto leg_type = legate::primitive_type(legate::type_code_of_v<T>);
  if (leg_type == legate::float32() || leg_type == legate::float64()) {
    double abs_error = 1.e-4;
    cupynumeric::check_array_near<T>(C, vec_c, vec_c_shape, abs_error);
  }
}

template <typename T>
void test_contract_full(std::vector<uint64_t> a_shape, std::vector<uint64_t> b_shape)
{
  auto leg_type = legate::primitive_type(legate::type_code_of_v<T>);
  if (leg_type == legate::float64()) {
    auto A = a_shape.size() == 0 ? cupynumeric::mk_array<T>({10}) : cupynumeric::random(a_shape);
    auto B = b_shape.size() == 0 ? cupynumeric::mk_array<T>({10}) : cupynumeric::random(b_shape);
    auto C = cupynumeric::dot(A, B);
    verify_dot_output<T>(A, B, C);
  } else {
    auto A = a_shape.size() == 0 ? cupynumeric::mk_array<T>({10})
                                 : cupynumeric::random(a_shape).as_type(leg_type);
    auto B = b_shape.size() == 0 ? cupynumeric::mk_array<T>({10})
                                 : cupynumeric::random(b_shape).as_type(leg_type);
    auto C = cupynumeric::dot(A, B);
    if (leg_type == legate::float32()) {
      verify_dot_output<T>(A, B, C);
    }
  }
}

template <typename T>
void test_contract_standard(std::vector<uint64_t> a_shape, std::vector<uint64_t> b_shape)
{
  auto A =
    a_shape.size() == 0
      ? cupynumeric::mk_array<T>({10})
      : cupynumeric::random(a_shape).as_type(legate::primitive_type(legate::type_code_of_v<T>));
  auto B =
    b_shape.size() == 0
      ? cupynumeric::mk_array<T>({10})
      : cupynumeric::random(b_shape).as_type(legate::primitive_type(legate::type_code_of_v<T>));
  auto C = cupynumeric::dot(A, B);

  auto leg_type                     = legate::primitive_type(legate::type_code_of_v<T>);
  std::vector<uint64_t> vec_c_shape = {};
  if (A.dim() == 0 || B.dim() == 0) {
    vec_c_shape = calc_c_shape_scalar(a_shape, b_shape);
  } else if (B.dim() == 1 && A.dim() >= 1) {
    vec_c_shape = calc_c_shape_b_is_vector(a_shape, b_shape);
  } else {
    vec_c_shape = calc_c_shape_contract(a_shape, b_shape);
  }
  EXPECT_EQ(C.type(), leg_type);
  EXPECT_EQ(C.shape(), vec_c_shape);
}

template <typename T>
void test_contract_full_all(void)
{
  test_contract_full<T>({}, {});  // 0x0
  test_contract_full<T>({},
                        {
                          3,
                        });        // 0x1
  test_contract_full<T>({3}, {});  // 1x0
  test_contract_full<T>(
    {
      3,
    },
    {
      3,
    });  // 1x1
  test_contract_full<T>(
    {
      3,
    },
    {3, 4});  // 1x2
  test_contract_full<T>({2, 3},
                        {
                          3,
                        });  // 2x1
  test_contract_full<T>(
    {
      3,
    },
    {2, 3, 4});  // 1x3
  test_contract_full<T>({2, 3, 4},
                        {
                          4,
                        });                  // 3x1
  test_contract_full<T>({2, 3}, {3, 4});     // 2x2
  test_contract_full<T>({2, 3}, {5, 3, 4});  // 2x3
  test_contract_full<T>({2, 3, 4}, {4, 2});  // 3x2
#if LEGATE_MAX_DIM >= 5
  test_contract_full<T>({2, 3, 4}, {5, 4, 7});  // 3x3
  test_contract_full<T>({2, 3}, {5, 2, 3, 4});  // 2x4
#endif
}

template <typename T>
void test_contract_standard_all(void)
{
  test_contract_standard<T>({}, {});  // 0x0
  test_contract_standard<T>({},
                            {
                              3,
                            });        // 0x1
  test_contract_standard<T>({3}, {});  // 1x0
  test_contract_standard<T>(
    {
      3,
    },
    {
      3,
    });  // 1x1
  test_contract_standard<T>(
    {
      3,
    },
    {3, 4});  // 1x2
  test_contract_standard<T>({2, 3},
                            {
                              3,
                            });  // 2x1
  test_contract_standard<T>(
    {
      3,
    },
    {2, 3, 4});  // 1x3
  test_contract_standard<T>({2, 3, 4},
                            {
                              4,
                            });                  // 3x1
  test_contract_standard<T>({2, 3}, {3, 4});     // 2x2
  test_contract_standard<T>({2, 3}, {5, 3, 4});  // 2x3
  test_contract_standard<T>({2, 3, 4}, {4, 2});  // 3x2
#if LEGATE_MAX_DIM >= 5
  test_contract_standard<T>({2, 3, 4}, {5, 4, 7});  // 3x3
  test_contract_standard<T>({2, 3}, {5, 2, 3, 4});  // 2x4
#endif
}

TEST(Dot, MMStandard)
{
  test_contract_full<float>({124, 30}, {30, 95});
  test_contract_full<double>({124, 30}, {30, 95});
}

TEST(Dot, MMComplex)
{
  test_contract_standard<complex<float>>({124, 30}, {30, 95});
  test_contract_standard<complex<float>>({124, 30}, {30, 95});
}

TEST(Dot, MMLarge)
{
  test_contract_full<float>({513, 4}, {4, 12});
  test_contract_full<float>({12, 30}, {30, 518});
  test_contract_full<float>({513, 30}, {30, 513});
  test_contract_full<double>({512, 4097}, {4097, 512});
  // test_contract_full<double>({1024, 4097}, {4097, 1024}); # There is not enough space because
  // Legate is reserving 67125248 of the available 268435456 bytes (minus the eager pool allocation)
  // for the following LogicalStores
}

TEST(Dot, AllFloat) { test_contract_full_all<float>(); }

TEST(Dot, AllDouble) { test_contract_full_all<double>(); }

TEST(Dot, AllComplex64) { test_contract_standard_all<complex<float>>(); }

TEST(Dot, AllComplex128) { test_contract_standard_all<complex<double>>(); }
