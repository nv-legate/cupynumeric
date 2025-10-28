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

template <typename T>
void test_where_basic(std::vector<T> in_a,
                      std::vector<std::vector<int64_t>>& exp_vec,
                      std::vector<uint64_t> in_shape)
{
  auto A = mk_array<T>(in_a, in_shape);
  auto B = where(A);
  assert(exp_vec.size() == B.size());
  for (size_t i = 0; i < B.size(); i++) {
    auto exp_arr                    = exp_vec[i];
    std::vector<uint64_t> exp_shape = {exp_arr.size()};
    check_array<int64_t>(B[i], exp_arr, exp_shape);
  }
}

template <typename T>
void test_where_full(
  NDArray A, NDArray X, NDArray Y, std::vector<T> exp_arr, std::vector<uint64_t> exp_shape)
{
  auto B = where(A, X, Y);
  check_array<T>(B, exp_arr, exp_shape);
}

TEST(Where, Basic)
{
  std::vector<int32_t> in_a = {-1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0, 1};
  std::vector<std::vector<uint64_t>> test_shapes = {{18}, {6, 3}, {3, 2, 3}};

  std::vector<int64_t> exp_vec1_1            = {0, 1, 2, 3, 5, 6, 7, 9, 11, 15, 17};
  std::vector<std::vector<int64_t>> exp_vec1 = {exp_vec1_1};
  test_where_basic<int32_t>(in_a, exp_vec1, test_shapes[0]);

  std::vector<int64_t> exp_vec2_1            = {0, 0, 0, 1, 1, 2, 2, 3, 3, 5, 5};
  std::vector<int64_t> exp_vec2_2            = {0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 2};
  std::vector<std::vector<int64_t>> exp_vec2 = {exp_vec2_1, exp_vec2_2};
  test_where_basic<int32_t>(in_a, exp_vec2, test_shapes[1]);

  std::vector<int64_t> exp_vec3_1            = {0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
  std::vector<int64_t> exp_vec3_2            = {0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> exp_vec3_3            = {0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 2};
  std::vector<std::vector<int64_t>> exp_vec3 = {exp_vec3_1, exp_vec3_2, exp_vec3_3};
  test_where_basic<int32_t>(in_a, exp_vec3, test_shapes[2]);
}

TEST(Where, Condition)
{
  std::vector<uint64_t> shape = {2, 2};
  auto X                      = mk_array<int32_t>({1, 2, 3, 4}, shape);
  auto Y                      = mk_array<int32_t>({9, 8, 7, 6}, shape);

  auto A1 = mk_array<bool>({true, false, true, true}, shape);
  test_where_full<int32_t>(A1, X, Y, {1, 8, 3, 4}, shape);

  auto A2 = mk_array<bool>({true, false}, {1, 2});
  test_where_full<int32_t>(A2, X, Y, {1, 8, 3, 6}, shape);

  auto A3 = mk_array<bool>({true, false},
                           {
                             2,
                           });
  test_where_full<int32_t>(A3, X, Y, {1, 8, 3, 6}, shape);

  auto A4 = mk_array<float>({0.0, 1.0, 0, -2}, shape);
  test_where_full<int32_t>(A4, X, Y, {9, 2, 7, 4}, shape);
}

TEST(Where, Type)
{
  std::vector<uint64_t> shape = {2, 2};
  auto A                      = mk_array<bool>({true, false, true, true}, shape);
  auto X_BOOL                 = mk_array<bool>({true, false, true, false}, shape);
  auto X_INT                  = mk_array<int32_t>({1, 2, 3, 4}, shape);
  auto X_FLOAT                = mk_array<float>({1, 2, 3, 4}, shape);
  auto X_COMPLEX128           = mk_array<legate::Complex<double>>({1, 2, 3, 4}, shape);
  auto Y_BOOL                 = mk_array<bool>({false, true, true, false}, shape);
  auto Y_INT                  = mk_array<int32_t>({9, 8, 7, 6}, shape);
  auto Y_FLOAT                = mk_array<float>({9, 8, 7, 6}, shape);
  auto Y_COMPLEX128           = mk_array<legate::Complex<double>>({9, 8, 7, 6}, shape);

  test_where_full<bool>(A, X_BOOL, Y_BOOL, {true, true, true, false}, shape);

  test_where_full<int32_t>(A, X_BOOL, Y_INT, {1, 8, 1, 0}, shape);
  test_where_full<int32_t>(A, X_INT, Y_INT, {1, 8, 3, 4}, shape);
  test_where_full<int32_t>(A, Y_INT, X_BOOL, {9, 0, 7, 6}, shape);

  test_where_full<float>(A, X_BOOL, Y_FLOAT, {1, 8, 1, 0}, shape);
  test_where_full<float>(A, X_INT, Y_FLOAT, {1, 8, 3, 4}, shape);
  test_where_full<float>(A, X_FLOAT, Y_FLOAT, {1, 8, 3, 4}, shape);
  test_where_full<float>(A, Y_FLOAT, X_BOOL, {9, 0, 7, 6}, shape);
  test_where_full<float>(A, Y_FLOAT, X_INT, {9, 2, 7, 6}, shape);

  test_where_full<legate::Complex<double>>(A, X_BOOL, Y_COMPLEX128, {1, 8, 1, 0}, shape);
  test_where_full<legate::Complex<double>>(A, X_INT, Y_COMPLEX128, {1, 8, 3, 4}, shape);
  test_where_full<legate::Complex<double>>(A, X_FLOAT, Y_COMPLEX128, {1, 8, 3, 4}, shape);
  test_where_full<legate::Complex<double>>(A, X_COMPLEX128, Y_COMPLEX128, {1, 8, 3, 4}, shape);
  test_where_full<legate::Complex<double>>(A, Y_COMPLEX128, X_BOOL, {9, 0, 7, 6}, shape);
  test_where_full<legate::Complex<double>>(A, Y_COMPLEX128, X_INT, {9, 2, 7, 6}, shape);
  test_where_full<legate::Complex<double>>(A, Y_COMPLEX128, X_FLOAT, {9, 2, 7, 6}, shape);
}

TEST(Where, BroadcastShape)
{
  auto X = mk_array<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
  auto Y = mk_array<int32_t>({10, 20, 30}, {1, 3});

  auto A1 = mk_array<bool>({false}, {1});
  test_where_full<int32_t>(A1, X, Y, {10, 20, 30, 10, 20, 30, 10, 20, 30}, {3, 3});

  auto A2 = mk_array<bool>({false, true, true}, {3});
  test_where_full<int32_t>(A2, X, Y, {10, 2, 3, 10, 5, 6, 10, 8, 9}, {3, 3});

  auto A3 = mk_array<bool>({false, true, true}, {1, 3});
  test_where_full<int32_t>(A3, X, Y, {10, 2, 3, 10, 5, 6, 10, 8, 9}, {3, 3});

  auto A4 = mk_array<bool>({false, true, true, true, false, false, true, false, false}, {3, 3});
  test_where_full<int32_t>(A4, X, Y, {10, 2, 3, 4, 20, 30, 7, 20, 30}, {3, 3});

  auto A5 = mk_array<bool>({false,
                            true,
                            true,
                            true,
                            false,
                            false,
                            true,
                            false,
                            false,
                            false,
                            true,
                            true,
                            true,
                            false,
                            false,
                            true,
                            false,
                            false},
                           {2, 3, 3});
  test_where_full<int32_t>(
    A5, X, Y, {10, 2, 3, 4, 20, 30, 7, 20, 30, 10, 2, 3, 4, 20, 30, 7, 20, 30}, {2, 3, 3});
}

TEST(Where, EmptyAndScalar)
{
  auto A        = mk_array<bool>({true},
                          {
                            1,
                          });
  auto A_SCALAR = mk_array<bool>({false}, {});
  auto A_EMPTY  = mk_array<bool>({},
                                {
                                  0,
                                });
  auto X        = mk_array<int32_t>({10},
                             {
                               1,
                             });
  auto Y        = mk_array<int32_t>({20},
                             {
                               1,
                             });
  auto X_SCALAR = mk_array<int32_t>({10}, {});
  auto Y_SCALAR = mk_array<int32_t>({20}, {});
  auto EMPTY    = mk_array<int32_t>({},
                                 {
                                   0,
                                 });

  auto B1 = where(A_EMPTY, X, Y);
  check_array<int32_t>(B1,
                       {},
                       {
                         0,
                       });

  auto B2 = where(A_EMPTY, X_SCALAR, Y_SCALAR);
  check_array<int32_t>(B2,
                       {},
                       {
                         0,
                       });

  auto B3 = where(A, EMPTY, Y_SCALAR);
  check_array<int32_t>(B3,
                       {},
                       {
                         0,
                       });

  auto B4 = where(A, EMPTY, EMPTY);
  check_array<int32_t>(B4,
                       {},
                       {
                         0,
                       });

  auto B5 = where(A_EMPTY, EMPTY, EMPTY);
  check_array<int32_t>(B5,
                       {},
                       {
                         0,
                       });

  auto B6 = where(A_SCALAR, X, Y_SCALAR);
  check_array<int32_t>(B6,
                       {20},
                       {
                         1,
                       });

  auto B7 = where(A_SCALAR, X_SCALAR, Y_SCALAR);
  check_array<int32_t>(B7, {20}, {});

  auto B8 = where(A, X_SCALAR, Y_SCALAR);
  check_array<int32_t>(B8,
                       {10},
                       {
                         1,
                       });

  auto B9 = where(A, X_SCALAR, Y);
  check_array<int32_t>(B9,
                       {10},
                       {
                         1,
                       });
}

TEST(Where, InvalidShape)
{
  auto A = mk_array<bool>({false, true, true, true, false, false, true, false, false}, {3, 3});
  auto X = mk_array<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});

  auto Y1 = mk_array<int32_t>({10, 20},
                              {
                                2,
                              });
  auto Y2 = mk_array<int32_t>({10, 20}, {1, 2});
  auto Y3 = mk_array<int32_t>({10, 20, 30, 40}, {4, 1});
  auto Y4 = mk_array<int32_t>({},
                              {
                                0,
                              });

  for (auto Y : {Y1, Y2, Y3, Y4}) {
    EXPECT_THROW(where(A, X, Y), std::exception);
  }
}
