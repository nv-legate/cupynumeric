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

#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include "legate.h"
#include "cupynumeric.h"
#include "common_utils.h"

using namespace cupynumeric;

namespace {

template <typename T>
auto test_standard(uint64_t m, uint64_t n, uint64_t k, legate::Type leg_type)
{
  std::vector<T> data_a(m * k);
  std::vector<T> data_b(n * k);
  std::iota(data_a.begin(), data_a.end(), 0);
  std::iota(data_b.begin(), data_b.end(), 0.0);

  auto A = cupynumeric::zeros({m, k}, leg_type);
  auto B = cupynumeric::zeros({k, n}, leg_type);

  assign_values_to_array<T, 2>(A, data_a.data(), m * k);
  assign_values_to_array<T, 2>(B, data_b.data(), n * k);

  auto C                          = dot(A, B);
  std::vector<uint64_t> exp_shape = {m, n};
  EXPECT_EQ(C.type(), leg_type);
  EXPECT_EQ(C.shape(), exp_shape);
}

TEST(Dot, Standard)
{
  test_standard<float>(124, 95, 30, legate::float32());
  test_standard<double>(124, 95, 30, legate::float64());
}

TEST(Dot, Complex)
{
  test_standard<complex<float>>(124, 95, 30, legate::complex64());
  test_standard<complex<double>>(124, 95, 30, legate::complex128());
}

TEST(Dot, Large)
{
  // activate tiling (m,n) and/or batching (k)
  test_standard<float>(513, 12, 4, legate::float32());
  test_standard<float>(12, 518, 30, legate::float32());
  test_standard<float>(513, 513, 30, legate::float32());
  test_standard<double>(512, 512, 4097, legate::float64());
  test_standard<double>(1024, 1024, 4097, legate::float64());
}

}  // namespace
