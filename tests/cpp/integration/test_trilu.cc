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
#include <tuple>

using namespace cupynumeric;

namespace {

template <typename T>
std::tuple<std::vector<T>, std::vector<uint64_t>> trilu_result(std::vector<T> a,
                                                               std::vector<uint64_t> shape,
                                                               int32_t k  = 0,
                                                               bool lower = true)
{
  if (shape.empty()) {
    throw std::invalid_argument("Array must be at least 1-D");
  }

  size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
  if (a.size() != size) {
    throw std::invalid_argument("size and shape mismatch");
  }

  bool is_1D = false;
  if (shape.size() == 1) {
    is_1D = true;
    shape.emplace_back(shape[0]);
    size = shape[0] * shape[0];
  }

  if (a.size() == 0) {
    return {a, shape};
  }

  int32_t ndim = static_cast<int32_t>(shape.size());
  size_t N     = shape[ndim - 2];
  size_t M     = shape[ndim - 1];
  std::vector<T> out;
  for (size_t idx = 0; idx < size; ++idx) {
    int32_t j = static_cast<int32_t>(idx % M);
    int32_t i = static_cast<int32_t>((idx / M) % N);
    bool flag = lower ? j <= i + k : j >= i + k;
    if (flag) {
      if (is_1D) {
        out.emplace_back(a[j]);
      } else {
        out.emplace_back(a[idx]);
      }
    } else {
      out.emplace_back(0);
    }
  }
  return {out, shape};
}

template <typename T>
void _test(std::string func, std::vector<T> x_in, std::vector<uint64_t> shape, int32_t k)
{
  bool lower            = (func == "tril") ? true : false;
  auto num_f            = (func == "tril") ? tril : triu;
  auto x                = mk_array(x_in, shape);
  auto x_out            = num_f(x, k);
  auto [x_gt, shape_gt] = trilu_result(x_in, shape, k, lower);
  check_array(x_out, x_gt, shape_gt);
}

TEST(Trilu, test_trilu)
{
  std::vector<std::string> func_list{"tril", "triu"};
  std::vector<int32_t> k_list{0, -1, 1, -2, 2, -10, 10};
  std::vector<std::vector<uint64_t>> shape_list{
    {0}, {1}, {10}, {1, 10}, {10, 10}, {1, 1, 10}, {1, 10, 10}, {10, 10, 10}};
  for (auto shape : shape_list) {
    auto x_int32 = mk_seq_vector<int32_t>(shape, 0, 1);
    auto x_float = mk_seq_vector<float>(shape, 0, 1);
    for (auto k : k_list) {
      for (auto func : func_list) {
        _test(func, x_int32, shape, k);
        _test(func, x_float, shape, k);
      }
    }
  }
}

TEST(Trilu, test_ndim)
{
  std::vector<std::string> func_list{"tril", "triu"};
  std::vector<uint64_t> shape;
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.push_back(ndim);
    for (int32_t k = -ndim; k <= ndim; ++k) {
      auto x_in = mk_seq_vector<int32_t>(shape);
      for (auto func : func_list) {
        _test(func, x_in, shape, k);
      }
    }
  }
}

class TriluErrors : public ::testing::Test {};

TEST_F(TriluErrors, test_m_scalar)
{
  auto x = mk_array<int32_t>({0});
  EXPECT_THROW(tril(x), std::invalid_argument);
}

}  // namespace
