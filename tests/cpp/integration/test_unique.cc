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
#include <algorithm>
#include <functional>

using namespace cupynumeric;

namespace {

template <typename T>
std::vector<T> unique_result(std::vector<T> v)
{
  std::sort(v.begin(), v.end());
  auto last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());
  return v;
}

TEST(Unique, test_basic)
{
  {
    auto x     = mk_array<int32_t>({1, 1, 2, 2, 3, 3});
    auto x_out = x.unique();
    check_array<int32_t>(x_out, {1, 2, 3});
  }
  {
    auto x     = mk_array<int32_t>({1, 1, 2, 3}, {2, 2});
    auto x_out = unique(x);
    check_array<int32_t>(x_out, {1, 2, 3});
  }
  {
    std::vector<int32_t> x_in{1, 2, 1, 1, 3, 3, 3, 4, 5, 4};
    auto x_gt  = unique_result(x_in);
    auto x     = mk_array(x_in);
    auto x_out = unique(x);
    check_array(x_out, x_gt);
  }
}

TEST(Unique, test_scalar)
{
  // If x is a 0-D scalar, "List of axes to broadcast must not be empty" exception will be thrown.
  // auto x     = mk_array<int64_t>({99});
  auto x     = mk_array<int64_t>({99}, {1});
  auto x_out = x.unique();
  check_array<int64_t>(x_out, {99}, {1});
}

template <typename T>
std::vector<T> mk_random_vector(std::vector<uint64_t> shape, std::function<T()> gen)
{
  uint64_t size =
    std::accumulate(shape.begin(), shape.end(), uint64_t(1), std::multiplies<uint64_t>());
  std::vector<T> v(size);
  std::generate(v.begin(), v.end(), gen);
  return v;
}

static int randint(int low, int high) { return rand() % (high - low) + low; }

TEST(Unique, test_ndim)
{
  srand(111);
  std::vector<uint64_t> shape;
  size_t size = 1;
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.emplace_back(4);
    size *= 4;
    auto x_in =
      mk_random_vector<int32_t>(shape, [] { return static_cast<int32_t>(randint(0, 10)); });
    auto x_gt  = unique_result(x_in);
    auto x     = mk_array(x_in, shape);
    auto x_out = unique(x);
    check_array(x_out, x_gt);
  }
}

}  // namespace
