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

TEST(Convolve, test_dtype)
{
  auto x    = mk_array<int32_t>({1, 2, 3});
  auto y    = mk_array<float>({0, 1, 0.5});
  auto out1 = convolve(x, y);
  auto out2 = convolve(y, x);
  debug_array(out1);
  debug_array(out2);
  // out1 = [1, 2, 3], out2 = [1, 2.5, 4]
  // It is a bug.
  // It violates the "NumPy type promotion rules".
}

TEST(Convolve, test_empty)
{
  auto a = mk_array<float>({}, {0});
  auto v = mk_array<float>({}, {0});
  debug_array(a);
  debug_array(v);
  // An exception should be thrown, but it doesn't.
  auto out = convolve(a, v);
  debug_array(out);
}

TEST(Convolve, test_diff_dims)
{
  auto a = zeros({5, 5, 5});
  auto v = zeros({5, 5});
  EXPECT_ANY_THROW(convolve(a, v));
}

std::vector<std::tuple<std::vector<int32_t>,
                       std::vector<int32_t>,
                       std::vector<int32_t>,
                       std::vector<uint64_t>,
                       std::vector<uint64_t>>>
  test_data{
    {{0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0},
     {1, 1, 0, 1, 0, 0, 1},
     {0, 0, 1, 2, 1, 1, 1, 0, 2, 3, 2, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 0, 1, 0},
     {30},
     {7}},
    {{0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
      1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1,
      0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
      1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0},
     {1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1},
     {2, 3, 3, 3, 4, 4, 3, 5, 2, 3, 4, 3, 6, 5, 7, 4, 5, 4, 2, 2, 4, 4, 6, 5, 6,
      4, 5, 3, 3, 1, 3, 4, 6, 6, 5, 7, 4, 6, 3, 2, 4, 1, 4, 5, 4, 5, 5, 6, 2, 3,
      3, 1, 4, 3, 3, 4, 3, 3, 2, 1, 4, 5, 3, 5, 3, 4, 3, 3, 2, 2, 4, 5, 4, 6, 2,
      3, 1, 5, 2, 3, 3, 4, 4, 5, 5, 5, 3, 4, 1, 2, 0, 2, 3, 2, 2, 2, 3, 1, 0, 0},
     {10, 10},
     {3, 5}},
    {{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
      0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
      0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1,
      0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0},
     {1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0},
     {2, 1, 3, 3, 2, 1, 4, 2, 3, 4, 5, 6, 3, 4, 2, 3, 4, 3, 4, 1, 1, 2, 3, 2, 1,
      1, 3, 3, 5, 2, 5, 7, 6, 5, 2, 5, 7, 4, 4, 4, 4, 6, 6, 5, 3, 2, 3, 2, 2, 1,
      4, 3, 6, 5, 4, 1, 7, 7, 7, 3, 6, 8, 8, 6, 3, 3, 5, 4, 5, 3, 2, 2, 3, 4, 2,
      2, 3, 2, 5, 2, 4, 5, 9, 4, 3, 2, 7, 7, 4, 1, 4, 7, 7, 8, 1, 0, 4, 3, 4, 3,
      3, 3, 3, 4, 2, 1, 5, 2, 4, 2, 2, 4, 5, 3, 2, 1, 4, 5, 4, 2, 2, 3, 3, 3, 0},
     {5, 5, 5},
     {3, 3, 3}}};

TEST(Convolve, test_int)
{
  for (auto [a_in, v_in, out_gt, shape_a, shape_v] : test_data) {
    auto a   = mk_array(a_in, shape_a);
    auto v   = mk_array(v_in, shape_v);
    auto out = convolve(a, v);
    check_array(out, out_gt, shape_a);
    debug_array(out, false);
  }
}

TEST(Convolve, test_double)
{
  for (auto [a_in, v_in, out_gt, shape_a, shape_v] : test_data) {
    auto a   = mk_array(as_type_vector<double>(a_in), shape_a);
    auto v   = mk_array(as_type_vector<double>(v_in), shape_v);
    auto out = convolve(a, v);
    check_array(out, as_type_vector<double>(out_gt), shape_a);
    debug_array(out, false);
  }
}

TEST(Convolve, test_ndim)
{
  std::vector<uint64_t> shape;
  std::vector<uint64_t> filter_shape;
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.emplace_back(5);
    filter_shape.emplace_back(3);
    auto a_in             = mk_seq_vector<int32_t>(shape);
    auto v_in             = mk_seq_vector<int32_t>(filter_shape, 0, 0);
    v_in[v_in.size() / 2] = 1;
    auto a                = mk_array(a_in, shape);
    auto v                = mk_array(v_in, filter_shape);
    if (ndim <= 3) {
      auto out = convolve(a, v);
      check_array(out, a_in, shape);
      debug_array(out, false);
    } else {
      EXPECT_ANY_THROW(convolve(a, v));
    }
  }
}

}  // namespace
