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

template <typename T, typename U, typename V>
std::vector<T> put_result(std::vector<T> const& a,
                          std::vector<U> const& indices,
                          std::vector<V> const& values,
                          std::string mode = "raise")
{
  if (a.size() == 0 || indices.size() == 0 || values.size() == 0) {
    return a;
  }
  std::vector<T> out(a);
  int64_t size = static_cast<int64_t>(a.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    auto val = static_cast<T>(values[i % values.size()]);
    auto ind = static_cast<int64_t>(indices[i]);
    if (mode == "wrap") {
      ind %= size;
    } else if (mode == "clip") {
      if (ind < 0) {
        ind = 0;
      }
      if (ind >= size) {
        ind = size - 1;
      }
    }
    if (ind < 0) {
      ind += size;
    }
    if (!(ind >= 0 && ind < size)) {
      throw std::invalid_argument("vector index out of bounds");
    }
    out[ind] = val;
  }
  return out;
}

TEST(Put, test_scalar_indices_values)
{
  {
    auto x       = mk_array<int32_t>({1, 2, 3, 4, 5, 6}, {2, 3});
    auto indices = mk_array<int32_t>({0});
    auto values  = mk_array<int32_t>({10});
    put(x, indices, values);  // put(x, 0, 10)
    check_array<int32_t>(x, {10, 2, 3, 4, 5, 6}, {2, 3});
  }
  {
    auto x       = mk_array<int32_t>({1, 2, 3, 4, 5, 6}, {2, 3});
    auto indices = mk_array<int32_t>({0});
    auto values  = mk_array<int32_t>({10, 20, 30});
    put(x, indices, values);  // put(x, 0, [10, 20, 30])
    check_array<int32_t>(x, {10, 2, 3, 4, 5, 6}, {2, 3});
  }
  {
    auto x       = mk_array<int32_t>({1, 2, 3, 4, 5, 6}, {2, 3});
    auto indices = mk_array<int32_t>({0}, {1});
    auto values  = mk_array<int64_t>({10});
    put(x, indices, values);  // put(x, [0], 10)
    check_array<int32_t>(x, {10, 2, 3, 4, 5, 6}, {2, 3});
  }
  {
    auto x       = mk_array<int32_t>({1, 2, 3, 4, 5, 6}, {2, 3});
    auto indices = mk_array<float>({0, 1, 2.5, 1});
    auto values  = mk_array<float>({10.5});
    put(x, indices, values);  // put(x, [0, 1, 2.5, 1], 10)
    check_array<int32_t>(x, {10, 10, 10, 4, 5, 6}, {2, 3});
  }
}

TEST(Put, test_scalar_indices_values_mode)
{
  std::vector<std::string> mode_list{"wrap", "clip"};
  std::vector<std::vector<int32_t>> values_list{{10}, {10, 20}};
  std::vector<std::vector<int32_t>> indices_list{{100}, {-100}};

  std::vector<uint64_t> shape{3, 4, 5};
  auto x_in = mk_seq_vector<int32_t>(shape);

  for (auto indices : indices_list) {
    for (auto values : values_list) {
      for (auto mode : mode_list) {
        auto x   = mk_array(x_in, shape);
        auto v   = mk_array(values);
        auto ind = mk_array(indices);
        put(x, ind, v, mode);
        auto x_gt = put_result(x_in, indices, values, mode);
        check_array(x, x_gt, shape);
      }
    }
  }
}

TEST(Put, test_scalar_arr)
{
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> values_list{
    {{10}, {}}, {{10}, {1}}, {{10, 20}, {}}};
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> indices_list{
    {{0}, {}}, {{0}, {1}}, {{-1}, {}}, {{-1}, {1}}};
  std::vector<int32_t> x_in{0};
  for (auto [indices, shape_ind] : indices_list) {
    for (auto [values, shape_val] : values_list) {
      auto x   = mk_array(x_in);
      auto v   = mk_array(values, shape_val);
      auto ind = mk_array(indices, shape_ind);
      put(x, ind, v);
      auto x_gt = put_result(x_in, indices, values);
      check_array(x, x_gt);
    }
  }
}

TEST(Put, test_scalar_arr_mode)
{
  std::vector<std::string> mode_list{"wrap", "clip"};
  std::vector<std::vector<int32_t>> indices_list{{-1}, {1}, {-1, 0}, {-1, 0, 1, 2}};
  std::vector<int32_t> values{10};
  std::vector<int32_t> x_in{0};

  for (auto indices : indices_list) {
    for (auto mode : mode_list) {
      auto x   = mk_array(x_in);
      auto v   = mk_array(values);
      auto ind = mk_array(indices);
      put(x, ind, v, mode);
      auto x_gt = put_result(x_in, indices, values, mode);
      check_array(x, x_gt);
    }
  }
}

TEST(Put, test_indices_type_convert)
{
  std::vector<uint64_t> shape{3, 4, 5};
  auto x_in   = mk_seq_vector<int64_t>(shape);
  auto values = mk_seq_vector<int64_t>({6}, 10);
  std::vector<int32_t> indices{-2, 2};
  auto x   = mk_array(x_in);
  auto v   = mk_array(values);
  auto ind = mk_array(indices);
  put(x, ind, v);
  auto x_gt = put_result(x_in, indices, values);
  check_array(x, x_gt);
}

TEST(Put, test_indices_array_and_shape_array)
{
  std::vector<std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>> INDICES_VALUES_SHAPE{
    {{0}, {1}},
    {{2}, {0}},
    {{2}, {1}},
    {{2}, {2}},
    {{2}, {3}},
    {{2}, {2, 1}},
    {{2}, {3, 2}},
    {{2, 2}, {1}},
    {{2, 2}, {4}},
    {{2, 2}, {5}},
    {{2, 2}, {2, 1}},
    {{2, 2}, {2, 2}},
    {{2, 2}, {3, 3}},
  };
  std::vector<std::vector<uint64_t>> shape_list{{2, 3, 4}, {6}};

  for (auto shape : shape_list) {
    for (auto [shape_ind, shape_val] : INDICES_VALUES_SHAPE) {
      auto x_in    = mk_seq_vector<int32_t>(shape);
      auto indices = mk_seq_vector<int64_t>(shape_ind);
      auto values  = mk_seq_vector<int32_t>(shape_val, 10);
      auto x       = mk_array(x_in, shape);
      auto v       = mk_array(values, shape_val);
      auto ind     = mk_array(indices, shape_ind);
      put(x, ind, v);
      auto x_gt = put_result(x_in, indices, values);
      check_array(x, x_gt, shape);
    }
  }
}

TEST(Put, test_ndim_default_mode)
{
  std::vector<uint64_t> shape, shape_ind, shape_val;

  for (int ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.push_back(5);
    shape_ind.push_back(3);
    shape_val.push_back(2);
    auto x_in    = mk_seq_vector<int64_t>(shape);
    auto indices = mk_seq_vector<int64_t>(shape_ind);
    auto values  = mk_seq_vector<int64_t>(shape_val, 10);
    auto x       = mk_array(x_in, shape);
    auto v       = mk_array(values, shape_val);
    auto ind     = mk_array(indices, shape_ind);
    put(x, ind, v);
    auto x_gt = put_result(x_in, indices, values);
    check_array(x, x_gt, shape);
  }
}

TEST(Put, test_ndim_mode)
{
  std::vector<std::string> mode_list{"wrap", "clip"};
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> INDICES = {
    {{1, 2, 3.2, 100}, {}}, {{2, 1, 3, 100}, {2, 2}}, {{1}, {1}}, {{100}, {1}}};

  std::vector<uint64_t> shape, shape_val;
  for (int ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.push_back(5);
    shape_val.push_back(2);
    auto x_in   = mk_seq_vector<int64_t>(shape);
    auto values = mk_seq_vector<int64_t>(shape_val, 10);
    for (auto [indices, shape_ind] : INDICES) {
      for (auto mode : mode_list) {
        auto x   = mk_array(x_in, shape);
        auto v   = mk_array(values, shape_val);
        auto ind = mk_array(indices, shape_ind);
        put(x, ind, v, mode);
        auto x_gt = put_result(x_in, indices, values, mode);
        check_array(x, x_gt, shape);
      }
    }
  }
}

TEST(Put, test_empty_array)
{
  auto x       = mk_array<int32_t>({}, {0});
  auto values  = mk_array<int32_t>({10});
  auto indices = mk_array<int64_t>({}, {0});
  put(x, indices, values);
  check_array<int32_t>(x, {}, {0});
}

TEST(Put, test_indices_out_of_bound)
{
  std::vector<std::vector<int32_t>> indices_list{{-13}, {12}, {0, 1, 12}};
  std::vector<uint64_t> shape{3, 4};
  auto x_in = mk_seq_vector<int32_t>(shape);
  auto x    = mk_array(x_in, shape);
  auto v    = mk_array<int32_t>({10});
  for (auto indices : indices_list) {
    auto ind = mk_array(indices);
    EXPECT_ANY_THROW(put(x, ind, v));
    EXPECT_ANY_THROW(put(x, ind, v, "raise"));
  }
}

TEST(Put, test_indices_out_of_bound_arr_is_scalar)
{
  std::vector<std::tuple<std::vector<int64_t>, std::vector<uint64_t>>> indices_list = {
    {{-2}, {}}, {{1}, {}}, {{1}, {1}}};
  auto x = mk_array<int32_t>({0});
  auto v = mk_array<int32_t>({10});
  for (auto [indices, shape_ind] : indices_list) {
    auto ind = mk_array(indices, shape_ind);
    EXPECT_ANY_THROW(put(x, ind, v));
    EXPECT_ANY_THROW(put(x, ind, v, "raise"));
  }
}

TEST(Put, test_invalid_mode)
{
  std::string mode = "unknown";
  std::vector<uint64_t> shape{3, 4};
  auto x_in = mk_seq_vector<int32_t>(shape);
  auto x    = mk_array(x_in, shape);
  auto ind  = mk_array<int32_t>({0});
  auto v    = mk_array<int32_t>({10});
  EXPECT_THROW(put(x, ind, v, mode), std::invalid_argument);
}

}  // namespace
