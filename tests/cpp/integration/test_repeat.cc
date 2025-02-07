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
#include <cassert>
#include <cstdlib>

using namespace cupynumeric;

namespace {

template <typename T>
std::tuple<std::vector<T>, std::vector<uint64_t>> repeat_result(std::vector<T> const& a,
                                                                std::vector<int64_t> repeats,
                                                                int32_t axis                = 0,
                                                                std::vector<uint64_t> shape = {})
{
  if (shape.empty()) {
    shape.push_back(a.size());
  }

  if (axis < 0) {
    axis += shape.size();
  }
  assert(axis >= 0 && axis < shape.size());

  size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
  assert(a.size() == size);

  assert(repeats.size() == 1 || repeats.size() == shape[axis]);
  assert(std::none_of(repeats.begin(), repeats.end(), [](int64_t x) { return x < 0; }));
  if (repeats.size() != shape[axis]) {
    repeats.resize(shape[axis], repeats[0]);
  }

  int64_t new_extent = 0;
  size_t new_size    = 0;
  if (shape[axis] != 0) {
    new_extent = std::accumulate(repeats.begin(), repeats.end(), int64_t(0));
    new_size   = size / shape[axis] * new_extent;
  }
  std::vector<uint64_t> new_shape{shape};
  new_shape[axis] = new_extent;

  std::vector<size_t> idx;
  for (size_t i = 0; i < repeats.size(); ++i) {
    idx.insert(idx.end(), repeats[i], i);
  }

  size_t step1 =
    std::accumulate(shape.begin() + (axis + 1), shape.end(), size_t(1), std::multiplies<size_t>());
  size_t step0     = step1 * shape[axis];
  size_t new_step0 = step1 * new_extent;

  std::vector<T> out(new_size);
  for (size_t i = 0; i < out.size(); ++i) {
    size_t i0 = i / new_step0;
    size_t r  = i % new_step0;
    size_t i1 = r / step1;
    size_t i2 = r % step1;
    i1        = idx[i1];
    out[i]    = a[step0 * i0 + step1 * i1 + i2];
  }
  return {out, new_shape};
}

TEST(Repeat, test_basic)
{
  std::vector<uint64_t> shape{2, 2};
  auto x_in = mk_seq_vector<int32_t>(shape);
  std::vector<int64_t> repeats{1, 2};
  int32_t axis          = 0;
  auto [x_gt, shape_gt] = repeat_result(x_in, repeats, axis, shape);

  auto x     = mk_array(x_in, shape);
  auto rep   = mk_array(repeats);
  auto x_out = repeat(x, rep, axis);
  check_array(x_out, x_gt, shape_gt);
}

TEST(Repeat, test_repeats_none)
{
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> x_in_list{
    {{}, {0}}, {{4}, {}}, {{2, 3}, {}}, {mk_seq_vector<int32_t>({3, 4, 2}), {3, 4, 2}}};
  for (auto [x_in, shape] : x_in_list) {
    auto x   = mk_array(x_in, shape);
    auto rep = mk_array<int32_t>({}, {0});
    EXPECT_THROW(repeat(x, rep), std::invalid_argument);
  }
};

TEST(Repeat, test_array_empty_repeats_valid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{
    {{0}, {}}, {{3}, {}}, {{4.7}, {}}, {{0}, {1}}, {{3}, {1}}, {{4.7}, {1}}};
  for (auto [repeats, shape] : repeats_list) {
    auto rep   = mk_array(repeats, shape);
    auto x     = mk_array<double>({}, {0});
    auto x_out = repeat(x, rep);
    check_array<double>(x_out, {}, {0});
  }
}

// numpy fail, cupynumeric pass
TEST(Repeat, test_array_empty_repeats_invalid_negative)
{
  std::vector<std::vector<uint64_t>> repeats_list{{3, 4}, {1, 2, 3}};
  for (auto repeats : repeats_list) {
    auto rep   = mk_array(repeats);
    auto x     = mk_array<int32_t>({}, {0});
    auto x_out = repeat(x, rep);
    check_array<int32_t>(x_out, {}, {0});
  }
}

TEST(Repeat, test_array_empty_axis_valid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{
    {{0}, {}}, {{3}, {}}, {{4.7}, {}}, {{0}, {1}}, {{3}, {1}}, {{4.7}, {1}}};
  for (auto [repeats, shape] : repeats_list) {
    auto rep   = mk_array(repeats, shape);
    auto x     = mk_array<double>({}, {1, 0});
    auto x_out = repeat(x, rep, 1);
    check_array<double>(x_out, {}, {1, 0});
  }
}

TEST(Repeat, test_array_empty_axis_invalid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{
    {{0}, {}}, {{3}, {}}, {{4.7}, {}}, {{0}, {1}}, {{3}, {1}}, {{4.7}, {1}}};
  for (auto [repeats, shape] : repeats_list) {
    auto rep = mk_array(repeats, shape);
    auto x   = mk_array<double>({}, {0});
    EXPECT_THROW(repeat(x, rep, 1), std::invalid_argument);
  }
}

TEST(Repeat, test_array_int_axis_negative)
{
  std::vector<int32_t> axis_list{-3, 3};
  for (auto axis : axis_list) {
    auto x      = mk_array<int32_t>({3});
    int64_t rep = 3;
    EXPECT_THROW(repeat(x, rep, axis), std::invalid_argument);
  }
}

TEST(Repeat, test_array_int_repeats_negative)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{{{-3}, {}},
                                                                                   {{-3}, {1}}};
  for (auto [repeats, shape] : repeats_list) {
    auto x = mk_array<int32_t>({3});
    EXPECT_THROW(repeat(x, repeats[0]), std::invalid_argument);
  }
}

TEST(Repeat, test_array_int_repeats_valid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{
    {{0}, {}}, {{3}, {}}, {{4.7}, {}}, {{0}, {1}}, {{3}, {1}}, {{4.7}, {1}}};
  for (auto [repeats, shape] : repeats_list) {
    auto x                = mk_array<int32_t>({3});
    auto x_out            = repeat(x, int64_t(repeats[0]));
    auto [x_gt, shape_gt] = repeat_result<int32_t>({3}, as_type_vector<int64_t>(repeats));
    check_array(x_out, x_gt, shape_gt);
  }
}

TEST(Repeat, test_array_int_repeats_invalid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{{{}, {0}},
                                                                                   {{1, 2}, {}}};
  for (auto [repeats, shape] : repeats_list) {
    auto rep = mk_array(repeats, shape);
    auto x   = mk_array<int32_t>({3});
    EXPECT_THROW(repeat(x, rep), std::invalid_argument);
  }
}

TEST(Repeat, test_array_1d_repeats_valid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{
    {{0}, {}}, {{3}, {}}, {{4.7}, {}}, {{0}, {1}}, {{3}, {1}}, {{4.7}, {1}}, {{2, 3, 4}, {}}};
  std::vector<int32_t> x_in{1, 2, 3};
  auto x = mk_array(x_in);
  for (auto [repeats, shape] : repeats_list) {
    auto rep              = mk_array(repeats, shape);
    auto x_out            = repeat(x, rep);
    auto [x_gt, shape_gt] = repeat_result(x_in, as_type_vector<int64_t>(repeats));
    check_array(x_out, x_gt, shape_gt);
  }
}

TEST(Repeat, test_array_1d_repeats_invalid)
{
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> repeats_list{{{}, {0}},
                                                                                    {{2, 3}, {}}};
  auto x = mk_array<int32_t>({1, 2, 3});
  for (auto [repeats, shape] : repeats_list) {
    auto rep = mk_array(repeats, shape);
    EXPECT_ANY_THROW(repeat(x, rep));
  }
}

TEST(Repeat, test_array_2d_repeats_valid)
{
  std::vector<std::tuple<std::vector<double>, std::vector<uint64_t>>> repeats_list{
    {{0}, {}}, {{0}, {1}}, {{3}, {}}, {{4.7}, {}}, {{3}, {1}}, {{4.7}, {1}}};
  std::vector<int32_t> x_in = {1, 3, 2, 4};
  auto x                    = mk_array(x_in, {2, 2});
  for (auto [repeats, shape] : repeats_list) {
    auto rep              = mk_array(repeats, shape);
    auto x_out            = repeat(x, rep);
    auto [x_gt, shape_gt] = repeat_result(x_in, as_type_vector<int64_t>(repeats));
    check_array(x_out, x_gt, shape_gt);
  }
}

TEST(Repeat, test_array_2d_repeats_invalid)
{
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> repeats_list{{{}, {0}},
                                                                                    {{2, 3}, {}}};
  auto x = mk_array<int32_t>({1, 3, 2, 4}, {2, 2});
  for (auto [repeats, shape] : repeats_list) {
    auto rep = mk_array(repeats, shape);
    EXPECT_ANY_THROW(repeat(x, rep));
  }
}

TEST(Repeat, test_array_1d_repeats_fatal_error)
{
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> arr_list{
    {{1, 2, 3}, {}}, {{1, 3, 2, 4}, {2, 2}}};
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> repeats_list{{{-3}, {}},
                                                                                    {{-3}, {1}}};
  for (auto [repeats, rep_shape] : repeats_list) {
    auto rep = mk_array(repeats, rep_shape);
    for (auto [x_in, x_shape] : arr_list) {
      auto x = mk_array(x_in, x_shape);
      // @pytest.mark.skip()
      // Aborted, got fatal error: "out of memory"
      // EXPECT_ANY_THROW(repeat(x, rep));
    }
  }
}

TEST(Repeat, test_repeats_nd)
{
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> arr_list{
    {{}, {0}}, {{3}, {}}, {{1, 2, 3}, {}}, {{1, 3, 2, 4}, {2, 2}}};
  std::vector<std::tuple<std::vector<int32_t>, std::vector<uint64_t>>> repeats_list{
    {{2, 3, 3, 3}, {2, 2}}, {mk_seq_vector<int32_t>({3, 3, 3}), {3, 3, 3}}};
  for (auto [repeats, rep_shape] : repeats_list) {
    auto rep = mk_array(repeats, rep_shape);
    for (auto [x_in, x_shape] : arr_list) {
      auto x = mk_array(x_in, x_shape);
      EXPECT_THROW(repeat(x, rep), std::invalid_argument);
    }
  }
}

TEST(Repeat, test_array_axis_out_bound)
{
  auto x      = mk_array<int32_t>({1, 2, 3, 4, 5});
  int64_t rep = 4;
  EXPECT_THROW(repeat(x, rep, 2), std::invalid_argument);
}

TEST(Repeat, test_array_axis_negative_equal)
{
  auto x                = mk_array<int32_t>({1, 2, 3, 4, 5});
  int64_t rep           = 4;
  auto x_out            = repeat(x, rep, -1);
  auto [x_gt, shape_gt] = repeat_result<int32_t>({1, 2, 3, 4, 5}, {4}, -1);
  check_array(x_out, x_gt, shape_gt);
}

static int randint(int low, int high) { return rand() % (high - low) + low; }

TEST(Repeat, test_nd_basic)
{
  srand(111);
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    std::vector<uint64_t> shape;
    for (int32_t i = 0; i < ndim; ++i) {
      shape.push_back(randint(1, 9));
    }
    auto x_in                    = mk_seq_vector<int32_t>(shape);
    std::vector<int64_t> repeats = {randint(0, 15)};
    auto x                       = mk_array(x_in, shape);
    auto x_out                   = repeat(x, repeats[0]);
    auto [x_gt, shape_gt]        = repeat_result(x_in, repeats);
    check_array(x_out, x_gt, shape_gt);
  }
}

TEST(Repeat, test_nd_axis)
{
  srand(222);
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    for (int32_t axis = 0; axis < ndim; ++axis) {
      std::vector<uint64_t> shape;
      for (int32_t i = 0; i < ndim; ++i) {
        shape.push_back(randint(1, 9));
      }
      auto x_in                    = mk_seq_vector<int32_t>(shape);
      std::vector<int64_t> repeats = {randint(0, 15)};
      auto x                       = mk_array(x_in, shape);
      auto x_out                   = repeat(x, repeats[0], axis);
      auto [x_gt, shape_gt]        = repeat_result(x_in, repeats, axis, shape);
      check_array(x_out, x_gt, shape_gt);
    }
  }
}

TEST(Repeat, test_nd_repeats)
{
  srand(333);
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    std::vector<uint64_t> shape;
    for (int32_t i = 0; i < ndim; ++i) {
      shape.push_back(randint(1, 9));
    }
    auto x_in = mk_seq_vector<int32_t>(shape);
    auto x    = mk_array(x_in, shape);
    for (int32_t axis = 0; axis < ndim; ++axis) {
      auto repeats          = mk_seq_vector<int64_t>({shape[axis]});
      auto rep              = mk_array(repeats);
      auto x_out            = repeat(x, rep, axis);
      auto [x_gt, shape_gt] = repeat_result(x_in, repeats, axis, shape);
      check_array(x_out, x_gt, shape_gt);
    }
  }
}

}  // namespace
