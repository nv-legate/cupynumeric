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

typedef std::vector<std::tuple<std::vector<size_t>, std::vector<int32_t>>> VEC_SHAPE_AXES;

std::vector<size_t> squeeze_result(
  const std::vector<size_t>& shape,
  std::optional<std::reference_wrapper<std::vector<int32_t> const>> axes = std::nullopt)
{
  std::vector<size_t> result;
  if (!axes.has_value()) {
    for (int i = 0; i < shape.size(); i++) {
      if (shape[i] != 1) {
        result.push_back(shape[i]);
      }
    }
  } else {
    auto computed_axes = normalize_axis_vector(axes.value(), shape.size());
    for (int i = 0; i < shape.size(); i++) {
      auto flag = true;
      if (shape[i] == 1) {
        for (int j = 0; j < computed_axes.size(); j++) {
          if (computed_axes[j] == i) {
            flag = false;
            break;
          }
        }
      }
      if (flag) {
        result.push_back(shape[i]);
      }
    }
  }
  return result;
}

void test_squeeze(
  const std::vector<size_t>& shape,
  std::optional<std::reference_wrapper<std::vector<int32_t> const>> axes = std::nullopt)
{
  auto vec_a        = mk_seq_vector<int32_t>(shape);
  auto arr_a        = mk_array<int32_t>(vec_a, shape);
  auto x            = axes.has_value() ? squeeze(arr_a, axes) : squeeze(arr_a);
  auto result_shape = squeeze_result(shape, axes);
  check_array<int32_t>(x, vec_a, result_shape);
}

static constexpr int32_t DIM           = 5;
std::vector<std::vector<size_t>> SIZES = {
  {},
  {
    0,
  },
  {1},
  {DIM},
  {0, 1},
  {1, 0},
  {1, 1},
  {1, DIM},
  {DIM, 1},
  {DIM, DIM},
  {1, 0, 0},
  {1, 1, 0},
  {1, 0, 1},
  {1, 1, 1},
  {DIM, 1, 1},
  {1, DIM, 1},
  {1, 1, DIM},
  {DIM, DIM, DIM},
};

VEC_SHAPE_AXES gen_shape_axes_all()
{
  VEC_SHAPE_AXES shape_axes;
  for (auto shape : SIZES) {
    std::vector<int32_t> axes;
    for (int i = 0; i < shape.size(); i++) {
      if (shape[i] == 1) {
        axes.push_back(i);
      }
    }
    shape_axes.push_back({shape, axes});
  }
  return shape_axes;
}

VEC_SHAPE_AXES gen_shape_axes_single()
{
  VEC_SHAPE_AXES shape_axes;
  for (auto shape : SIZES) {
    std::vector<int32_t> axes;
    for (int i = 0; i < shape.size(); i++) {
      if (shape[i] == 1) {
        axes.push_back(i);
      }
    }
    for (int i = 0; i < axes.size(); i++) {
      shape_axes.push_back({shape, {axes[i]}});
    }
  }
  return shape_axes;
}

VEC_SHAPE_AXES gen_shape_axes_negative()
{
  VEC_SHAPE_AXES shape_axes;
  for (auto shape : SIZES) {
    std::vector<int32_t> axes;
    for (int i = 0; i < shape.size(); i++) {
      if (shape[i] == 1) {
        axes.push_back(i - shape.size());
      }
    }
    if (axes.size() > 0) {
      shape_axes.push_back({shape, axes});
    }
  }
  return shape_axes;
}

TEST(Squeeze, Basic)
{
  for (auto shape : SIZES) {
    test_squeeze(shape);
  }
}

TEST(Squeeze, AxesAll)
{
  auto SHAPE_AXES = gen_shape_axes_all();
  for (auto [shape, axes] : SHAPE_AXES) {
    test_squeeze(shape, axes);
  }
}

TEST(Squeeze, AxesSingle)
{
  auto SHAPE_AXES = gen_shape_axes_single();
  for (auto [shape, axes] : SHAPE_AXES) {
    test_squeeze(shape, axes);
  }
}

TEST(Squeeze, AxesNegative)
{
  auto SHAPE_AXES = gen_shape_axes_negative();
  for (auto [shape, axes] : SHAPE_AXES) {
    test_squeeze(shape, axes);
  }
}

TEST(Squeeze, InvalidAxesNotEqualToOne)
{
  std::vector<size_t> shape                  = {1, 2, 1};
  std::vector<std::vector<int32_t>> vec_axes = {{
                                                  1,
                                                },
                                                {0, 1}};
  auto vec_a                                 = mk_seq_vector<int32_t>(shape);
  auto arr_a                                 = mk_array<int32_t>(vec_a, shape);
  for (auto axes : vec_axes) {
    EXPECT_THROW(squeeze(arr_a, axes), std::invalid_argument);
  }
}

TEST(Squeeze, InvalidAxesOutOfBound)
{
  std::vector<size_t> shape                  = {1, 2, 1};
  std::vector<std::vector<int32_t>> vec_axes = {{
                                                  3,
                                                },
                                                {0, 3},
                                                {-4},
                                                {-4, 0}};
  auto vec_a                                 = mk_seq_vector<int32_t>(shape);
  auto arr_a                                 = mk_array<int32_t>(vec_a, shape);
  for (auto axes : vec_axes) {
    EXPECT_THROW(squeeze(arr_a, axes), std::invalid_argument);
  }
}

TEST(Squeeze, InvalidAxesDuplicate)
{
  std::vector<size_t> shape                  = {1, 2, 1};
  std::vector<std::vector<int32_t>> vec_axes = {{0, -3}, {-1, 0, 2}};
  auto vec_a                                 = mk_seq_vector<int32_t>(shape);
  auto arr_a                                 = mk_array<int32_t>(vec_a, shape);
  for (auto axes : vec_axes) {
    EXPECT_THROW(squeeze(arr_a, axes), std::invalid_argument);
  }
}

}  // namespace
