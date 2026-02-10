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
#include <limits>
#include <cmath>

using namespace cupynumeric;

namespace {

TEST(Fill, test_fill_empty_array)
{
  auto x = mk_array<int32_t>({}, {0});
  x.fill(Scalar(int32_t(1)));
  check_array<int32_t>(x, {}, {0});
}

TEST(Fill, test_fill_float_with_nan)
{
  auto x        = zeros({6}, legate::float32());
  float val_nan = std::numeric_limits<float>::quiet_NaN();
  x.fill(Scalar(val_nan));
  auto accessor = x.get_read_accessor<float, 1>();
  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_TRUE(std::isnan(accessor[i]));
  }
}

TEST(Fill, test_fill_inf_to_float)
{
  float val_inf                 = std::numeric_limits<float>::infinity();
  std::vector<float> INF_VALUES = {val_inf, -val_inf};
  for (auto value : INF_VALUES) {
    auto x = zeros({6}, legate::float32());
    std::vector<float> x_gt(6, value);
    x.fill(Scalar(value));
    check_array(x, x_gt);
  }
}

TEST(Fill, test_fill_float_to_float)
{
  std::vector<double> FLOAT_FILL_VALUES{-2.4e120, -1.3, 8.9e-130, 0.0, 5.7e-150, 0.6, 3.7e160};
  for (auto value : FLOAT_FILL_VALUES) {
    auto x = zeros({6}, legate::float64());
    std::vector<double> x_gt(6, value);
    x.fill(Scalar(value));
    check_array(x, x_gt);
  }
}

TEST(Fill, test_fill_ndim)
{
  std::vector<uint64_t> shape;
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.push_back(ndim);
    int32_t value = ndim * 10;
    auto x        = zeros(shape, legate::int32());
    auto x_gt     = mk_seq_vector<int32_t>(shape, 0, value);
    x.fill(Scalar(value));
    check_array(x, x_gt, shape);
  }
}

TEST(Fill, test_full_ndim)
{
  std::vector<uint64_t> shape;
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.push_back(ndim);
    int32_t value = ndim * 10;
    auto x        = full(shape, Scalar(value));
    auto x_gt     = mk_seq_vector<int32_t>(shape, 0, value);
    check_array(x, x_gt, shape);
  }
}

TEST(Fill, test_transformed_array)
{
  auto arr = zeros({2, 3}, legate::int32());

  auto transposed = arr.transpose();

  transposed.fill(Scalar(int32_t(1)));

  const std::vector<int32_t> expect{1, 1, 1, 1, 1, 1};
  const std::vector<uint64_t> expect_shape{3, 2};

  check_array(transposed, expect, expect_shape);
}

}  // namespace
