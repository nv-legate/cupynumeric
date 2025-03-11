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

using namespace cupynumeric;

namespace {

std::vector<std::vector<int64_t>> SQUARE_CASES{
  {10, 5, 2},
  // (-1, 5, 2),
  {5, 2, 10},
  {5, 2, 5, 2},
  {10, 10, 1},
  {10, 1, 10},
  {1, 10, 10},
};

class Reshape_TestSquare : public ::testing::Test {
 public:
  std::vector<int32_t> a_gt = mk_seq_vector<int32_t>({100}, 1, -1);
};

TEST_F(Reshape_TestSquare, test_basic)
{
  auto a = arange<int32_t>(100).reshape({10, -1});
  check_array(a, a_gt, {10, 10});
}

TEST_F(Reshape_TestSquare, test_shape)
{
  for (auto shape : SQUARE_CASES) {
    auto a = arange<int32_t>(100).reshape({10, 10});
    check_array(reshape(a, shape), a_gt, as_type_vector<uint64_t>(shape));
  }
  {
    auto a = arange<int32_t>(100).reshape({10, 10});
    check_array(reshape(a, {-1, 5, 2}), a_gt, {10, 5, 2});
  }
}

TEST_F(Reshape_TestSquare, test_shape_mode)
{
  for (std::string order : {"C", "F", "A"}) {
    for (auto shape : SQUARE_CASES) {
      auto a = arange<int32_t>(100).reshape({10, 10});
      if (order == "F") {
        EXPECT_THROW(reshape(a, shape, order), std::invalid_argument);
      } else {
        check_array(reshape(a, shape, order), a_gt, as_type_vector<uint64_t>(shape));
      }
    }
    {
      auto a = arange<int32_t>(100).reshape({10, 10});
      if (order == "F") {
        EXPECT_THROW(reshape(a, {-1, 5, 2}, order), std::invalid_argument);
      } else {
        check_array(reshape(a, {-1, 5, 2}, order), a_gt, {10, 5, 2});
      }
    }
  }
}

TEST_F(Reshape_TestSquare, test_1d)
{
  auto a = arange<int32_t>(100).reshape({10, 10});
  check_array(reshape(a, {100}), a_gt, {100});
}

TEST_F(Reshape_TestSquare, test_ravel)
{
  auto a = arange<int32_t>(100).reshape({10, 10});
  check_array(ravel(a), a_gt, {100});
}

TEST_F(Reshape_TestSquare, test_ravel_empty_array)
{
  auto a = full({3, 0}, Scalar(int64_t(1)));
  check_array<int64_t>(ravel(a), {}, {0});

  a = full({0, 3}, Scalar(int64_t(1)));
  check_array<int64_t>(ravel(a), {}, {0});
}

std::vector<std::vector<int64_t>> RECT_CASES = {
  {10, 2, 10},
  {20, 10},
  // {20, -5},
  {5, 40},
  {200, 1},
  {1, 200},
  {10, 20},
};

class Reshape_TestRect : public ::testing::Test {
 public:
  std::vector<int32_t> a_gt = mk_seq_vector<int32_t>({200});
};

TEST_F(Reshape_TestRect, test_shape)
{
  for (auto shape : RECT_CASES) {
    auto a = mk_array(a_gt, {5, 4, 10});
    check_array(reshape(a, shape), a_gt, as_type_vector<uint64_t>(shape));
  }
  {
    auto a = mk_array(a_gt, {5, 4, 10});
    check_array(reshape(a, {20, -5}), a_gt, {20, 10});
  }
}

TEST_F(Reshape_TestRect, test_shape_mode)
{
  for (std::string order : {"C", "F", "A"}) {
    for (auto shape : RECT_CASES) {
      auto a = mk_array(a_gt, {5, 4, 10});
      if (order == "F") {
        EXPECT_THROW(reshape(a, shape, order), std::invalid_argument);
      } else {
        check_array(reshape(a, shape, order), a_gt, as_type_vector<uint64_t>(shape));
      }
    }
    {
      auto a = mk_array(a_gt, {5, 4, 10});
      if (order == "F") {
        EXPECT_THROW(reshape(a, {20, -5}, order), std::invalid_argument);
      } else {
        check_array(reshape(a, {20, -5}, order), a_gt, {20, 10});
      }
    }
  }
}

TEST_F(Reshape_TestRect, test_ravel)
{
  for (std::string order : {"C", "F", "A"}) {
    auto a = mk_array(a_gt, {5, 4, 10});
    if (order == "F") {
      EXPECT_THROW(ravel(a, order), std::invalid_argument);
    } else {
      check_array(ravel(a, order), a_gt, {200});
    }
  }
}

TEST(Reshape, test_reshape_empty_array)
{
  std::vector<std::vector<int64_t>> shape_list{
    {0},
    {1, 0},
    {0, 1, 1},
  };
  auto a = mk_array<int32_t>({}, {0, 1});
  for (auto shape : shape_list) {
    check_array<int32_t>(reshape(a, shape), {}, as_type_vector<uint64_t>(shape));
  }
}

TEST(Reshape, test_reshape_same_shape)
{
  std::vector<uint64_t> shape{1, 2, 3};
  auto a_gt  = mk_seq_vector<int32_t>(shape);
  auto a     = mk_array<int32_t>(a_gt, shape);
  auto a_out = reshape(a, as_type_vector<int64_t>(shape));
  check_array(a_out, a_gt, shape);
}

class Reshape_Errors : public ::testing::Test {
 public:
  NDArray a = mk_array(mk_seq_vector<int32_t>({24}));
};

TEST_F(Reshape_Errors, test_empty_array_shape_invalid_size)
{
  auto a = mk_array<int32_t>({}, {0, 1, 1});
  std::vector<int64_t> shape{1, 1};
  EXPECT_THROW(reshape(a, shape), std::invalid_argument);
}

TEST_F(Reshape_Errors, test_shape_invalid_size)
{
  std::vector<std::vector<int64_t>> shape_list{
    {-1, 0, 2},
    {4, 3, 4},
    {4, 3, 0},
    {4, 3},
    {4},
    {0},
  };
  for (auto shape : shape_list) {
    EXPECT_THROW(reshape(a, shape), std::invalid_argument);
  }
}

TEST_F(Reshape_Errors, test_shape_unknown_dimensions)
{
  std::vector<int64_t> shape{-5, -1, 2};
  EXPECT_THROW(reshape(a, shape), std::invalid_argument);
}

TEST_F(Reshape_Errors, test_invalid_order)
{
  EXPECT_THROW(reshape(a, {4, 3, 2}, "Z"), std::invalid_argument);
}

TEST_F(Reshape_Errors, test_reshape_no_args)
{
  {
    auto a = mk_array<int32_t>({1}, {1, 1, 1});
    check_array<int32_t>(reshape(a, {}), {1});
  }
  {
    auto a = mk_array<int32_t>({}, {0});
    EXPECT_THROW(reshape(a, {}), std::invalid_argument);
  }
}

}  // namespace
