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
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

namespace cupynumeric {

template <typename T>
void show_array(NDArray& a)
{
  auto acc = a.get_read_accessor<T, 1>();
  std::cerr << "[";
  for (size_t i = 0; i < a.size(); ++i) {
    std::cerr << acc[i];
    if (i != a.size() - 1) {
      std::cerr << ", ";
    }
  }
  std::cerr << "]" << std::endl;
}

void debug_array(NDArray a, bool show_data)
{
  auto store = a.get_store();
  if (store.has_scalar_storage()) {
    std::cerr << "(S) ";
  } else {
    std::cerr << "( ) ";
  }
  if (store.transformed()) {
    std::cerr << "(T) ";
  } else {
    std::cerr << "( ) ";
  }
  std::cerr << "<" << store.type().to_string() << "> " << store.to_string() << std::endl;
  if (!show_data) {
    return;
  }
  if (a.size() == 0) {
    std::cerr << "[]" << std::endl;
    return;
  }
  if (a.dim() > 1) {
    a = a._wrap(a.size());
  }
  switch (a.type().code()) {
    case legate::Type::Code::INT32: show_array<int32_t>(a); break;
    case legate::Type::Code::INT64: show_array<int64_t>(a); break;
    case legate::Type::Code::FLOAT32: show_array<float>(a); break;
    case legate::Type::Code::FLOAT64: show_array<double>(a); break;
    default: std::cerr << "[ Not implemented ]" << std::endl; break;
  }
}

}  // namespace cupynumeric

using namespace cupynumeric;

// unit test for common_utils
namespace {

TEST(Utils, test_check_array)
{
  {
    auto x = mk_array<int32_t>({99});
    debug_array(x);
  }
  {
    auto x = mk_array<int32_t>({99}, {1});
    debug_array(x);
  }
  {
    auto x = mk_array<int32_t>({1, 2, 3, 4}, {2, 2});
    debug_array(x);
    check_array<int32_t>(x, {1, 2, 3, 4}, {2, 2});
  }
  {
    std::vector<uint64_t> shape{2, 3, 4};
    auto x_in = mk_seq_vector<int32_t>(shape, 10);
    auto x    = mk_array(x_in, shape);
    debug_array(x);
    check_array(x, x_in, shape);
  }
}

void fail1()
{
  std::vector<uint64_t> shape{2, 3};
  auto x    = mk_array<int32_t>({1, 2, 3, 4, 50, 6}, shape);
  auto x_gt = mk_seq_vector<int32_t>(shape);
  check_array(x, x_gt, shape);
};

void fail2()
{
  auto x    = mk_array<float>({1 + 1e-8, 1 + 1e-7, 1 + 1e-6, 1 + 1e-5});
  auto x_gt = mk_seq_vector<float>({4}, 0, 1);
  check_array(x, x_gt);
};

void fail3()
{
  auto x    = mk_array<double>({1 + 1e-8, 1 + 1e-7, 1 + 1e-6, 1 + 1e-5});
  auto x_gt = mk_seq_vector<double>({4}, 0, 1);
  check_array(x, x_gt);
};

TEST(Utils, test_check_array_neg)
{
  EXPECT_FATAL_FAILURE(fail1(), "check_array");
  EXPECT_FATAL_FAILURE(fail2(), "check_array");
  EXPECT_FATAL_FAILURE(fail3(), "check_array");
}

TEST(Utils, test_as_type_vector)
{
  auto x = mk_seq_vector<double>({16}, 0.25);
  debug_vector(x);
  auto y = as_type_vector<int32_t>(x);
  debug_vector(y);
}

TEST(Utils, test_ndarray_wrap)
{
  auto x = mk_array<int32_t>({1, 2, 3, 4});
  debug_array(x);
  auto y = x._wrap(0);
  debug_array(y);
  auto z = y._wrap(0);
  debug_array(z);
  EXPECT_ANY_THROW(y._wrap(1););
}

TEST(Utils, test_ndarray_warn_and_convert)
{
  auto x_in = mk_seq_vector<double>({8}, 0.5);
  auto x    = mk_array(x_in);
  auto y    = x._warn_and_convert(legate::int32());
  debug_array(x);
  debug_array(y);
  cupynumeric_log().warning() << "Just a test!";
}

TEST(Utils, test_wrap_indices_and_clip_indices)
{
  std::vector<uint64_t> shape{10};
  auto x_in   = mk_seq_vector<int64_t>(shape);
  auto x      = mk_array(x_in, shape);
  auto x_warp = x.wrap_indices(Scalar(int64_t(4)));
  auto x_clip = x.clip_indices(Scalar(int64_t(3)), Scalar(int64_t(7)));
  debug_array(x);
  debug_array(x_warp);
  debug_array(x_clip);
}

}  // namespace
