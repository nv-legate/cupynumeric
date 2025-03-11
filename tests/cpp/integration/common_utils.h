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

#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <numeric>
#include <gtest/gtest.h>

#include "legate.h"
#include "cupynumeric.h"
#include "cupynumeric/runtime.h"
#include "util.inl"

namespace cupynumeric {

void debug_array(NDArray a, bool show_data = true);

template <typename T>
NDArray mk_array(std::vector<T> const& values, std::vector<uint64_t> shape = {})
{
  if (shape.empty() && values.size() > 1) {
    shape.push_back(values.size());
  }
  auto out = zeros(shape, legate::primitive_type(legate::type_code_of_v<T>));
  if (values.size() != out.size()) {
    throw std::invalid_argument("size and shape mismatch");
  }
  if (out.size() == 0) {
    return out;
  }
  if (out.size() == 1) {
    // must static cast here in case T = bool, in which case operator[] may return the stupid
    // "proxy bool" that std::vector<bool> is allowed to return. In that case the Scalar
    // constructor cannot deduce the type, because it has not been specialized for that type.
    out.fill(legate::Scalar(static_cast<const T&>(values[0])));
    return out;
  }
  auto assign_values = [](NDArray& a, std::vector<T> const& values) {
    auto acc = a.get_write_accessor<T, 1>();
    for (size_t i = 0; i < values.size(); ++i) {
      acc[i] = values[i];
    }
  };
  if (out.dim() == 1) {
    assign_values(out, values);
  } else {
    auto a1 = zeros({out.size()}, out.type());
    assign_values(a1, values);
    auto runtime = CuPyNumericRuntime::get_runtime();
    auto a2      = runtime->create_array(std::move(a1.get_store().delinearize(0, shape)));
    out.assign(a2);
  }
  return out;
}

template <typename T>
void check_and_wrap(NDArray& a, const std::vector<T>& values, std::vector<uint64_t>& shape)
{
  if (shape.empty() && values.size() > 1) {
    shape.push_back(values.size());
  }
  ASSERT_EQ(a.size(), values.size());
  ASSERT_EQ(a.shape(), shape);
  ASSERT_EQ(a.type().code(), legate::type_code_of_v<T>);

  if (a.dim() > 1) {
    a = a._wrap(a.size());
  }
}

template <typename T>
void check_array(NDArray a, const std::vector<T>& values, std::vector<uint64_t> shape = {})
{
  check_and_wrap<T>(a, values, shape);
  if (a.size() == 0) {
    return;
  }

  auto err_msg = [](auto i) {
    std::stringstream ss;
    ss << "check_array failed at [i = " << i << "]";
    return ss.str();
  };

  auto acc = a.get_read_accessor<T, 1>();
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(acc[i], values[i]) << err_msg(i);
  }
}

template <typename T>
void check_array_near(NDArray a,
                      const std::vector<T>& values,
                      std::vector<uint64_t> shape = {},
                      double abs_error            = 1.e-8)
{
  check_and_wrap<T>(a, values, shape);
  if (a.size() == 0) {
    return;
  }

  auto err_msg = [](auto i) {
    std::stringstream ss;
    ss << "check_array_near failed at [i = " << i << "]";
    return ss.str();
  };

  auto acc = a.get_read_accessor<T, 1>();
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_NEAR(acc[i], values[i], abs_error) << err_msg(i);
  }
}

template <typename T>
struct PrintArray {
  template <int32_t DIM>
  void operator()(cupynumeric::NDArray array)
  {
    auto acc            = array.get_read_accessor<T, DIM>();
    auto& shape         = array.shape();
    auto logical_store  = array.get_store();
    auto physical_store = logical_store.get_physical_store();
    auto rect           = physical_store.shape<DIM>();
    std::cerr << to_string<T, DIM>(acc, shape, rect) << std::endl;
  }
};

template <typename T>
void print_array(NDArray array)
{
  if (array.size() == 0) {
    std::cerr << "[]" << std::endl;
    return;
  }
  if (array.dim() == 0) {
    auto acc = array.get_read_accessor<T, 1>();
    std::cerr << "[" << acc[0] << "]" << std::endl;
    return;
  }
  legate::dim_dispatch(array.dim(), PrintArray<T>{}, array);
}

template <typename T>
void debug_vector(const std::vector<T>& vec)
{
  std::cerr << "[";
  for (auto i = vec.begin(); i != vec.end(); ++i) {
    std::cerr << *i;
    if (i != vec.end() - 1) {
      std::cerr << ", ";
    }
  }
  std::cerr << "]" << std::endl;
}

// x = a * i + b, i = 1, 2, 3, ...
template <typename T>
std::vector<T> mk_seq_vector(std::vector<uint64_t> shape, T a = 1, T b = 0)
{
  size_t size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
  std::vector<T> v(size);
  std::generate(v.begin(), v.end(), [a, x = b]() mutable { return x += a; });
  return v;
}

template <typename T, typename U>
std::vector<T> as_type_vector(std::vector<U> const& in)
{
  std::vector<T> out;
  for (auto elem : in) {
    out.push_back(static_cast<T>(elem));
  }
  return out;
}

template <typename T>
std::vector<T> to_vector(NDArray a)
{
  std::vector<T> result;
  if (a.size() == 0) {
    return result;
  }
  if (a.dim() > 1) {
    a = a._wrap(a.size());
  }
  auto acc = a.get_read_accessor<T, 1>();
  result.reserve(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result.push_back(acc[i]);
  }
  return result;
}

}  // namespace cupynumeric
