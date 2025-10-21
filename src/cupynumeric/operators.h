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

#include <memory>
#include <optional>

#include "legate.h"
#include "cupynumeric/ndarray.h"
#include "cupynumeric/typedefs.h"

namespace cupynumeric {

legate::Logger& cupynumeric_log();

void initialize(int32_t argc, char** argv);

NDArray array(std::vector<uint64_t> shape, const legate::Type& type);

NDArray abs(NDArray input);

NDArray add(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray multiply(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray divide(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray dot(NDArray a, NDArray b);

NDArray negative(NDArray input);

NDArray random(std::vector<uint64_t> shape);

NDArray zeros(std::vector<uint64_t> shape, std::optional<legate::Type> type = std::nullopt);

NDArray full(std::vector<uint64_t> shape, const Scalar& value);

NDArray all(NDArray input,
            std::vector<int32_t> axis    = {},
            std::optional<NDArray> out   = std::nullopt,
            bool keepdims                = false,
            std::optional<NDArray> where = std::nullopt);

NDArray sum(NDArray input);

NDArray amax(NDArray input,
             std::vector<int32_t> axis         = {},
             std::optional<legate::Type> dtype = std::nullopt,
             std::optional<NDArray> out        = std::nullopt,
             bool keepdims                     = false,
             std::optional<Scalar> initial     = std::nullopt,
             std::optional<NDArray> where      = std::nullopt);

NDArray amin(NDArray input,
             std::vector<int32_t> axis         = {},
             std::optional<legate::Type> dtype = std::nullopt,
             std::optional<NDArray> out        = std::nullopt,
             bool keepdims                     = false,
             std::optional<Scalar> initial     = std::nullopt,
             std::optional<NDArray> where      = std::nullopt);

NDArray unique(NDArray input);

NDArray swapaxes(NDArray input, int32_t axis1, int32_t axis2);

template <typename T>
NDArray arange(T start, std::optional<T> stop = std::nullopt, T step = 1);

NDArray arange(Scalar start, Scalar stop = legate::Scalar{}, Scalar step = legate::Scalar{});

NDArray as_array(legate::LogicalStore store);

NDArray array_equal(NDArray input0, NDArray input1);

std::vector<NDArray> nonzero(NDArray input);

NDArray eye(int32_t n,
            std::optional<int32_t> m = std::nullopt,
            int32_t k                = 0,
            const legate::Type& type = legate::float64());

NDArray tril(NDArray rhs, int32_t k = 0);

NDArray triu(NDArray rhs, int32_t k = 0);

NDArray bartlett(int64_t M);

NDArray blackman(int64_t M);

NDArray hamming(int64_t M);

NDArray hanning(int64_t M);

NDArray kaiser(int64_t M, double beta);

NDArray bincount(NDArray x, std::optional<NDArray> weights = std::nullopt, uint32_t min_length = 0);

NDArray convolve(NDArray a, NDArray v);

NDArray sort(NDArray input, std::optional<int32_t> axis = -1, std::string kind = "quicksort");

NDArray argsort(NDArray input, std::optional<int32_t> axis = -1, std::string kind = "quicksort");

NDArray sort_complex(NDArray input);

NDArray transpose(NDArray a);

NDArray transpose(NDArray a, std::vector<int32_t> axes);

NDArray moveaxis(NDArray a, std::vector<int32_t> source, std::vector<int32_t> destination);

NDArray argwhere(NDArray input);

NDArray flip(NDArray input, std::optional<std::vector<int32_t>> axis = std::nullopt);

void put(NDArray& a, NDArray indices, NDArray values, std::string mode = "raise");

NDArray repeat(NDArray a, NDArray repeats, std::optional<int32_t> axis = std::nullopt);

NDArray repeat(NDArray a, int64_t repeats, std::optional<int32_t> axis = std::nullopt);

// helper methods
int32_t normalize_axis_index(int32_t axis, int32_t ndim);

std::vector<int32_t> normalize_axis_vector(const std::vector<int32_t>& axis,
                                           int32_t ndim,
                                           bool allow_duplicate = false);

NDArray diag(NDArray v, int32_t k = 0);

NDArray diagonal(NDArray a,
                 int32_t offset               = 0,
                 std::optional<int32_t> axis1 = std::nullopt,
                 std::optional<int32_t> axis2 = std::nullopt,
                 std::optional<bool> extract  = std::nullopt);

NDArray trace(NDArray a,
              int32_t offset                   = 0,
              int32_t axis1                    = 0,
              int32_t axis2                    = 1,
              std::optional<legate::Type> type = std::nullopt,
              std::optional<NDArray> out       = std::nullopt);

NDArray reshape(NDArray a, std::vector<int64_t> newshape, std::string order = "C");

NDArray ravel(NDArray a, std::string order = "C");

NDArray squeeze(
  NDArray a, std::optional<std::reference_wrapper<std::vector<int32_t> const>> axis = std::nullopt);

std::vector<NDArray> where(NDArray a);

NDArray where(NDArray a, NDArray x, NDArray y);

// helper methods
legate::Type find_common_type(const std::vector<NDArray>& arrays);

template <typename T>
bool vec_is_equal(const std::vector<T>& vec1, const std::vector<T>& vec2)
{
  return vec1.size() == vec2.size() && std::equal(vec1.begin(), vec1.end(), vec2.begin());
}

template <typename T>
T vec_prod(const std::vector<T>& vec)
{
  T result = 1;
  for (const auto& elem : vec) {
    result *= elem;
  }
  return result;
}

template <typename T, typename U>
std::vector<U> vec_convert(const std::vector<T>& input)
{
  std::vector<U> output;
  output.reserve(input.size());
  for (const auto& elem : input) {
    output.push_back(static_cast<U>(elem));
  }
  return output;
}

template <typename T>
bool _is_in_vector(const std::vector<T>& vec, T item)
{
  if (std::find(vec.begin(), vec.end(), item) != vec.end()) {
    return true;
  } else {
    return false;
  }
}

std::vector<std::vector<char>> dot_modes(uint a_ndim, uint b_ndim);

template <typename T>
std::vector<T> merge_vectors(
  std::vector<T> a, std::vector<T> b, uint start_a, uint end_a, uint start_b, uint end_b);

}  // namespace cupynumeric
#include "cupynumeric/operators.inl"
