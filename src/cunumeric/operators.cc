/* Copyright 2021 NVIDIA Corporation
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

#include "cunumeric/operators.h"

#include "cunumeric/runtime.h"
#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/unary/unary_op_util.h"
#include "cunumeric/random/rand_util.h"
#include "cunumeric/nullary/window_util.h"

namespace cunumeric {

NDArray array(std::vector<size_t> shape, const legate::Type& type)
{
  return CuNumericRuntime::get_runtime()->create_array(std::move(shape), type);
}

NDArray unary_op(UnaryOpCode op_code, NDArray input)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(input.shape(), input.type());
  out.unary_op(static_cast<int32_t>(op_code), std::move(input));
  return std::move(out);
}

NDArray unary_reduction(UnaryRedCode op_code, NDArray input)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array({1}, input.type());
  out.unary_reduction(static_cast<int32_t>(op_code), std::move(input));
  return std::move(out);
}

NDArray binary_op(BinaryOpCode op_code, NDArray rhs1, NDArray rhs2, std::optional<NDArray> out)
{
  auto runtime = CuNumericRuntime::get_runtime();
  if (!out.has_value()) {
    auto out_shape = broadcast_shapes({rhs1, rhs2});
    out            = runtime->create_array(out_shape, rhs1.type());
  }
  out->binary_op(static_cast<int32_t>(op_code), std::move(rhs1), std::move(rhs2));
  return std::move(out.value());
}

NDArray abs(NDArray input) { return unary_op(UnaryOpCode::ABSOLUTE, std::move(input)); }

NDArray add(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out)
{
  return binary_op(BinaryOpCode::ADD, std::move(rhs1), std::move(rhs2), std::move(out));
}

NDArray multiply(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out)
{
  return binary_op(BinaryOpCode::MULTIPLY, std::move(rhs1), std::move(rhs2), std::move(out));
}

NDArray negative(NDArray input) { return unary_op(UnaryOpCode::NEGATIVE, std::move(input)); }

NDArray random(std::vector<size_t> shape)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), legate::float64());
  out.random(static_cast<int32_t>(RandGenCode::UNIFORM));
  return std::move(out);
}

namespace {

struct generate_zero_fn {
  template <legate::Type::Code CODE>
  legate::Scalar operator()()
  {
    using VAL = legate::legate_type_of<CODE>;
    return legate::Scalar(VAL(0));
  }
};

struct generate_int_value_fn {
  template <legate::Type::Code CODE, std::enable_if_t<legate::is_integral<CODE>::value>* = nullptr>
  int operator()(NDArray& array)
  {
    using VAL = legate::legate_type_of<CODE>;
    return static_cast<int>(array.get_read_accessor<VAL, 1>()[0]);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_integral<CODE>::value>* = nullptr>
  int operator()(NDArray& array)
  {
    assert(false);
    return -1;
  }
};

}  // namespace

NDArray zeros(std::vector<size_t> shape, std::optional<legate::Type> type)
{
  auto code = type.has_value() ? type.value().code() : legate::Type::Code::FLOAT64;
  if (static_cast<int32_t>(code) >= static_cast<int32_t>(legate::Type::Code::FIXED_ARRAY))
    throw std::invalid_argument("Type must be a primitive type");
  auto zero = legate::type_dispatch(code, generate_zero_fn{});
  return full(shape, zero);
}

NDArray full(std::vector<size_t> shape, const Scalar& value)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), value.type());
  out.fill(value, false);
  return std::move(out);
}

NDArray eye(size_t n, std::optional<size_t> m, int32_t k, const legate::Type& type)
{
  if (static_cast<int32_t>(type.code()) >= static_cast<int32_t>(legate::Type::Code::FIXED_ARRAY))
    throw std::invalid_argument("Type must be a primitive type");

  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array({n, m.value_or(n)}, type);
  out.eye(k);
  return std::move(out);
}

NDArray bincount(NDArray x,
                 std::optional<NDArray> weights /*=std::nullopt*/,
                 uint32_t min_length /*=0*/)
{
  if (x.dim() != 1) throw std::invalid_argument("The input array must be 1-dimensional");
  if (x.size() == 0) throw std::invalid_argument("The input array must be non-empty");

  int32_t x_type_code = static_cast<int32_t>(x.type().code());
  if (x_type_code < static_cast<int32_t>(legate::Type::Code::INT8) ||
      x_type_code > static_cast<int32_t>(legate::Type::Code::UINT64))
    throw std::invalid_argument("input array for bincount must be integer type");

  auto max_val_arr = amax(x);
  auto max_val =
    legate::type_dispatch(max_val_arr.type().code(), generate_int_value_fn{}, max_val_arr);
  auto min_val_arr = amin(x);
  auto min_val =
    legate::type_dispatch(min_val_arr.type().code(), generate_int_value_fn{}, min_val_arr);
  if (min_val < 0) throw std::invalid_argument("the input array must have no negative elements");
  if (min_length < max_val + 1) min_length = max_val + 1;

  auto runtime = CuNumericRuntime::get_runtime();
  if (!weights.has_value()) {
    auto out = runtime->create_array({min_length}, legate::int64());
    out.bincount(x);
    return out;
  } else {
    auto weight_array = weights.value();
    if (weight_array.shape() != x.shape())
      throw std::invalid_argument("weights array must have the same shape as the input array");
    auto weight_code = weight_array.type().code();
    if (static_cast<int32_t>(weight_code) >= static_cast<int32_t>(legate::Type::Code::COMPLEX64))
      throw std::invalid_argument("weights must be convertible to float64");
    if (weight_code != legate::Type::Code::FLOAT64)
      weight_array = weight_array.as_type(legate::float64());

    auto out = runtime->create_array({min_length}, weight_array.type());
    out.bincount(x, weight_array);
    return out;
  }
}

NDArray trilu(NDArray rhs, int32_t k, bool lower)
{
  auto dim    = rhs.dim();
  auto& shape = rhs.shape();
  std::vector<size_t> out_shape(shape);
  if (dim == 0) throw std::invalid_argument("Dim of input array must be > 0");
  if (dim == 1) out_shape.emplace_back(shape[0]);

  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(out_shape), rhs.type());
  out.trilu(std::move(rhs), k, lower);
  return std::move(out);
}

NDArray tril(NDArray rhs, int32_t k) { return trilu(rhs, k, true); }

NDArray triu(NDArray rhs, int32_t k) { return trilu(rhs, k, false); }

NDArray dot(NDArray rhs1, NDArray rhs2)
{
  if (rhs1.dim() != 2 || rhs2.dim() != 2) {
    fprintf(stderr, "cunumeric::dot only supports matrices now");
    LEGATE_ABORT;
  }

  auto& rhs1_shape = rhs1.shape();
  auto& rhs2_shape = rhs2.shape();

  if (rhs1_shape[1] != rhs2_shape[0]) {
    fprintf(stderr,
            "Incompatible matrices: (%zd, %zd) x (%zd, %zd)\n",
            rhs1_shape[0],
            rhs1_shape[1],
            rhs2_shape[0],
            rhs2_shape[1]);
    LEGATE_ABORT;
  }

  auto runtime = CuNumericRuntime::get_runtime();
  std::vector<size_t> shape;
  shape.push_back(rhs1_shape[0]);
  shape.push_back(rhs2_shape[1]);

  auto out = runtime->create_array(std::move(shape), rhs1.type());
  out.dot(std::move(rhs1), std::move(rhs2));
  return std::move(out);
}

NDArray sum(NDArray input) { return unary_reduction(UnaryRedCode::SUM, std::move(input)); }

NDArray amax(NDArray input) { return unary_reduction(UnaryRedCode::MAX, std::move(input)); }

NDArray amin(NDArray input) { return unary_reduction(UnaryRedCode::MIN, std::move(input)); }

NDArray unique(NDArray input) { return input.unique(); }

NDArray arange(std::optional<double> start,
               std::optional<double> stop,
               std::optional<double> step,
               const legate::Type& type)
{
  if (!stop.has_value()) {
    stop  = start;
    start = 0;
  }

  size_t N = ceil((stop.value() - start.value()) / step.value());
  auto out = CuNumericRuntime::get_runtime()->create_array({N}, type);
  out.arange(start.value(), stop.value(), step.value());
  return std::move(out);
}

NDArray as_array(legate::LogicalStore store)
{
  return CuNumericRuntime::get_runtime()->create_array(std::move(store));
}

NDArray array_equal(NDArray input0, NDArray input1)
{
  auto dst = CuNumericRuntime::get_runtime()->create_array({1}, legate::bool_());

  if (input0.shape() != input1.shape()) {
    dst.fill(legate::Scalar(false), false);
  } else {
    dst.binary_reduction(static_cast<int32_t>(BinaryOpCode::EQUAL), input0, input1);
  }
  return dst;
}

std::vector<NDArray> nonzero(NDArray input) { return input.nonzero(); }

// window functions
NDArray create_window(int64_t M, WindowOpCode op_code, std::vector<double> args)
{
  auto type    = legate::float64();
  auto runtime = CuNumericRuntime::get_runtime();
  if (M <= 0) {
    return runtime->create_array({0}, std::move(type));
  } else if (M == 1) {
    auto out = runtime->create_array({1}, std::move(type));
    auto one = legate::Scalar(static_cast<double>(1));
    out.fill(one, false);
    return out;
  }
  auto out = runtime->create_array({static_cast<size_t>(M)}, std::move(type));
  out.create_window(static_cast<int32_t>(op_code), M, args);
  return out;
}

NDArray bartlett(int64_t M) { return create_window(M, WindowOpCode::BARLETT, {}); }

NDArray blackman(int64_t M) { return create_window(M, WindowOpCode::BLACKMAN, {}); }

NDArray hamming(int64_t M) { return create_window(M, WindowOpCode::HAMMING, {}); }

NDArray hanning(int64_t M) { return create_window(M, WindowOpCode::HANNING, {}); }

NDArray kaiser(int64_t M, double beta) { return create_window(M, WindowOpCode::KAISER, {beta}); }

NDArray convolve(NDArray a, NDArray v)
{
  if (a.dim() != v.dim()) { throw std::invalid_argument("Arrays should have the same dimensions"); }
  if (a.dim() > 3) {
    throw std::runtime_error(std::to_string(a.dim()) + "-D arrays are not yet supported");
  }
  auto out = CuNumericRuntime::get_runtime()->create_array(a.shape(), a.type());
  if (a.type() != v.type()) { v = v.as_type(a.type()); }
  out.convolve(std::move(a), std::move(v));
  return out;
}

}  // namespace cunumeric
