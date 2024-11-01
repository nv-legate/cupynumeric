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

#include "cunumeric/operators.h"

#include "cunumeric/runtime.h"
#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/unary/unary_op_util.h"
#include "cunumeric/unary/unary_red_util.h"
#include "cunumeric/random/rand_util.h"
#include "cunumeric/nullary/window_util.h"

namespace cunumeric {

static legate::Logger log_cunumeric("cunumeric");

legate::Logger& cunumeric_log() { return log_cunumeric; }

NDArray array(std::vector<uint64_t> shape, const legate::Type& type)
{
  return CuNumericRuntime::get_runtime()->create_array(std::move(shape), type);
}

NDArray unary_op(UnaryOpCode op_code,
                 NDArray input,
                 const std::vector<legate::Scalar>& extra_args = {})
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(input.shape(), input.type());
  out.unary_op(static_cast<int32_t>(op_code), std::move(input), extra_args);
  return out;
}

NDArray unary_reduction(UnaryRedCode op_code, NDArray input)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array({1}, input.type());
  out.unary_reduction(static_cast<int32_t>(op_code), std::move(input));
  return out;
}

NDArray binary_op(BinaryOpCode op_code, NDArray rhs1, NDArray rhs2, std::optional<NDArray> out)
{
  auto runtime = CuNumericRuntime::get_runtime();
  if (!out.has_value()) {
    auto out_shape = broadcast_shapes({rhs1, rhs2});
    out            = runtime->create_array(out_shape, rhs1.type());
  }
  out->binary_op(static_cast<int32_t>(op_code), std::move(rhs1), std::move(rhs2));
  return out.value();
}

NDArray abs(NDArray input) { return unary_op(UnaryOpCode::ABSOLUTE, std::move(input)); }

NDArray add(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out)
{
  return binary_op(BinaryOpCode::ADD, std::move(rhs1), std::move(rhs2), std::move(out));
}

NDArray angle(NDArray input, bool deg)
{
  const std::vector<Scalar> extra_args = {Scalar(deg)};

  return unary_op(UnaryOpCode::ANGLE, std::move(input), extra_args);
}

NDArray multiply(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out)
{
  return binary_op(BinaryOpCode::MULTIPLY, std::move(rhs1), std::move(rhs2), std::move(out));
}

NDArray negative(NDArray input) { return unary_op(UnaryOpCode::NEGATIVE, std::move(input)); }

NDArray random(std::vector<uint64_t> shape)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), legate::float64());
  out.random(static_cast<int32_t>(RandGenCode::UNIFORM));
  return out;
}

namespace {

struct generate_zero_fn {
  template <legate::Type::Code CODE>
  legate::Scalar operator()()
  {
    using VAL = legate::type_of<CODE>;
    return legate::Scalar(VAL(0));
  }
};

struct generate_arange_shape_fn {
  template <legate::Type::Code CODE,
            std::enable_if_t<legate::is_integral<CODE>::value ||
                             legate::is_floating_point<CODE>::value>* = nullptr>
  size_t operator()(Scalar& start, Scalar& stop, Scalar& step)
  {
    using VAL = legate::type_of<CODE>;
    if (stop.type().code() == legate::Type::Code::NIL) {
      stop  = start;
      start = legate::Scalar(VAL(0));
    }

    if (step.type().code() == legate::Type::Code::NIL) {
      step = legate::Scalar(VAL(1));
    }

    return static_cast<size_t>(ceil((stop.value<VAL>() - start.value<VAL>()) / step.value<VAL>()));
  }

  template <legate::Type::Code CODE,
            std::enable_if_t<!(legate::is_integral<CODE>::value ||
                               legate::is_floating_point<CODE>::value)>* = nullptr>
  size_t operator()(Scalar& start, Scalar& stop, Scalar& step)
  {
    throw std::invalid_argument("arange input should be integer or real");
  }
};

struct generate_int_value_fn {
  template <legate::Type::Code CODE, std::enable_if_t<legate::is_integral<CODE>::value>* = nullptr>
  int operator()(NDArray& array)
  {
    using VAL = legate::type_of<CODE>;
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

NDArray zeros(std::vector<uint64_t> shape, std::optional<legate::Type> type)
{
  auto code = type.has_value() ? type.value().code() : legate::Type::Code::FLOAT64;
  if (static_cast<int32_t>(code) >= static_cast<int32_t>(legate::Type::Code::FIXED_ARRAY)) {
    throw std::invalid_argument("Type must be a primitive type");
  }
  auto zero = legate::type_dispatch(code, generate_zero_fn{});
  return full(shape, zero);
}

NDArray full(std::vector<uint64_t> shape, const Scalar& value)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), value.type());
  out.fill(value);
  return out;
}

NDArray eye(int32_t n, std::optional<int32_t> m, int32_t k, const legate::Type& type)
{
  if (n < 0 || (m.has_value() && m.value() < 0)) {
    throw std::invalid_argument("eye input n and m should not less then zero");
  }

  if (!type.is_primitive()) {
    throw std::invalid_argument("Type must be a primitive type");
  }

  auto runtime = CuNumericRuntime::get_runtime();
  auto out =
    runtime->create_array({static_cast<size_t>(n), static_cast<size_t>(m.value_or(n))}, type);
  out.eye(k);
  return out;
}

NDArray bincount(NDArray x,
                 std::optional<NDArray> weights /*=std::nullopt*/,
                 uint32_t min_length /*=0*/)
{
  if (x.dim() != 1) {
    throw std::invalid_argument("The input array must be 1-dimensional");
  }
  if (x.size() == 0) {
    throw std::invalid_argument("The input array must be non-empty");
  }

  int32_t x_type_code = static_cast<int32_t>(x.type().code());
  if (x_type_code < static_cast<int32_t>(legate::Type::Code::INT8) ||
      x_type_code > static_cast<int32_t>(legate::Type::Code::UINT64)) {
    throw std::invalid_argument("input array for bincount must be integer type");
  }

  auto max_val_arr = amax(x);
  auto max_val =
    legate::type_dispatch(max_val_arr.type().code(), generate_int_value_fn{}, max_val_arr);
  auto min_val_arr = amin(x);
  auto min_val =
    legate::type_dispatch(min_val_arr.type().code(), generate_int_value_fn{}, min_val_arr);
  if (min_val < 0) {
    throw std::invalid_argument("the input array must have no negative elements");
  }
  if (static_cast<int32_t>(min_length) < max_val + 1) {
    min_length = max_val + 1;
  }

  auto runtime = CuNumericRuntime::get_runtime();
  if (!weights.has_value()) {
    auto out = runtime->create_array({min_length}, legate::int64());
    out.bincount(x);
    return out;
  } else {
    auto weight_array = weights.value();
    if (weight_array.shape() != x.shape()) {
      throw std::invalid_argument("weights array must have the same shape as the input array");
    }
    auto weight_code = weight_array.type().code();
    if (static_cast<int32_t>(weight_code) >= static_cast<int32_t>(legate::Type::Code::COMPLEX64)) {
      throw std::invalid_argument("weights must be convertible to float64");
    }
    if (weight_code != legate::Type::Code::FLOAT64) {
      weight_array = weight_array.as_type(legate::float64());
    }

    auto out = runtime->create_array({min_length}, weight_array.type());
    out.bincount(x, weight_array);
    return out;
  }
}

NDArray trilu(NDArray rhs, int32_t k, bool lower)
{
  auto dim    = rhs.dim();
  auto& shape = rhs.shape();
  std::vector<uint64_t> out_shape(shape);
  if (dim == 0) {
    throw std::invalid_argument("Dim of input array must be > 0");
  }
  if (dim == 1) {
    out_shape.emplace_back(shape[0]);
  }

  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(out_shape), rhs.type());
  out.trilu(std::move(rhs), k, lower);
  return out;
}

NDArray tril(NDArray rhs, int32_t k) { return trilu(rhs, k, true); }

NDArray triu(NDArray rhs, int32_t k) { return trilu(rhs, k, false); }

NDArray dot(NDArray rhs1, NDArray rhs2)
{
  if (rhs1.dim() != 2 || rhs2.dim() != 2) {
    LEGATE_ABORT("cunumeric::dot only supports matrices now");
  }

  auto& rhs1_shape = rhs1.shape();
  auto& rhs2_shape = rhs2.shape();

  if (rhs1_shape[1] != rhs2_shape[0]) {
    LEGATE_ABORT("Incompatible matrices: (",
                 rhs1_shape[0],
                 ", ",
                 rhs1_shape[1],
                 ") x (",
                 rhs2_shape[0],
                 ", ",
                 rhs2_shape[1],
                 ")");
  }

  auto runtime = CuNumericRuntime::get_runtime();
  std::vector<uint64_t> shape;
  shape.push_back(rhs1_shape[0]);
  shape.push_back(rhs2_shape[1]);

  auto out = runtime->create_array(std::move(shape), rhs1.type());
  out.dot(std::move(rhs1), std::move(rhs2));
  return out;
}

NDArray all(NDArray input,
            std::vector<int32_t> axis,
            std::optional<NDArray> out,
            bool keepdims,
            std::optional<NDArray> where)
{
  return input.all(axis, out, keepdims, std::nullopt, where);
}

NDArray sum(NDArray input) { return unary_reduction(UnaryRedCode::SUM, std::move(input)); }

NDArray amax(NDArray input,
             std::vector<int32_t> axis,
             std::optional<legate::Type> dtype,
             std::optional<NDArray> out,
             bool keepdims,
             std::optional<Scalar> initial,
             std::optional<NDArray> where)
{
  return input._perform_unary_reduction(static_cast<int32_t>(UnaryRedCode::MAX),
                                        input,
                                        axis,
                                        dtype,
                                        std::nullopt,
                                        out,
                                        keepdims,
                                        {},
                                        initial,
                                        where);
}

NDArray amin(NDArray input,
             std::vector<int32_t> axis,
             std::optional<legate::Type> dtype,
             std::optional<NDArray> out,
             bool keepdims,
             std::optional<Scalar> initial,
             std::optional<NDArray> where)
{
  return input._perform_unary_reduction(static_cast<int32_t>(UnaryRedCode::MIN),
                                        input,
                                        axis,
                                        dtype,
                                        std::nullopt,
                                        out,
                                        keepdims,
                                        {},
                                        initial,
                                        where);
}

NDArray unique(NDArray input) { return input.unique(); }

NDArray swapaxes(NDArray input, int32_t axis1, int32_t axis2)
{
  return input.swapaxes(axis1, axis2);
}

NDArray arange(Scalar start, Scalar stop, Scalar step)
{
  size_t N =
    legate::type_dispatch(start.type().code(), generate_arange_shape_fn{}, start, stop, step);

  if (start.type() != stop.type() || start.type() != step.type()) {
    throw std::invalid_argument("start/stop/step should be of the same type");
  }

  auto out = CuNumericRuntime::get_runtime()->create_array({N}, start.type());
  out.arange(start, stop, step);
  return out;
}

NDArray as_array(legate::LogicalStore store)
{
  return CuNumericRuntime::get_runtime()->create_array(std::move(store));
}

NDArray array_equal(NDArray input0, NDArray input1)
{
  auto dst = CuNumericRuntime::get_runtime()->create_array({1}, legate::bool_());

  if (input0.shape() != input1.shape()) {
    dst.fill(legate::Scalar(false));
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
    out.fill(one);
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
  if (a.dim() != v.dim()) {
    throw std::invalid_argument("Arrays should have the same dimensions");
  }
  if (a.dim() > 3) {
    throw std::runtime_error(std::to_string(a.dim()) + "-D arrays are not yet supported");
  }
  auto out = CuNumericRuntime::get_runtime()->create_array(a.shape(), a.type());
  if (a.type() != v.type()) {
    v = v.as_type(a.type());
  }
  out.convolve(std::move(a), std::move(v));
  return out;
}

NDArray sort(NDArray input, std::optional<int32_t> axis /*=-1*/, std::string kind /*="quicksort"*/)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto result  = runtime->create_array(input.shape(), input.type());
  result.sort(input, false, axis, kind);
  return result;
}

NDArray argsort(NDArray input,
                std::optional<int32_t> axis /*=-1*/,
                std::string kind /*="quicksort"*/)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto result  = runtime->create_array(input.shape(), legate::int64());
  result.sort(input, true, axis, kind);
  return result;
}

NDArray msort(NDArray input) { return sort(input, 0); }

NDArray sort_complex(NDArray input)
{
  auto result = sort(input);

  auto type = result.type();
  if (type == legate::complex64() || type == legate::complex128()) {
    return result;
  } else if (type == legate::int8() || type == legate::int16() || type == legate::uint8() ||
             type == legate::uint16()) {
    return result.as_type(legate::complex64());
  } else {
    return result.as_type(legate::complex128());
  }
}

NDArray transpose(NDArray a) { return a.transpose(); }

NDArray transpose(NDArray a, std::vector<int32_t> axes) { return a.transpose(axes); }

int32_t normalize_axis_index(int32_t axis, int32_t ndim)
{
  if (-ndim <= axis && axis < ndim) {
    axis = axis < 0 ? axis + ndim : axis;
  } else {
    std::stringstream ss;
    ss << "AxisError: axis " << axis << " is out of bounds for array of dimension " << ndim;
    throw std::invalid_argument(ss.str());
  }
  return axis;
}

std::vector<int32_t> normalize_axis_vector(std::vector<int32_t> axis,
                                           int32_t ndim,
                                           bool allow_duplicate)
{
  std::vector<int32_t> new_axis;
  for (auto ax : axis) {
    new_axis.emplace_back(normalize_axis_index(ax, ndim));
  }
  std::set<int32_t> s(new_axis.begin(), new_axis.end());
  if (!allow_duplicate && s.size() != new_axis.size()) {
    throw std::invalid_argument("repeated axis");
  }
  return new_axis;
}

NDArray moveaxis(NDArray a, std::vector<int32_t> source, std::vector<int32_t> destination)
{
  if (source.size() != destination.size()) {
    throw std::invalid_argument(
      "`source` and `destination` arguments must have the same number "
      "of elements");
  }
  auto ndim = a.dim();
  auto src  = normalize_axis_vector(source, ndim);
  auto dst  = normalize_axis_vector(destination, ndim);
  std::vector<int32_t> order;
  std::set<int32_t> set_src(src.begin(), src.end());
  for (auto i = 0; i < ndim; ++i) {
    if (set_src.find(i) == set_src.end()) {
      order.emplace_back(i);
    }
  }
  std::vector<std::pair<int32_t, int32_t>> vp;
  for (size_t i = 0; i < src.size(); ++i) {
    vp.push_back(std::make_pair(dst[i], src[i]));
  }
  std::sort(vp.begin(), vp.end());
  for (auto p : vp) {
    order.emplace(order.begin() + p.first, p.second);
  }
  return a.transpose(order);
}

NDArray argwhere(NDArray input) { return input.argwhere(); }

NDArray diag(NDArray v, int32_t k)
{
  int32_t dim = v.dim();
  switch (dim) {
    case 0: throw std::invalid_argument("Input must be 1- or 2-d");
    case 1: return v.diagonal(k, 0, 1, false);
    case 2: return v.diagonal(k, 0, 1, true);
    default: throw std::invalid_argument("diag requires 1- or 2-D array, use diagonal instead");
  }
}

NDArray diagonal(NDArray a,
                 int32_t offset,
                 std::optional<int32_t> axis1,
                 std::optional<int32_t> axis2,
                 std::optional<bool> extract)
{
  return a.diagonal(offset, axis1, axis2, extract);
}

NDArray flip(NDArray input, std::optional<std::vector<int32_t>> axis) { return input.flip(axis); }

void put(NDArray& a, NDArray indices, NDArray values, std::string mode)
{
  a.put(indices, values, mode);
}

NDArray trace(NDArray a,
              int32_t offset,
              int32_t axis1,
              int32_t axis2,
              std::optional<legate::Type> type,
              std::optional<NDArray> out)
{
  return a.trace(offset, axis1, axis2, type, out);
}

NDArray repeat(NDArray a, NDArray repeats, std::optional<int32_t> axis)
{
  return a.repeat(repeats, axis);
}

NDArray repeat(NDArray a, int64_t repeats, std::optional<int32_t> axis)
{
  return a.repeat(repeats, axis);
}

NDArray reshape(NDArray a, std::vector<int64_t> newshape, std::string order)
{
  return a.reshape(newshape, order);
}

NDArray ravel(NDArray a, std::string order) { return a.ravel(order); }

}  // namespace cunumeric
