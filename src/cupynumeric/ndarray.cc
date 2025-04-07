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

#include "cupynumeric/ndarray.h"
#include <stdexcept>
#include <sys/types.h>

#include "cupynumeric/binary/binary_op_util.h"
#include "cupynumeric/operators.h"
#include "cupynumeric/random/rand_util.h"
#include "cupynumeric/runtime.h"
#include "cupynumeric/unary/convert_util.h"
#include "cupynumeric/unary/unary_op_util.h"
#include "cupynumeric/unary/unary_red_util.h"

namespace cupynumeric {

// ==========================================================================================

// Reduction utilities

namespace {

struct generate_zero_fn {
  template <legate::Type::Code CODE>
  legate::Scalar operator()()
  {
    using VAL = legate::type_of<CODE>;
    return legate::Scalar(VAL(0));
  }
};

struct check_nonzero_scalar_fn {
  template <legate::Type::Code CODE>
  bool operator()(cupynumeric::NDArray array)
  {
    assert(array.dim() == 0);
    using VAL = legate::type_of<CODE>;
    auto acc  = array.get_read_accessor<VAL, 1>();
    if (acc[0] == VAL(0)) {
      return false;
    } else {
      return true;
    }
  }
};

struct get_typesize_fn {
  template <legate::Type::Code CODE>
  uint64_t operator()()
  {
    using VAL = legate::type_of<CODE>;
    return uint64_t(sizeof(VAL));
  }
};

struct generate_identity_fn {
  template <UnaryRedCode OP>
  struct generator {
    template <legate::Type::Code CODE,
              std::enable_if_t<UnaryRedOp<OP, CODE>::valid && !is_arg_reduce<OP>::value>* = nullptr>
    Scalar operator()(const legate::Type&)
    {
      auto value = UnaryRedOp<OP, CODE>::OP::identity;
      return Scalar(value);
    }

    template <legate::Type::Code CODE,
              std::enable_if_t<UnaryRedOp<OP, CODE>::valid && is_arg_reduce<OP>::value>* = nullptr>
    Scalar operator()(const legate::Type& type)
    {
      auto value       = UnaryRedOp<OP, CODE>::OP::identity;
      auto argred_type = CuPyNumericRuntime::get_runtime()->get_argred_type(type);
      return Scalar(value, argred_type);
    }

    template <legate::Type::Code CODE, std::enable_if_t<!UnaryRedOp<OP, CODE>::valid>* = nullptr>
    Scalar operator()(const legate::Type&)
    {
      assert(false);
      return Scalar(0);
    }
  };

  template <UnaryRedCode OP>
  Scalar operator()(const legate::Type& type)
  {
    return legate::type_dispatch(type.code(), generator<OP>{}, type);
  }
};

Scalar get_reduction_identity(UnaryRedCode op, const legate::Type& type)
{
  static std::map<std::pair<UnaryRedCode, legate::Type::Code>, Scalar> identities;

  auto key    = std::make_pair(op, type.code());
  auto finder = identities.find(key);
  if (identities.end() != finder) {
    return finder->second;
  }

  auto identity = op_dispatch(op, generate_identity_fn{}, type);
  identities.insert({key, identity});
  return identity;
}

const std::unordered_map<UnaryRedCode, legate::ReductionOpKind> TO_CORE_REDOP = {
  {UnaryRedCode::ALL, legate::ReductionOpKind::MUL},
  {UnaryRedCode::ANY, legate::ReductionOpKind::ADD},
  {UnaryRedCode::ARGMAX, legate::ReductionOpKind::MAX},
  {UnaryRedCode::ARGMIN, legate::ReductionOpKind::MIN},
  {UnaryRedCode::CONTAINS, legate::ReductionOpKind::ADD},
  {UnaryRedCode::COUNT_NONZERO, legate::ReductionOpKind::ADD},
  {UnaryRedCode::MAX, legate::ReductionOpKind::MAX},
  {UnaryRedCode::MIN, legate::ReductionOpKind::MIN},
  {UnaryRedCode::PROD, legate::ReductionOpKind::MUL},
  {UnaryRedCode::SUM, legate::ReductionOpKind::ADD},
};

legate::ReductionOpKind get_reduction_op(UnaryRedCode op) { return TO_CORE_REDOP.at(op); }

}  // namespace

// ==========================================================================================

NDArray::NDArray(legate::LogicalStore&& store) : store_(std::forward<legate::LogicalStore>(store))
{
}

int32_t NDArray::dim() const { return store_.dim(); }

const std::vector<uint64_t>& NDArray::shape() const { return store_.extents().data(); }

size_t NDArray::size() const { return store_.volume(); }

legate::Type NDArray::type() const { return store_.type(); }

static std::vector<int64_t> compute_strides(const std::vector<uint64_t>& shape)
{
  std::vector<int64_t> strides(shape.size());
  if (shape.size() > 0) {
    int64_t stride = 1;
    for (int32_t dim = shape.size() - 1; dim >= 0; --dim) {
      strides[dim] = stride;
      stride *= shape[dim];
    }
  }
  return strides;
}

NDArray NDArray::operator+(const NDArray& other) const { return add(*this, other); }

NDArray NDArray::operator+(const legate::Scalar& other) const
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  return operator+(NDArray(std::move(scalar)));
}

NDArray& NDArray::operator+=(const NDArray& other)
{
  add(*this, other, *this);
  return *this;
}

NDArray NDArray::operator*(const NDArray& other) const { return multiply(*this, other); }

NDArray NDArray::operator*(const legate::Scalar& other) const
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  return operator*(NDArray(std::move(scalar)));
}

NDArray& NDArray::operator*=(const NDArray& other)
{
  multiply(*this, other, *this);
  return *this;
}

NDArray NDArray::operator/(const NDArray& other) const { return divide(*this, other); }

NDArray NDArray::operator/(const legate::Scalar& other) const
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  return operator/(NDArray(std::move(scalar)));
}

NDArray NDArray::operator[](std::initializer_list<slice> slices) const
{
  if (slices.size() > static_cast<size_t>(dim())) {
    std::string err_msg = "Can't slice a " + std::to_string(dim()) + "-D ndarray with " +
                          std::to_string(slices.size()) + " slices";
    throw std::invalid_argument(std::move(err_msg));
  }

  uint32_t dim = 0;
  auto sliced  = store_;
  for (const auto& sl : slices) {
    sliced = sliced.slice(dim, sl);
    ++dim;
  }

  return NDArray(std::move(sliced));
}

NDArray::operator bool() const { return ((NDArray*)this)->get_read_accessor<bool, 1>()[0]; }

void NDArray::assign(const NDArray& other)
{
  unary_op(static_cast<int32_t>(UnaryOpCode::COPY), other);
}

void NDArray::assign(const legate::Scalar& other)
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  assign(NDArray(std::move(scalar)));
}

void NDArray::random(int32_t gen_code)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_RAND);

  task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  task.add_scalar_arg(legate::Scalar(runtime->get_next_random_epoch()));
  auto strides = compute_strides(shape());
  task.add_scalar_arg(legate::Scalar(strides));

  runtime->submit(std::move(task));
}

void NDArray::fill(const Scalar& value)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  if (!store_.transformed()) {
    legate::Runtime::get_runtime()->issue_fill(store_, value);
    return;
  }

  auto fill_value = runtime->create_scalar_store(value);

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_FILL);

  task.add_output(store_);
  task.add_input(fill_value);

  runtime->submit(std::move(task));
}

void NDArray::_fill(legate::LogicalStore const& value)
{
  if (value.volume() != 1 || value.dim() > 1) {
    throw std::invalid_argument("Filled value size is not equal to 1");
  }
  if (type() != value.type()) {
    throw std::invalid_argument("Filled value type mismatch");
  }
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();
  auto task    = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_FILL);
  task.add_output(store_);
  task.add_input(value);
  task.add_scalar_arg(Scalar(false));
  runtime->submit(std::move(task));
}

void NDArray::eye(int32_t k)
{
  if (size() == 0) {
    return;
  }

  assert(dim() == 2);

  auto zero = legate::type_dispatch(type().code(), generate_zero_fn{});
  fill(zero);

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_EYE);

  task.add_input(store_);
  task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(k));

  runtime->submit(std::move(task));
}

void NDArray::bincount(NDArray rhs, std::optional<NDArray> weights /*=std::nullopt*/)
{
  if (size() == 0) {
    return;
  }

  assert(dim() == 1);

  auto runtime = CuPyNumericRuntime::get_runtime();

  if (weights.has_value()) {
    assert(rhs.shape() == weights.value().shape());
  }

  auto zero = legate::type_dispatch(type().code(), generate_zero_fn{});
  fill(zero);

  auto task                     = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_BINCOUNT);
  legate::ReductionOpKind redop = legate::ReductionOpKind::ADD;

  auto p_lhs = task.add_reduction(store_, redop);
  auto p_rhs = task.add_input(rhs.store_);
  task.add_constraint(legate::broadcast(p_lhs, {0}));
  if (weights.has_value()) {
    auto p_weight = task.add_input(weights.value().store_);
    task.add_constraint(legate::align(p_rhs, p_weight));
  }

  runtime->submit(std::move(task));
}

void NDArray::sort_task(NDArray rhs, bool argsort, bool stable)
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto task    = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_SORT);
  auto p_rhs   = task.add_input(rhs.store_);

  auto machine             = legate::Runtime::get_runtime()->get_machine();
  bool uses_unbound_output = machine.count() > 1 and rhs.dim() == 1;
  std::optional<NDArray> unbound;
  if (uses_unbound_output) {
    unbound = runtime->create_array(type());
    task.add_output(unbound.value().get_store());
  } else {
    auto p_lhs = task.add_output(store_);
    task.add_constraint(align(p_lhs, p_rhs));
  }

  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    task.add_communicator("nccl");
  } else {
    task.add_communicator("cpu");
  }

  task.add_scalar_arg(legate::Scalar(argsort));
  task.add_scalar_arg(legate::Scalar(rhs.shape()));
  task.add_scalar_arg(legate::Scalar(stable));
  runtime->submit(std::move(task));
  if (uses_unbound_output) {
    store_ = unbound.value().get_store();
  }
}

void NDArray::sort_swapped(NDArray rhs, bool argsort, int32_t sort_axis, bool stable)
{
  sort_axis = normalize_axis_index(sort_axis, rhs.dim());

  auto swapped      = rhs.swapaxes(sort_axis, rhs.dim() - 1);
  auto runtime      = CuPyNumericRuntime::get_runtime();
  auto swapped_copy = runtime->create_array(swapped.shape(), swapped.type());
  swapped_copy.assign(swapped);

  if (argsort) {
    auto sort_result = runtime->create_array(swapped_copy.shape(), type());
    sort_result.sort(swapped_copy, argsort, -1, stable);
    store_ = sort_result.swapaxes(rhs.dim() - 1, sort_axis).get_store();
  } else {
    swapped_copy.sort(swapped_copy, argsort, -1, stable);
    store_ = swapped_copy.swapaxes(rhs.dim() - 1, sort_axis).get_store();
  }
}

void NDArray::sort(NDArray rhs, bool argsort, std::optional<int32_t> axis, bool stable)
{
  if (!axis.has_value() && rhs.dim() > 1) {
    // TODO: need to flatten the rhs and sort it, the implementation depends on reshape method.
    assert(false);
  }
  int32_t computed_axis = 0;
  if (axis.has_value()) {
    computed_axis = normalize_axis_index(axis.value(), rhs.dim());
  }

  if (computed_axis == rhs.dim() - 1) {
    sort_task(rhs, argsort, stable);
  } else {
    sort_swapped(rhs, argsort, computed_axis, stable);
  }
}

void NDArray::sort(NDArray rhs,
                   bool argsort /*=false*/,
                   std::optional<int32_t> axis /*=-1*/,
                   std::string kind /*="quicksort"*/)
{
  if (axis.has_value() && (axis >= rhs.dim() || axis < -rhs.dim())) {
    throw std::invalid_argument("invalid axis");
  }

  if (!(kind == "quicksort" || kind == "mergesort" || kind == "heapsort" || kind == "stable")) {
    throw std::invalid_argument("invalid kind");
  }

  bool stable = (kind == "stable");
  sort(rhs, argsort, axis, stable);
}

void NDArray::trilu(NDArray rhs, int32_t k, bool lower)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_TRILU);

  auto& out_shape = shape();
  rhs             = rhs.broadcast(out_shape, rhs.store_);

  task.add_scalar_arg(legate::Scalar(lower));
  task.add_scalar_arg(legate::Scalar(k));

  auto p_lhs = task.add_output(store_);
  auto p_rhs = task.add_input(rhs.store_);

  task.add_constraint(align(p_lhs, p_rhs));

  runtime->submit(std::move(task));
}

void NDArray::dot(NDArray rhs1, NDArray rhs2) { dot_MM(rhs1.get_store(), rhs2.get_store()); }

void NDArray::binary_op(int32_t op_code, NDArray rhs1, NDArray rhs2)
{
  if (rhs1.type() != rhs2.type()) {
    throw std::invalid_argument("Operands must have the same type");
  }

  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_BINARY_OP);

  auto& out_shape = shape();
  auto rhs1_store = broadcast(out_shape, rhs1.store_);
  auto rhs2_store = broadcast(out_shape, rhs2.store_);

  auto p_lhs  = task.add_output(store_);
  auto p_rhs1 = task.add_input(rhs1_store);
  auto p_rhs2 = task.add_input(rhs2_store);
  task.add_scalar_arg(legate::Scalar(op_code));

  task.add_constraint(align(p_lhs, p_rhs1));
  task.add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void NDArray::binary_reduction(int32_t op_code, NDArray rhs1, NDArray rhs2)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto rhs1_store = broadcast(rhs1, rhs2);
  auto rhs2_store = broadcast(rhs2, rhs1);

  legate::ReductionOpKind redop;
  if (op_code == static_cast<int32_t>(BinaryOpCode::NOT_EQUAL)) {
    redop = get_reduction_op(UnaryRedCode::SUM);
    fill(legate::Scalar(false));
  } else {
    redop = get_reduction_op(UnaryRedCode::PROD);
    fill(legate::Scalar(true));
  }
  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_BINARY_RED);

  task.add_reduction(store_, redop);
  auto p_rhs1 = task.add_input(rhs1_store);
  auto p_rhs2 = task.add_input(rhs2_store);
  task.add_scalar_arg(legate::Scalar(op_code));

  task.add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void NDArray::unary_op(int32_t op_code,
                       NDArray input,
                       const std::vector<legate::Scalar>& extra_args /*= {}*/)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_UNARY_OP);

  auto rhs = broadcast(shape(), input.store_);

  auto p_out = task.add_output(store_);
  auto p_in  = task.add_input(rhs);
  task.add_scalar_arg(legate::Scalar(op_code));

  for (auto&& arg : extra_args) {
    task.add_scalar_arg(arg);
  }

  task.add_constraint(align(p_out, p_in));

  runtime->submit(std::move(task));
}

void NDArray::unary_reduction(int32_t op_code_, NDArray input)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto op_code = static_cast<UnaryRedCode>(op_code_);

  fill(get_reduction_identity(op_code, type()));

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_SCALAR_UNARY_RED);

  task.add_reduction(store_, get_reduction_op(op_code));
  task.add_input(input.store_);
  task.add_scalar_arg(legate::Scalar(op_code_));
  task.add_scalar_arg(legate::Scalar(input.shape()));
  task.add_scalar_arg(legate::Scalar(false));  // has_where

  runtime->submit(std::move(task));
}

uint64_t ceildiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

void NDArray::dot_MM(const legate::LogicalStore& rhs1_store, const legate::LogicalStore& rhs2_store)
{
  auto runtime         = CuPyNumericRuntime::get_runtime();
  const auto num_procs = legate::Runtime::get_runtime()->get_machine().count();
  bool is_single_proc  = num_procs == 1;

  fill(get_reduction_identity(UnaryRedCode::SUM, type()));

  assert(dim() == 2 && rhs1_store.dim() == 2 && rhs2_store.dim() == 2);

  auto m = rhs1_store.shape()[0];
  auto n = rhs2_store.shape()[1];
  auto k = rhs1_store.shape()[1];

  static constexpr std::size_t MIN_MATRIX_SIZE = 1 << 20;

  auto get_color_shape =
    [&](const std::vector<std::uint64_t>& shape) -> std::vector<std::uint64_t> {
    if (!is_in_test_mode() && shape[0] * shape[1] <= MIN_MATRIX_SIZE) {
      return {1, 1};
    }
    auto color_shape = std::vector<std::uint64_t>{num_procs, 1};

    while ((shape[0] / color_shape[0] < 2 * shape[1] / color_shape[1]) && color_shape[0] % 2 == 0) {
      color_shape[0] /= 2;
      color_shape[1] *= 2;
    }

    return color_shape;
  };

  auto get_batchsize = [&](const std::vector<std::uint64_t>& tilesize, std::uint64_t k) {
    uint64_t typesize = legate::type_dispatch(type().code(), get_typesize_fn{});
    // default corresponds to 128MB (to store A and B tile)
    uint64_t max_elements_per_tile = cupynumeric_matmul_cache_size() / typesize;
    uint64_t total_elements_rhs    = (tilesize[0] + tilesize[1]) * k;
    uint64_t num_batches           = ceildiv(total_elements_rhs, max_elements_per_tile);
    uint64_t batch_size            = ceildiv(k, num_batches);
    return batch_size;
  };

  auto initial_color_shape = get_color_shape({m, n});
  auto tile_shape          = std::vector<std::uint64_t>{ceildiv(m, initial_color_shape[0]),
                                                        ceildiv(n, initial_color_shape[1])};
  auto color_shape =
    legate::tuple<std::uint64_t>{ceildiv(m, tile_shape[0]), ceildiv(n, tile_shape[1])};

  std::uint64_t k_batch_size = is_single_proc ? k : get_batchsize(tile_shape, k);

  std::vector<std::uint64_t> tile_shape_rhs1 = {tile_shape[0], k_batch_size};
  std::vector<std::uint64_t> tile_shape_rhs2 = {k_batch_size, tile_shape[1]};
  auto color_k                               = ceildiv(k, k_batch_size);

  auto p_lhs  = store_.partition_by_tiling(tile_shape);
  auto p_rhs1 = rhs1_store.partition_by_tiling(tile_shape_rhs1);
  auto p_rhs2 = rhs2_store.partition_by_tiling(tile_shape_rhs2);

  for (std::uint64_t i = 0; i < color_k; ++i) {
    auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_MATMUL, color_shape);
    task.add_output(p_lhs);
    task.add_input(p_lhs);
    task.add_input(p_rhs1, legate::SymbolicPoint{legate::dimension(0), legate::constant(i)});
    task.add_input(p_rhs2, legate::SymbolicPoint{legate::constant(i), legate::dimension(1)});
    runtime->submit(std::move(task));
  }
}

void NDArray::arange(Scalar start, Scalar stop, Scalar step)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  if (start.type() != type() || stop.type() != type() || step.type() != type()) {
    throw std::invalid_argument("start/stop/step should have the same type as the array");
  }

  assert(dim() == 1);

  // TODO: Optimization when value is a scalar

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_ARANGE);

  task.add_output(store_);

  task.add_scalar_arg(start);
  task.add_scalar_arg(step);

  runtime->submit(std::move(task));
}

std::vector<NDArray> NDArray::nonzero()
{
  auto runtime = CuPyNumericRuntime::get_runtime();

  std::vector<NDArray> outputs;
  auto ndim = dim();
  for (int32_t i = 0; i < ndim; ++i) {
    outputs.emplace_back(runtime->create_array(legate::int64()));
  }

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_NONZERO);

  for (auto& output : outputs) {
    task.add_output(output.store_);
  }
  auto p_rhs = task.add_input(store_);

  if (ndim > 1) {
    task.add_constraint(legate::broadcast(p_rhs, legate::from_range<uint32_t>(1, ndim)));
  }

  runtime->submit(std::move(task));

  return outputs;
}

NDArray NDArray::unique()
{
  auto machine  = legate::Runtime::get_runtime()->get_machine();
  bool has_gpus = machine.count(legate::mapping::TaskTarget::GPU) > 0;

  auto runtime = CuPyNumericRuntime::get_runtime();
  auto result  = runtime->create_array(type());

  auto task     = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_UNIQUE);
  auto part_out = task.declare_partition();
  auto part_in  = task.declare_partition();
  task.add_output(result.store_, part_out);
  task.add_input(store_, part_in);

  if (has_gpus) {
    task.add_communicator("nccl");
  } else {
    task.add_constraint(legate::broadcast(part_in, legate::from_range<uint32_t>(0, dim())));
  }

  runtime->submit(std::move(task));
  return result;
}

NDArray NDArray::swapaxes(int32_t axis1, int32_t axis2)
{
  axis1 = normalize_axis_index(axis1, dim());
  axis2 = normalize_axis_index(axis2, dim());

  if (shape().size() == 1 || axis1 == axis2) {
    return *this;
  }

  auto ndim = dim();
  std::vector<int32_t> dims;
  for (auto i = 0; i < ndim; ++i) {
    dims.push_back(i);
  }

  if (axis1 < 0 || axis2 < 0) {
    throw std::out_of_range("Index is out of range");
  }

  std::swap(dims[axis1], dims[axis2]);

  auto transposed = store_.transpose(std::move(dims));
  auto runtime    = CuPyNumericRuntime::get_runtime();
  return runtime->create_array(std::move(transposed));
}

NDArray NDArray::as_type(const legate::Type& type) const
{
  auto runtime = CuPyNumericRuntime::get_runtime();

  // TODO: Check if conversion is valid

  auto out = runtime->create_array(shape(), type);

  if (size() == 0) {
    return out;
  }

  out.convert(*this);
  return out;
}

void NDArray::create_window(int32_t op_code, int64_t M, std::vector<double> args)
{
  if (size() == 0) {
    return;
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_WINDOW);

  task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(op_code));
  task.add_scalar_arg(legate::Scalar(M));

  for (double arg : args) {
    task.add_scalar_arg(legate::Scalar(arg));
  }

  runtime->submit(std::move(task));
}

void NDArray::convolve(NDArray input, NDArray filter)
{
  auto runtime = CuPyNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_CONVOLVE);

  auto p_filter = task.add_input(filter.store_);
  auto p_input  = task.add_input(input.store_);
  auto p_halo   = task.declare_partition();
  task.add_input(input.store_, p_halo);
  auto p_output = task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(shape()));
  task.add_scalar_arg(
    legate::Scalar(static_cast<int32_t>(CuPyNumericConvolveMethod::CUPYNUMERIC_CONVOLVE_AUTO)));

  auto offsets = (filter.store_.extents() + 1) / 2;

  task.add_constraint(legate::align(p_input, p_output));
  task.add_constraint(legate::bloat(p_input, p_halo, offsets, offsets));
  task.add_constraint(legate::broadcast(p_filter, legate::from_range<uint32_t>(dim())));

  runtime->submit(std::move(task));
}

NDArray NDArray::transpose()
{
  if (dim() == 1) {
    return NDArray(legate::LogicalStore(store_));
  }
  std::vector<int32_t> axes;
  for (int32_t i = dim() - 1; i > -1; --i) {
    axes.push_back(i);
  }
  return transpose(axes);
}

NDArray NDArray::transpose(std::vector<int32_t> axes)
{
  if (dim() == 1) {
    return NDArray(legate::LogicalStore(store_));
  }
  if (static_cast<int32_t>(axes.size()) != dim()) {
    throw std::invalid_argument("axes must be the same size as ndim for transpose");
  }
  return NDArray(store_.transpose(std::move(axes)));
}

NDArray NDArray::argwhere()
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  if (dim() == 0) {
    auto not_zero = legate::type_dispatch(type().code(), check_nonzero_scalar_fn{}, *this);
    if (not_zero) {
      auto result = runtime->create_array({1, 0}, legate::int64());
      return result;
    } else {
      auto result = runtime->create_array({0, 0}, legate::int64());
      return result;
    }
  }

  auto result = runtime->create_array(legate::int64(), 2);

  auto task     = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_ARGWHERE);
  auto part_out = task.declare_partition();
  auto part_in  = task.declare_partition();
  task.add_output(result.store_, part_out);
  task.add_input(store_, part_in);
  if (dim() > 1) {
    task.add_constraint(legate::broadcast(part_in, legate::from_range<uint32_t>(1, dim())));
  }
  runtime->submit(std::move(task));
  return result;
}

NDArray NDArray::flip(std::optional<std::vector<int32_t>> axis)
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto result  = runtime->create_array(shape(), type());

  result.flip(*this, axis);

  return result;
}

void NDArray::flip(NDArray rhs, std::optional<std::vector<int32_t>> axis)
{
  auto input  = rhs.store_;
  auto output = (*this).store_;

  std::vector<int32_t> axes;
  if (!axis.has_value()) {
    for (int32_t i = 0; i < dim(); ++i) {
      axes.push_back(i);
    }
  } else {
    axes = normalize_axis_vector(axis.value(), dim());
  }

  auto runtime = CuPyNumericRuntime::get_runtime();
  auto task    = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_FLIP);
  auto p_out   = task.add_output(output);
  auto p_in    = task.add_input(input);
  task.add_scalar_arg(legate::Scalar(axes));
  task.add_constraint(legate::broadcast(p_in, legate::from_range<uint32_t>(dim())));
  task.add_constraint(legate::align(p_in, p_out));

  runtime->submit(std::move(task));
}

NDArray NDArray::all(std::vector<int32_t> axis,
                     std::optional<NDArray> out,
                     bool keepdims,
                     std::optional<Scalar> initial,
                     std::optional<NDArray> where)
{
  return _perform_unary_reduction(static_cast<int32_t>(UnaryRedCode::ALL),
                                  *this,
                                  axis,
                                  std::nullopt,
                                  legate::bool_(),
                                  out,
                                  keepdims,
                                  {},
                                  initial,
                                  where);
}

NDArray NDArray::_perform_unary_reduction(int32_t op,
                                          NDArray src,
                                          const std::vector<int32_t>& axis,
                                          std::optional<legate::Type> dtype,
                                          std::optional<legate::Type> res_dtype,
                                          std::optional<NDArray> out,
                                          bool keepdims,
                                          const std::vector<NDArray>& args,
                                          std::optional<Scalar> initial,
                                          std::optional<NDArray> where)
{
  if (src.size() == 0 && !initial.has_value()) {
    if (static_cast<UnaryRedCode>(op) == UnaryRedCode::MAX ||
        static_cast<UnaryRedCode>(op) == UnaryRedCode::MIN) {
      throw std::invalid_argument("Min/max reduction is not yet supported for empty arrays");
    }
  }

  if (src.type() == legate::complex64() || src.type() == legate::complex128()) {
    if (static_cast<UnaryRedCode>(op) == UnaryRedCode::MAX ||
        static_cast<UnaryRedCode>(op) == UnaryRedCode::MIN ||
        static_cast<UnaryRedCode>(op) == UnaryRedCode::ARGMAX ||
        static_cast<UnaryRedCode>(op) == UnaryRedCode::ARGMIN) {
      throw std::runtime_error("(arg)max/min not supported for complex-type arrays");
    }
  }

  if (where.has_value() && where.value().type() != legate::bool_()) {
    throw std::invalid_argument("where array should be bool");
  }

  if ((dtype.has_value() && !dtype.value().is_primitive()) ||
      (res_dtype.has_value() && !res_dtype.value().is_primitive())) {
    throw std::invalid_argument("dtype and res_dtype should be primitive type");
  }

  // Handle scalar array without any other inputs
  if (src.dim() == 0 && !dtype.has_value() && !res_dtype.has_value() && !out.has_value() &&
      !initial.has_value() && !where.has_value()) {
    return src;
  }

  if (res_dtype.has_value()) {
    assert(!dtype.has_value());
    dtype = src.type();
  } else {
    if (dtype.has_value()) {
      res_dtype = dtype;
    } else if (out.has_value()) {
      dtype     = out.value().type();
      res_dtype = out.value().type();
    } else {
      dtype     = src.type();
      res_dtype = src.type();
    }
  }

  std::vector<int32_t> axes;
  if (axis.empty()) {
    for (auto i = 0; i < src.dim(); ++i) {
      axes.push_back(i);
    }
  } else {
    axes = normalize_axis_vector(axis, src.dim());
  }

  std::vector<uint64_t> out_shape;
  for (auto i = 0; i < src.dim(); ++i) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(src.shape()[i]);
    } else if (keepdims) {
      out_shape.push_back(1);
    }
  }

  auto runtime = CuPyNumericRuntime::get_runtime();
  if (!out.has_value()) {
    out = runtime->create_array(out_shape, res_dtype.value());
  } else if (out.value().shape() != out_shape) {
    std::string err_msg = "the output shapes do not match: expected" +
                          std::string(out_shape.begin(), out_shape.end()) + "but got " +
                          std::string(out.value().shape().begin(), out.value().shape().end());
    throw std::invalid_argument(std::move(err_msg));
  }

  if (dtype.value() != src.type()) {
    src = src.as_type(dtype.value());
  }

  NDArray result(out.value());
  if (out.value().type() != res_dtype.value()) {
    result = runtime->create_array(out_shape, res_dtype.value());
  }

  std::optional<NDArray> where_array = std::nullopt;
  if (where.has_value()) {
    where_array = broadcast_where(where.value(), src);
  }

  bool argred = static_cast<UnaryRedCode>(op) == UnaryRedCode::ARGMAX ||
                static_cast<UnaryRedCode>(op) == UnaryRedCode::ARGMIN ||
                static_cast<UnaryRedCode>(op) == UnaryRedCode::NANARGMAX ||
                static_cast<UnaryRedCode>(op) == UnaryRedCode::NANARGMIN;
  if (argred) {
    assert(!initial.has_value());
    auto argred_dtype = runtime->get_argred_type(src.type());
    result            = runtime->create_array(result.shape(), argred_dtype);
  }

  result.unary_reduction(op, src, where_array, axis, axes, keepdims, args, initial);

  if (argred) {
    unary_op(static_cast<int32_t>(UnaryOpCode::GETARG), result);
  }

  if (out.value().type() != result.type()) {
    out.value().convert(result);
  }
  return out.value();
}

void NDArray::unary_reduction(int32_t op,
                              NDArray src,
                              std::optional<NDArray> where,
                              const std::vector<int32_t>& orig_axis,
                              const std::vector<int32_t>& axes,
                              bool keepdims,
                              const std::vector<NDArray>& args,
                              std::optional<Scalar> initial)
{
  auto lhs_array = *this;
  auto rhs_array = src;
  assert(lhs_array.dim() <= rhs_array.dim());

  auto runtime = CuPyNumericRuntime::get_runtime();
  auto op_code = static_cast<UnaryRedCode>(op);

  if (initial.has_value()) {
    lhs_array.fill(initial.value());
  } else {
    lhs_array.fill(get_reduction_identity(op_code, lhs_array.type()));
  }

  auto is_where = where.has_value();
  if (lhs_array.size() == 1) {
    assert(axes.empty() || lhs_array.dim() == (rhs_array.dim() -
                                               (keepdims ? 0 : static_cast<int32_t>(axes.size()))));

    auto p_lhs = lhs_array.store_;
    while (p_lhs.dim() > 1) {
      p_lhs = p_lhs.project(0, 0);
    }

    auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_SCALAR_UNARY_RED);

    task.add_reduction(p_lhs, get_reduction_op(op_code));
    auto p_rhs = task.add_input(rhs_array.store_);
    task.add_scalar_arg(legate::Scalar(op));
    if (rhs_array.dim() > 0) {
      task.add_scalar_arg(legate::Scalar(rhs_array.shape()));
    } else {
      task.add_scalar_arg(legate::Scalar(std::vector<size_t>({1})));
    }
    task.add_scalar_arg(legate::Scalar(is_where));
    if (is_where) {
      auto p_where = task.add_input(where.value().store_);
      task.add_constraint(align(p_rhs, p_where));
    }
    for (auto& arg : args) {
      task.add_input(arg.store_);
    }

    runtime->submit(std::move(task));
  } else {
    assert(!axes.empty());
    auto result = lhs_array.store_;
    if (keepdims) {
      for (auto axis : axes) {
        result = result.project(axis, 0);
      }
    }
    auto rhs_shape = rhs_array.shape();
    for (auto axis : axes) {
      result = result.promote(axis, rhs_shape[axis]);
    }

    if (axes.size() > 1) {
      throw std::runtime_error("Need support for reducing multiple dimensions");
    }

    auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_UNARY_RED);

    auto p_lhs = task.add_reduction(result, get_reduction_op(op_code));
    auto p_rhs = task.add_input(rhs_array.store_);
    task.add_scalar_arg(legate::Scalar(axes[0]));
    task.add_scalar_arg(legate::Scalar(op));
    task.add_scalar_arg(legate::Scalar(is_where));
    if (is_where) {
      auto p_where = task.add_input(where.value().store_);
      task.add_constraint(align(p_rhs, p_where));
    }
    for (auto& arg : args) {
      task.add_input(arg.store_);
    }
    task.add_constraint(align(p_lhs, p_rhs));

    runtime->submit(std::move(task));
  }
}

NDArray NDArray::broadcast_where(NDArray where, NDArray source)
{
  if (where.shape() == source.shape()) {
    return where;
  }

  auto where_shape = broadcast_shapes({where, source});
  auto where_store = broadcast(where_shape, where.store_);

  auto runtime = CuPyNumericRuntime::get_runtime();
  return runtime->create_array(std::move(where_store));
}

void NDArray::convert(NDArray rhs, int32_t nan_op)
{
  NDArray lhs_array(*this);
  NDArray rhs_array(rhs);
  assert(lhs_array.type() != rhs_array.type());

  auto lhs_s = lhs_array.store_;
  auto rhs_s = rhs_array.store_;

  auto runtime = CuPyNumericRuntime::get_runtime();
  auto task    = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_CONVERT);
  auto p_lhs   = task.add_output(lhs_s);
  auto p_rhs   = task.add_input(rhs_s);
  task.add_scalar_arg(legate::Scalar(nan_op));
  task.add_constraint(legate::align(p_lhs, p_rhs));
  runtime->submit(std::move(task));
}

NDArray NDArray::diag_helper(int32_t offset,
                             std::vector<int32_t> axes,
                             bool extract,
                             bool trace,
                             const std::optional<legate::Type>& type,
                             std::optional<NDArray> out)
{
  auto runtime = CuPyNumericRuntime::get_runtime();

  if (dim() <= 1) {
    throw std::invalid_argument("diag_helper is implemented for dim > 1");
  }
  if (out.has_value() && !trace) {
    throw std::invalid_argument("diag_helper supports out only for trace=true");
  }
  if (type.has_value() && !trace) {
    throw std::invalid_argument("diag_helper supports type only for trace=true");
  }

  auto N = axes.size();
  assert(N > 0);

  std::set<size_t> s_axes(axes.begin(), axes.end());
  if (N != s_axes.size()) {
    throw std::invalid_argument("axes passed to diag_helper should be all different");
  }
  if (static_cast<size_t>(dim()) < N) {
    throw std::invalid_argument("Dimension of input array shouldn't be less than number of axes");
  }
  std::vector<int32_t> transpose_axes;
  for (int32_t ax = 0; ax < dim(); ++ax) {
    if (std::find(axes.begin(), axes.end(), ax) == axes.end()) {
      transpose_axes.push_back(ax);
    }
  }

  NDArray a = runtime->create_array(store_.type());

  uint64_t diag_size;
  if (N == 2) {
    if (offset >= 0) {
      transpose_axes.push_back(axes[0]);
      transpose_axes.push_back(axes[1]);
    } else {
      transpose_axes.push_back(axes[1]);
      transpose_axes.push_back(axes[0]);
      offset = -offset;
    }
    a = transpose(transpose_axes);
    if (offset >= static_cast<int32_t>(a.shape()[dim() - 1])) {
      throw std::invalid_argument("'offset' for diag or diagonal must be in range");
    }
    diag_size = std::max(static_cast<uint64_t>(0),
                         std::min(a.shape().end()[-2], a.shape().end()[-1] - offset));
  } else if (N > 2) {
    if (offset != 0) {
      throw std::invalid_argument("offset supported for number of axes == 2");
    }
    auto sort_axes = [this](size_t i, size_t j) { return (shape()[i] < shape()[j]); };
    std::sort(axes.begin(), axes.end(), sort_axes);
    std::reverse(axes.begin(), axes.end());
    transpose_axes.insert(transpose_axes.end(), axes.begin(), axes.end());
    a         = transpose(transpose_axes);
    diag_size = a.shape()[a.dim() - 1];
  } else if (N < 2) {
    throw std::invalid_argument("number of axes should be more than 1");
  }

  std::vector<uint64_t> tr_shape;
  for (size_t i = 0; i < a.dim() - N; ++i) {
    tr_shape.push_back(a.shape()[i]);
  }

  std::vector<uint64_t> out_shape;
  if (trace) {
    if (N != 2) {
      throw std::invalid_argument("exactly 2 axes should be passed to trace");
    }
    if (dim() == 2) {
      out_shape = {1};
    } else if (dim() > 2) {
      out_shape = tr_shape;
    } else {
      throw std::invalid_argument("dimension of the array for trace operation should be >=2");
    }
  } else {
    tr_shape.push_back(diag_size);
    out_shape = tr_shape;
  }

  if (out && out->shape() != out_shape) {
    throw std::invalid_argument("output array has the wrong shape");
  }

  auto res_type = [&] {
    if (type) {
      return type.value();
    } else if (out) {
      return out->type();
    } else {
      return store_.type();
    }
  }();

  if (store_.type() != res_type) {
    a = a.as_type(res_type);
  }

  if (out && out->type() == res_type) {
    out->diag_task(a, offset, N, extract, trace);
    return out.value();
  } else {
    auto res = runtime->create_array(out_shape, res_type);
    res.diag_task(a, offset, N, extract, trace);
    if (out) {
      out->assign(res);
    }
    return res;
  }
}

void NDArray::diag_task(NDArray rhs, int32_t offset, int32_t naxes, bool extract, bool trace)
{
  auto runtime = CuPyNumericRuntime::get_runtime();

  legate::LogicalStore diag   = get_store();
  legate::LogicalStore matrix = get_store();

  auto zero = legate::type_dispatch(type().code(), generate_zero_fn{});
  fill(zero);

  if (extract) {
    diag       = store_;
    matrix     = rhs.store_;
    auto ndim  = rhs.dim();
    auto start = matrix.dim() - naxes;
    auto n     = ndim - 1;
    if (naxes == 2) {
      if (offset > 0) {
        matrix = matrix.slice(start + 1, legate::Slice(offset));
      }
      if (trace) {
        if (ndim == 2) {
          diag = diag.promote(0, matrix.extents().data()[0]);
          diag = diag.project(1, 0).promote(1, matrix.extents().data()[1]);
        } else {
          for (int32_t i = 0; i < naxes; ++i) {
            diag = diag.promote(start, matrix.extents().data().end()[-i - 1]);
          }
        }
      } else {
        if (matrix.extents().data()[n - 1] < matrix.extents().data()[n]) {
          diag = diag.promote(start + 1, matrix.extents().data()[ndim - 1]);
        } else {
          diag = diag.promote(start, matrix.extents().data()[ndim - 2]);
        }
      }
    } else {
      for (int32_t i = 1; i < naxes; ++i) {
        diag = diag.promote(start, matrix.extents().data().end()[-i - 1]);
      }
    }
  } else {
    matrix = store_;
    diag   = rhs.store_;
    if (offset > 0) {
      matrix = matrix.slice(1, slice(offset));
    } else if (offset < 0) {
      matrix = matrix.slice(0, slice(-offset));
    }

    if (shape()[0] < shape()[1]) {
      diag = diag.promote(1, shape()[1]);
    } else {
      diag = diag.promote(0, shape()[0]);
    }
  }

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_DIAG);
  if (extract) {
    auto p_diag   = task.add_reduction(diag, get_reduction_op(UnaryRedCode::SUM));
    auto p_matrix = task.add_input(matrix);
    task.add_constraint(legate::align(p_matrix, p_diag));
  } else {
    auto p_matrix = task.add_output(matrix);
    auto p_diag   = task.add_input(diag);
    task.add_input(matrix, p_matrix);
    task.add_constraint(legate::align(p_diag, p_matrix));
  }

  task.add_scalar_arg(legate::Scalar(naxes));
  task.add_scalar_arg(legate::Scalar(extract));

  runtime->submit(std::move(task));
}

void NDArray::put(NDArray indices, NDArray values, std::string mode)
{
  if (values.size() == 0 || indices.size() == 0 || size() == 0) {
    return;
  }

  if (mode != "raise" && mode != "wrap" && mode != "clip") {
    std::stringstream ss;
    ss << "mode must be one of 'clip', 'raise', or 'wrap' (got  " << mode << ")";
    throw std::invalid_argument(ss.str());
  }

  // type conversion
  indices = indices._warn_and_convert(legate::int64());
  values  = values._warn_and_convert(type());

  // reshape indices
  if (indices.dim() > 1) {
    // When reshape is ready, reshape can be used.
    indices = indices._wrap(indices.size());
  }

  // call _wrap on the values if they need to be wrapped
  if (values.dim() != indices.dim() || values.size() != indices.size()) {
    values = values._wrap(indices.size());
  }

  if (mode == "wrap") {
    indices = indices.wrap_indices(Scalar(int64_t(size())));
  } else if (mode == "clip") {
    indices = indices.clip_indices(Scalar(int64_t(0)), Scalar(int64_t(size() - 1)));
  }

  if (indices.store_.has_scalar_storage() || indices.store_.transformed()) {
    bool change_shape = indices.store_.has_scalar_storage();
    indices           = indices._convert_future_to_regionfield(change_shape);
  }
  if (values.store_.has_scalar_storage() || values.store_.transformed()) {
    bool change_shape = values.store_.has_scalar_storage();
    values            = values._convert_future_to_regionfield(change_shape);
  }
  bool need_copy = false;
  auto self_tmp  = *this;
  if (self_tmp.store_.has_scalar_storage() || self_tmp.store_.transformed()) {
    need_copy         = true;
    bool change_shape = self_tmp.store_.has_scalar_storage();
    self_tmp          = self_tmp._convert_future_to_regionfield(change_shape);
  }

  auto runtime      = CuPyNumericRuntime::get_runtime();
  bool check_bounds = (mode == "raise");
  auto task         = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_WRAP);
  auto indirect = runtime->create_array(indices.shape(), legate::point_type(self_tmp.dim()), false);
  auto p_indirect = task.add_output(indirect.store_);
  auto p_indices  = task.add_input(indices.store_);
  task.add_scalar_arg(legate::Scalar(self_tmp.shape()));
  task.add_scalar_arg(legate::Scalar(true));  // has_input
  task.add_scalar_arg(legate::Scalar(check_bounds));
  task.add_constraint(legate::align(p_indices, p_indirect));
  task.throws_exception(true);
  runtime->submit(std::move(task));

  auto legate_runtime = legate::Runtime::get_runtime();
  legate_runtime->issue_scatter(self_tmp.store_, indirect.store_, values.store_);

  if (need_copy) {
    if (store_.has_scalar_storage()) {
      self_tmp = runtime->create_array(std::move(self_tmp.store_.project(0, 0)));
    }
    assign(self_tmp);
  }
}

NDArray NDArray::copy()
{
  auto runtime        = CuPyNumericRuntime::get_runtime();
  auto legate_runtime = legate::Runtime::get_runtime();
  auto out            = runtime->create_array(shape(), type());
  if (store_.has_scalar_storage() && out.store_.has_scalar_storage()) {
    legate_runtime->issue_fill(out.store_, store_);
  } else {
    out.assign(*this);
  }
  return out;
}

NDArray NDArray::repeat(int64_t repeats, std::optional<int32_t> axis)
{
  if (repeats < 0) {
    throw std::invalid_argument("negative dimensions are not allowed");
  }

  auto runtime = CuPyNumericRuntime::get_runtime();

  // when array is a scalar
  if (dim() == 0) {
    if (axis.has_value() && axis.value() != 0 && axis.value() != -1) {
      throw std::invalid_argument("axis is out of bounds for array of dimension 0");
    }
    auto out = runtime->create_array({static_cast<size_t>(repeats)}, type());
    out._fill(store_);
    return out;
  }

  // if no axes specified, flatten array
  auto src = *this;
  if (!axis.has_value() && src.dim() > 1) {
    // When reshape is ready, reshape can be used.
    src = src._wrap(src.size());
  }

  int32_t axis_int = normalize_axis_index(axis.value_or(0), src.dim());

  // If repeats is on a zero sized axis_int, then return the array.
  if (src.shape()[axis_int] == 0) {
    return src.copy();
  }

  // repeats is a scalar
  if (0 == repeats) {
    auto empty_shape      = src.shape();
    empty_shape[axis_int] = 0;
    return runtime->create_array(empty_shape, src.type());
  }

  auto task = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_REPEAT);

  auto out_shape = src.shape();
  out_shape[axis_int] *= repeats;
  auto out    = runtime->create_array(out_shape, src.type());
  auto p_self = task.declare_partition();
  auto p_out  = task.declare_partition();
  task.add_input(src.store_, p_self);
  task.add_output(out.store_, p_out);
  std::vector<std::uint64_t> factors(src.dim(), 1);
  factors[axis_int] = uint64_t(repeats);
  task.add_constraint(legate::scale(legate::tuple<std::uint64_t>(factors), p_self, p_out));
  task.add_scalar_arg(Scalar(axis_int));
  task.add_scalar_arg(Scalar(true));  // scalar_repeats
  task.add_scalar_arg(Scalar(repeats));
  runtime->submit(std::move(task));

  return out;
}

NDArray NDArray::repeat(NDArray repeats, std::optional<int32_t> axis)
{
  if (repeats.size() == 0 || repeats.dim() > 1) {
    throw std::invalid_argument("`repeats` should be scalar or 1D array");
  }

  // when array is a scalar
  if (dim() == 0) {
    throw std::invalid_argument(
      "`repeat` with a scalar is only "
      "implemented for scalar values of the parameter `repeats`.");
  }

  // if no axes specified, flatten array
  auto src = *this;
  if (!axis.has_value() && src.dim() > 1) {
    // When reshape is ready, reshape can be used.
    src = src._wrap(src.size());
  }

  int32_t axis_int = normalize_axis_index(axis.value_or(0), src.dim());

  // If repeats is on a zero sized axis_int, then return the array.
  if (src.shape()[axis_int] == 0) {
    return src.copy();
  }

  if (repeats.get_store().has_scalar_storage()) {
    size_t len = src.shape()[axis_int];
    if (len > 1) {
      repeats = repeats._wrap(len);
    } else {
      repeats = repeats._convert_future_to_regionfield(true);
    }
  }

  // repeats is an array
  if (repeats.shape()[0] != src.shape()[axis_int]) {
    throw std::invalid_argument("incorrect shape of repeats array");
  }
  if (repeats.type() != legate::int64()) {
    repeats = repeats._warn_and_convert(legate::int64());
  }

  auto runtime        = CuPyNumericRuntime::get_runtime();
  auto legate_runtime = legate::Runtime::get_runtime();
  auto out_store      = legate_runtime->create_store(src.type(), src.dim());
  auto task           = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_REPEAT);
  auto p_src          = task.add_input(src.store_);
  task.add_output(out_store);
  task.add_scalar_arg(Scalar(axis_int));
  task.add_scalar_arg(Scalar(false));  // scalar_repeats
  auto shape         = src.shape();
  auto repeats_store = repeats.store_;
  for (int32_t dim = 0; dim < src.dim(); ++dim) {
    if (dim == axis_int) {
      continue;
    }
    repeats_store = repeats_store.promote(dim, shape[dim]);
  }
  auto p_repeats = task.add_input(repeats_store);
  task.add_constraint(legate::align(p_src, p_repeats));
  runtime->submit(std::move(task));

  return runtime->create_array(std::move(out_store));
}

NDArray NDArray::_convert_future_to_regionfield(bool change_shape)
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  if (change_shape && dim() == 0) {
    auto out = runtime->create_array({1}, type(), false);
    out.assign(*this);
    return out;
  }
  auto out = runtime->create_array(shape(), type(), false);
  out.assign(*this);
  return out;
}

NDArray NDArray::_wrap(size_t new_len)
{
  auto runtime = CuPyNumericRuntime::get_runtime();

  if (0 == new_len) {
    return runtime->create_array({0}, type());
  }
  if (size() == 0) {
    throw std::invalid_argument("Unable to wrap an empty array to a length greater than 0.");
  }
  if (1 == new_len) {
    auto tmp_store = store_;
    for (int32_t i = 0; i < dim(); ++i) {
      tmp_store = tmp_store.project(0, 0);
    }
    NDArray tmp_arr(std::move(tmp_store));
    auto out = runtime->create_array(tmp_arr.shape(), tmp_arr.type());
    out.assign(tmp_arr);
    return out;
  }

  auto src = *this;
  if (src.store_.has_scalar_storage() || src.store_.transformed()) {
    bool change_shape = src.store_.has_scalar_storage();
    src               = src._convert_future_to_regionfield(change_shape);
  }

  auto task     = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_WRAP);
  auto indirect = runtime->create_array({new_len}, legate::point_type(src.dim()), false);
  task.add_output(indirect.store_);
  task.add_scalar_arg(legate::Scalar(src.shape()));
  task.add_scalar_arg(legate::Scalar(false));  // has_input
  task.add_scalar_arg(legate::Scalar(false));  // check bounds
  runtime->submit(std::move(task));

  auto legate_runtime = legate::Runtime::get_runtime();
  auto out            = runtime->create_array({new_len}, src.type(), false);
  legate_runtime->issue_gather(out.store_, src.store_, indirect.store_);

  return out;
}

NDArray NDArray::_warn_and_convert(legate::Type const& type)
{
  if (this->type() != type) {
    std::stringstream ss;
    ss << "converting array to " << type.to_string() << " type";
    cupynumeric_log().warning() << ss.str();
    return as_type(type);
  }
  return *this;
}

NDArray NDArray::wrap_indices(Scalar const& n)
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto out     = runtime->create_array(shape(), type());
  auto divisor = cupynumeric::full({}, n);
  out.binary_op(static_cast<int32_t>(cupynumeric::BinaryOpCode::MOD), *this, divisor);
  return out;
}

NDArray NDArray::clip_indices(Scalar const& min, Scalar const& max)
{
  auto runtime = CuPyNumericRuntime::get_runtime();
  auto out     = runtime->create_array(shape(), type());
  auto task    = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_UNARY_OP);
  auto p_out   = task.add_output(out.store_);
  auto p_in    = task.add_input(store_);
  task.add_scalar_arg(legate::Scalar(static_cast<int32_t>(UnaryOpCode::CLIP)));
  task.add_scalar_arg(min);
  task.add_scalar_arg(max);
  task.add_constraint(align(p_out, p_in));
  runtime->submit(std::move(task));
  return out;
}

NDArray NDArray::diagonal(int32_t offset,
                          std::optional<int32_t> axis1,
                          std::optional<int32_t> axis2,
                          std::optional<bool> extract)
{
  if (dim() == 1) {
    if (extract.has_value() && extract.value()) {
      throw std::invalid_argument("extract can be true only for dim >=2");
    }
    if (axis1 || axis2) {
      throw std::invalid_argument("Axes shouldn't be specified when getting diagonal for 1D array");
    }
    auto runtime = CuPyNumericRuntime::get_runtime();
    auto m       = shape()[0] + std::abs(offset);
    auto res     = runtime->create_array({m, m}, store_.type());
    res.diag_task(*this, offset, 0, false, false);
    return res;
  } else {
    if (!axis1) {
      axis1 = 0;
    }
    if (!axis2) {
      axis2 = 1;
    }
    if (!extract.has_value()) {
      extract = true;
    }
    return diag_helper(offset, {axis1.value(), axis2.value()}, extract.value());
  }
}

NDArray NDArray::trace(int32_t offset,
                       int32_t axis1,
                       int32_t axis2,
                       std::optional<legate::Type> type,
                       std::optional<NDArray> out)
{
  if (dim() < 2) {
    throw std::invalid_argument("trace operation can't be called on a array with DIM<2");
  }
  return diag_helper(offset, {axis1, axis2}, true, true, type, out);
}

NDArray NDArray::ravel(std::string order) { return reshape({-1}, order); }

NDArray NDArray::reshape(std::vector<int64_t> newshape, std::string order)
{
  if (order == "A") {
    order = "C";
  }
  if (order == "F") {
    throw std::invalid_argument(
      "cuPyNumeric has not implemented reshape using Fortran-like index order.");
  }
  if (order != "C") {
    throw std::invalid_argument("order must be one of 'C', 'F', 'A'");
  }
  return reshape(newshape);
}

NDArray NDArray::reshape(std::vector<int64_t> newshape)
{
  auto runtime     = cupynumeric::CuPyNumericRuntime::get_runtime();
  int num_unknowns = std::count_if(newshape.begin(), newshape.end(), [](auto x) { return x < 0; });
  if (num_unknowns > 1) {
    throw std::invalid_argument("can only specify one unknown dimension");
  }

  // case 1: zero size
  if (size() == 0) {
    if (1 == num_unknowns) {
      std::replace_if(newshape.begin(), newshape.end(), [](auto x) { return x < 0; }, 0);
    }
    auto out_size = vec_prod(newshape);
    if (out_size != 0) {
      throw std::invalid_argument("new shape is not the same size as the original");
    }
    return runtime->create_array(vec_convert<int64_t, uint64_t>(newshape), type());
  }

  int64_t known_volume = 1;
  for (auto x : newshape) {
    if (x >= 0) {
      known_volume *= x;
    }
  }
  if (num_unknowns > 0 && 0 == known_volume) {
    throw std::invalid_argument("cannot reshape, size mismatch");
  }
  int64_t unknown_extent = (0 == num_unknowns) ? 1 : size() / known_volume;
  if (unknown_extent * known_volume != size()) {
    throw std::invalid_argument("cannot reshape, size mismatch");
  }
  std::replace_if(newshape.begin(), newshape.end(), [](auto x) { return x < 0; }, unknown_extent);

  auto in_shape  = shape();
  auto out_shape = vec_convert<int64_t, uint64_t>(newshape);

  // case 2: same shape
  if (vec_is_equal(in_shape, out_shape)) {
    return *this;
  }

  bool need_copy = false;
  auto out_iter  = out_shape.rbegin();
  std::for_each(
    in_shape.rbegin(), in_shape.rend(), [&out_shape, &out_iter, &need_copy](uint64_t elem_in) {
      uint64_t prod = 1;
      for (; prod < elem_in && out_iter != out_shape.rend(); ++out_iter) {
        prod *= *out_iter;
      }
      if (prod != elem_in) {
        need_copy = true;
      }
    });

  // case 3: need copy
  if (need_copy) {
    auto flat_arr       = array({size()}, type());
    auto in_shape_store = flat_arr.get_store().delinearize(0, in_shape);
    NDArray in_shape_arr(std::move(in_shape_store));
    in_shape_arr.assign(*this);
    auto out_shape_store = flat_arr.get_store().delinearize(0, out_shape);
    NDArray out_shape_arr(std::move(out_shape_store));
    return out_shape_arr;
  }

  // case 4: No need to copy, provides a view to the input store
  out_iter       = out_shape.rbegin();
  auto out_store = get_store();

  std::for_each(in_shape.rbegin(),
                in_shape.rend(),
                [&out_shape, &out_iter, &out_store, dim_in = int32_t(in_shape.size())](
                  uint64_t elem_in) mutable {
                  --dim_in;
                  if (out_iter != out_shape.rend() && elem_in == *out_iter) {
                    ++out_iter;
                    // NOOP
                    return;
                  }
                  if (elem_in == 1) {
                    // "project" operation
                    out_store = out_store.project(dim_in, 0);
                    return;
                  }
                  // "delinearize" operation
                  std::vector<uint64_t> new_sizes;
                  new_sizes.reserve(8);
                  for (uint64_t prod = 1; prod < elem_in && out_iter != out_shape.rend();
                       ++out_iter) {
                    prod *= *out_iter;
                    new_sizes.push_back(*out_iter);
                  }
                  std::reverse(new_sizes.begin(), new_sizes.end());
                  out_store = out_store.delinearize(dim_in, new_sizes);
                });

  for (; out_iter != out_shape.rend(); ++out_iter) {
    // "promote" operation
    out_store = out_store.promote(0, 1);
  }

  return NDArray(std::move(out_store));
}

NDArray NDArray::squeeze(
  std::optional<std::reference_wrapper<std::vector<int32_t> const>> axis) const
{
  auto result = store_;
  if (!axis.has_value()) {
    int shift = 0;
    for (int d = 0; d < dim(); d++) {
      if (result.extents().data()[d + shift] == 1) {
        result = result.project(d + shift, 0);
        shift -= 1;
      }
    }
  } else {
    auto computed_axis = normalize_axis_vector(axis.value(), dim());
    for (auto ax : computed_axis) {
      if (shape()[ax] != 1) {
        throw std::invalid_argument("can only select axes to squeeze out with size equal to one");
      }
    }
    int shift = 0;
    for (auto dim : computed_axis) {
      result = result.project(dim + shift, 0);
      shift -= 1;
    }
  }
  if (result.extents().data() == store_.extents().data()) {
    return *this;
  } else {
    auto runtime = CuPyNumericRuntime::get_runtime();
    return runtime->create_array(std::move(result));
  }
}

void NDArray::where(NDArray rhs1, NDArray rhs2, NDArray rhs3)
{
  const auto& out_shape = shape();
  auto rhs1_store       = broadcast(out_shape, rhs1.store_);
  auto rhs2_store       = broadcast(out_shape, rhs2.store_);
  auto rhs3_store       = broadcast(out_shape, rhs3.store_);
  assert(store_.type() == rhs2.store_.type());
  assert(store_.type() == rhs3.store_.type());

  auto runtime = CuPyNumericRuntime::get_runtime();
  auto task    = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_WHERE);

  auto p_lhs  = task.declare_partition();
  auto p_rhs1 = task.declare_partition();
  auto p_rhs2 = task.declare_partition();
  auto p_rhs3 = task.declare_partition();

  task.add_output(store_, p_lhs);
  task.add_input(rhs1_store, p_rhs1);
  task.add_input(rhs2_store, p_rhs2);
  task.add_input(rhs3_store, p_rhs3);

  task.add_constraint(legate::align(p_lhs, p_rhs1));
  task.add_constraint(legate::align(p_lhs, p_rhs2));
  task.add_constraint(legate::align(p_lhs, p_rhs3));

  runtime->submit(std::move(task));
}

NDArray NDArray::_maybe_convert(const legate::Type& type) const
{
  if (type == store_.type()) {
    return *this;
  } else {
    return as_type(type);
  }
}

void NDArray::_verify_mode_extent(const std::map<char, int>& mode2extent,
                                  const std::vector<char>& modes,
                                  const std::vector<std::uint64_t>& shape) const
{
  for (int32_t i = 0; i < modes.size(); i++) {
    assert(mode2extent.at(modes[i]) == shape[i]);
  }
}

legate::LogicalStore NDArray::_alphabetical_transpose(legate::LogicalStore store,
                                                      const std::vector<char>& modes) const
{
  std::map<char, unsigned int> map_mode_id;
  for (int i = 0; i < modes.size(); i++) {
    map_mode_id[modes[i]] = i;
  }

  auto modes_copy{modes};
  std::sort(modes_copy.begin(), modes_copy.end());
  std::vector<int32_t> axes;
  for (int i = 0; i < modes_copy.size(); i++) {
    axes.push_back(map_mode_id[modes_copy[i]]);
  }
  return store.transpose(std::move(axes));
}

enum BlasOps {
  BlasOperationNone = 0,
  BlasOperationVV   = 1,
  BlasOperationMV   = 2,
  BlasOperationMM   = 3,
};

// This function ports contract function in cupynumeric/_thunk/deferred.py
// to-do: handle store overlap
// to-do: support np.float16
void NDArray::contract(const std::vector<char>& lhs_modes,
                       NDArray rhs1,
                       const std::vector<char>& rhs1_modes,
                       NDArray rhs2,
                       const std::vector<char>& rhs2_modes,
                       const std::map<char, int>& mode2extent)
{
  // Sanity checks
  // no duplicate modes within an array
  std::set<char> s_lhs_modes(lhs_modes.begin(), lhs_modes.end());
  assert(lhs_modes.size() == s_lhs_modes.size());
  std::set<char> s_rhs1_modes(rhs1_modes.begin(), rhs1_modes.end());
  assert(rhs1_modes.size() == s_rhs1_modes.size());
  std::set<char> s_rhs2_modes(rhs2_modes.begin(), rhs2_modes.end());
  assert(rhs2_modes.size() == s_rhs2_modes.size());
  // no singleton modes
  std::unordered_map<char, size_t> mode_counts;
  for (auto v : lhs_modes) {
    ++mode_counts[v];
  }
  for (auto v : rhs1_modes) {
    ++mode_counts[v];
  }
  for (auto v : rhs2_modes) {
    ++mode_counts[v];
  }
  for (auto const& count : mode_counts) {
    assert(count.second == 2 || count.second == 3);
  }

  // arrays and mode lists agree on dimensionality
  assert(dim() == lhs_modes.size());
  assert(rhs1.dim() == rhs1_modes.size());
  assert(rhs2.dim() == rhs2_modes.size());
  // array shapes agree with mode extents (broadcasting should have been
  // handled by the frontend)
  _verify_mode_extent(mode2extent, lhs_modes, shape());
  _verify_mode_extent(mode2extent, rhs1_modes, rhs1.shape());
  _verify_mode_extent(mode2extent, rhs2_modes, rhs2.shape());
  // casting has been handled by the frontend
  assert(type() == rhs1.type());
  assert(type() == rhs2.type());

  // to-do: Handle store overlap

  enum BlasOps blas_op    = BlasOperationNone;
  bool count_unequals_two = false;
  for (auto const& count : mode_counts) {
    if (count.second != 2) {
      count_unequals_two = true;
      break;
    }
  }
  if (!count_unequals_two) {
    if (lhs_modes.size() == 0 && rhs1_modes.size() == 1 && rhs2_modes.size() == 1) {
      blas_op = BlasOperationVV;
    } else if (lhs_modes.size() == 1 && (rhs1_modes.size() == 2 and rhs2_modes.size() == 1 ||
                                         rhs1_modes.size() == 1 and rhs2_modes.size() == 2)) {
      blas_op = BlasOperationMV;
    } else if (lhs_modes.size() == 2 and rhs1_modes.size() == 2 and rhs2_modes.size() == 2) {
      blas_op = BlasOperationMM;
    }
  }
  // to-do: support np.float16

  // Clear output array
  auto zero = legate::type_dispatch(type().code(), generate_zero_fn{});
  fill(zero);
  // Pull out the stores
  auto lhs_s   = store_;
  auto rhs1_s  = rhs1.store_;
  auto rhs2_s  = rhs2.store_;
  auto runtime = CuPyNumericRuntime::get_runtime();

  if (blas_op != BlasOperationNone) {
    if (blas_op == BlasOperationVV) {  // for Scalar, dim() == 0 or 1?
      auto task  = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_DOT);
      auto redop = get_reduction_op(UnaryRedCode::SUM);
      task.add_reduction(lhs_s, redop);
      auto p_rhs1 = task.add_input(rhs1_s);
      auto p_rhs2 = task.add_input(rhs2_s);
      task.add_constraint(align(p_rhs1, p_rhs2));
      runtime->submit(std::move(task));
    } else if (blas_op == BlasOperationMV) {
      // Matrix-vector or vector-matrix multiply
      // b,(ab/ba)->a --> (ab/ba),b->a
      if (rhs1_modes.size() == 1) {
        std::swap(rhs1_s, rhs2_s);
      }
      // ba,b->a --> ab,b->a
      if (rhs1_modes[0] == rhs2_modes[0]) {
        rhs1_s = rhs1_s.transpose({1, 0});
      }
      auto m = rhs1_s.extents().data()[0];
      auto n = rhs1_s.extents().data()[1];
      rhs2_s = rhs2_s.promote(0, m);
      lhs_s  = lhs_s.promote(1, n);

      auto task   = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_MATVECMUL);
      auto redop  = get_reduction_op(UnaryRedCode::SUM);
      auto p_lhs  = task.add_reduction(lhs_s, redop);
      auto p_rhs1 = task.add_input(rhs1_s);
      auto p_rhs2 = task.add_input(rhs2_s);
      task.add_constraint(align(p_lhs, p_rhs1));
      task.add_constraint(align(p_rhs1, p_rhs2));
      runtime->submit(std::move(task));
    } else if (blas_op == BlasOperationMM) {
      auto rhs1_modes_copy{rhs1_modes};
      auto rhs2_modes_copy{rhs2_modes};
      // (cb/bc),(ab/ba)->ac --> (ab/ba),(cb/bc)->ac
      if (!_is_in_vector<char>(rhs1_modes, lhs_modes[0])) {
        std::swap(rhs1_s, rhs2_s);
        std::swap(rhs1_modes_copy, rhs2_modes_copy);
      }
      assert(_is_in_vector<char>(rhs1_modes_copy, lhs_modes[0]) &&
             _is_in_vector<char>(rhs2_modes_copy, lhs_modes[1]));
      // ba,?->ac --> ab,?->ac
      if (lhs_modes[0] != rhs1_modes_copy[0]) {
        rhs1_s = rhs1_s.transpose({1, 0});
      }
      // ?,cb->ac --> ?,bc->ac
      if (lhs_modes[1] != rhs2_modes_copy[1]) {
        rhs2_s = rhs2_s.transpose({1, 0});
      }
      auto m = shape()[0];
      auto n = shape()[1];
      auto k = rhs1.shape()[1];
      assert(m == rhs1.shape()[0]);
      assert(n == rhs2.shape()[1]);
      assert(k == rhs2.shape()[0]);

      dot_MM(rhs1_s, rhs2_s);
    } else {
      assert(false);
    }
    return;
  }

  lhs_s  = _alphabetical_transpose(lhs_s, lhs_modes);
  rhs1_s = _alphabetical_transpose(rhs1_s, rhs1_modes);
  rhs2_s = _alphabetical_transpose(rhs2_s, rhs2_modes);

  std::vector<int8_t> lhs_dim_mask;
  std::vector<int8_t> rhs1_dim_mask;
  std::vector<int8_t> rhs2_dim_mask;

  std::vector<char> sorted_modes;
  for (std::map<char, int>::const_iterator it = mode2extent.begin(); it != mode2extent.end();
       it++) {
    sorted_modes.push_back(it->first);
  }
  std::sort(sorted_modes.begin(), sorted_modes.end());
  for (int i = 0; i < sorted_modes.size(); i++) {
    auto dim    = i;
    auto mode   = sorted_modes[i];
    auto extent = mode2extent.at(mode);

    auto add_mode = [&](legate::LogicalStore store,
                        const std::vector<char>& modes,
                        std::vector<int8_t>& dim_mask) {
      if (!_is_in_vector<char>(modes, mode)) {
        dim_mask.emplace_back(false);
        return store.promote(dim, extent);
      } else {
        dim_mask.emplace_back(true);
        return store;
      }
    };

    lhs_s  = add_mode(lhs_s, lhs_modes, lhs_dim_mask);
    rhs1_s = add_mode(rhs1_s, rhs1_modes, rhs1_dim_mask);
    rhs2_s = add_mode(rhs2_s, rhs2_modes, rhs2_dim_mask);
  }

  assert(lhs_s.extents().data() == rhs1_s.extents().data());
  assert(lhs_s.extents().data() == rhs2_s.extents().data());

  auto task   = runtime->create_task(CuPyNumericOpCode::CUPYNUMERIC_CONTRACT);
  auto redop  = get_reduction_op(UnaryRedCode::SUM);
  auto p_lhs  = task.add_reduction(lhs_s, redop);
  auto p_rhs1 = task.add_input(rhs1_s);
  auto p_rhs2 = task.add_input(rhs2_s);

  auto add_scalar_arg = [&](std::vector<int8_t> dim_mask) {
    auto arg_type = legate::fixed_array_type(legate::bool_(), dim_mask.size());
    task.add_scalar_arg(legate::Scalar(arg_type, dim_mask.data(), true));
  };
  add_scalar_arg(lhs_dim_mask);
  add_scalar_arg(rhs1_dim_mask);
  add_scalar_arg(rhs2_dim_mask);

  task.add_constraint(align(p_lhs, p_rhs1));
  task.add_constraint(align(p_rhs1, p_rhs2));
  runtime->submit(std::move(task));
}

legate::LogicalStore NDArray::get_store() { return store_; }

legate::LogicalStore NDArray::broadcast(const std::vector<uint64_t>& shape,
                                        legate::LogicalStore& store) const
{
  int32_t diff = static_cast<int32_t>(shape.size()) - store.dim();

#ifdef DEBUG_CUPYNUMERIC
  assert(diff >= 0);
#endif

  auto result = store;
  for (int32_t dim = 0; dim < diff; ++dim) {
    result = result.promote(dim, shape[dim]);
  }

  std::vector<uint64_t> orig_shape = result.extents().data();
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    if (orig_shape[dim] != shape[dim]) {
#ifdef DEBUG_CUPYNUMERIC
      assert(orig_shape[dim] == 1);
#endif
      result = result.project(dim, 0).promote(dim, shape[dim]);
    }
  }

#ifdef DEBUG_CUPYNUMERIC
  assert(static_cast<size_t>(result.dim()) == shape.size());
#endif

  return result;
}

legate::LogicalStore NDArray::broadcast(NDArray rhs1, NDArray rhs2)
{
  if (rhs1.shape() == rhs2.shape()) {
    return rhs1.store_;
  }
  auto out_shape = broadcast_shapes({rhs1, rhs2});
  return broadcast(out_shape, rhs1.store_);
}

/*static*/ legate::Library NDArray::get_library()
{
  return CuPyNumericRuntime::get_runtime()->get_library();
}

}  // namespace cupynumeric
