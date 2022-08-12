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

#include "cunumeric/array.h"
#include "cunumeric/runtime.h"
#include "cunumeric/random/rand_util.h"
#include "cunumeric/unary/unary_red_util.h"

namespace cunumeric {

Array::Array(legate::LogicalStore&& store) : store_(std::forward<legate::LogicalStore>(store)) {}

int32_t Array::dim() const { return store_.dim(); }

const std::vector<size_t>& Array::shape() const { return store_.extents().data(); }

legate::LegateTypeCode Array::code() const { return store_.code(); }

static std::vector<int64_t> compute_strides(const std::vector<size_t>& shape)
{
  std::vector<int64_t> strides(shape.size());
  if (shape.size() > 0) {
    int64_t stride = 1;
    for (int32_t dim = shape.size() - 1; dim >= 0; --dim) {
      strides[dim] = stride;
      stride *= shape[dim];
    }
  }
  return std::move(strides);
}

void Array::random(int32_t gen_code)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_RAND);

  auto p_lhs = task->declare_partition();

  task->add_output(store_, p_lhs);
  task->add_scalar_arg(legate::Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  task->add_scalar_arg(legate::Scalar(runtime->get_next_random_epoch()));
  auto strides = compute_strides(shape());
  task->add_scalar_arg(legate::Scalar(strides));

  runtime->submit(std::move(task));
}

void Array::fill(const Scalar& value, bool argval)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto fill_value = runtime->create_scalar_store(value);

  auto task         = runtime->create_task(CuNumericOpCode::CUNUMERIC_FILL);
  auto p_lhs        = task->declare_partition();
  auto p_fill_value = task->declare_partition();

  task->add_output(store_, p_lhs);
  task->add_input(fill_value, p_fill_value);
  task->add_scalar_arg(legate::Scalar(argval));

  runtime->submit(std::move(task));
}

void Array::binary_op(int32_t op_code, Array rhs1, Array rhs2)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_BINARY_OP);

  auto p_lhs  = task->declare_partition();
  auto p_rhs1 = task->declare_partition();
  auto p_rhs2 = task->declare_partition();

  task->add_output(store_, p_lhs);
  task->add_input(rhs1.store_, p_rhs1);
  task->add_input(rhs2.store_, p_rhs2);
  task->add_scalar_arg(legate::Scalar(op_code));

  task->add_constraint(align(p_lhs, p_rhs1));
  task->add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void Array::unary_op(int32_t op_code, Array input)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_UNARY_OP);

  auto p_out = task->declare_partition();
  auto p_in  = task->declare_partition();

  task->add_output(store_, p_out);
  task->add_input(input.store_, p_in);
  task->add_scalar_arg(legate::Scalar(op_code));

  task->add_constraint(align(p_out, p_in));

  runtime->submit(std::move(task));
}

void Array::unary_reduction(int32_t op_code_, Array input)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto op_code = static_cast<UnaryRedCode>(op_code_);

  auto identity = runtime->get_reduction_identity(op_code, code());
  fill(identity, false);

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_SCALAR_UNARY_RED);

  auto p_out = task->declare_partition();
  auto p_in  = task->declare_partition();

  auto redop = runtime->get_reduction_op(op_code, code());

  task->add_reduction(store_, redop, p_out);
  task->add_input(input.store_, p_in);
  task->add_scalar_arg(legate::Scalar(op_code_));
  task->add_scalar_arg(legate::Scalar(input.shape()));

  runtime->submit(std::move(task));
}

void Array::dot(Array rhs1, Array rhs2)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto identity = runtime->get_reduction_identity(UnaryRedCode::SUM, code());
  fill(identity, false);

  assert(dim() == 2 && rhs1.dim() == 2 && rhs2.dim() == 2);

  auto m = rhs1.shape()[0];
  auto n = rhs2.shape()[1];
  auto k = rhs1.shape()[1];

  auto lhs_s  = store_.promote(1, k);
  auto rhs1_s = rhs1.store_.promote(2, n);
  auto rhs2_s = rhs2.store_.promote(0, m);

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_MATMUL);

  auto p_lhs  = task->declare_partition();
  auto p_rhs1 = task->declare_partition();
  auto p_rhs2 = task->declare_partition();

  auto redop = LEGION_REDOP_BASE + LEGION_TYPE_TOTAL * LEGION_REDOP_KIND_SUM + code();

  task->add_reduction(lhs_s, redop, p_lhs);
  task->add_input(rhs1_s, p_rhs1);
  task->add_input(rhs2_s, p_rhs2);

  task->add_constraint(align(p_lhs, p_rhs1));
  task->add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

/*static*/ legate::LibraryContext* Array::get_context()
{
  return CuNumericRuntime::get_runtime()->get_context();
}

}  // namespace cunumeric
