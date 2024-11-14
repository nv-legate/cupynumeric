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

// Useful for IDEs
#include "cupynumeric/binary/binary_op.h"
#include "cupynumeric/binary/binary_op_util.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryOpImplBody;

template <VariantKind KIND, BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <Type::Code CODE, int DIM, std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    using OP   = BinaryOp<OP_CODE, CODE>;
    using RHS1 = type_of<CODE>;
    using RHS2 = rhs2_of_binary_op<OP_CODE, CODE>;
    using LHS  = std::result_of_t<OP(RHS1, RHS2)>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto out = args.out.write_accessor<LHS, DIM>(rect);
    auto in1 = args.in1.read_accessor<RHS1, DIM>(rect);
    auto in2 = args.in2.read_accessor<RHS2, DIM>(rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{args.args};
    BinaryOpImplBody<KIND, OP_CODE, CODE, DIM>()(func, out, in1, in2, pitches, rect, dense);
  }

  template <Type::Code CODE, int DIM, std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(BinaryOpArgs& args) const
  {
    auto dim = std::max(1, args.out.dim());
    double_dispatch(dim, args.in1.code(), BinaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void binary_op_template(TaskContext& context)
{
  auto scalars = context.scalars();
  auto op_code = scalars.front().value<BinaryOpCode>();

  scalars.erase(scalars.begin());

  BinaryOpArgs args{
    context.input(0), context.input(1), context.output(0), op_code, std::move(scalars)};
  op_dispatch(args.op_code, BinaryOpDispatch<KIND>{}, args);
}

}  // namespace cupynumeric
