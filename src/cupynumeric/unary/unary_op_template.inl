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
#include "cupynumeric/unary/unary_op.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct UnaryOpImplBody;

template <VariantKind KIND, typename VAL, int DIM>
struct PointCopyImplBody;

template <VariantKind KIND, UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct MultiOutUnaryOpImplBody;

template <VariantKind KIND, UnaryOpCode OP_CODE>
struct UnaryOpImpl {
  TaskContext context;
  explicit UnaryOpImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM, std::enable_if_t<UnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(UnaryOpArgs& args) const
  {
    using OP  = UnaryOp<OP_CODE, CODE>;
    using ARG = typename OP::T;
    using RES = std::result_of_t<OP(ARG)>;

    auto rect = args.out.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto out = args.out.write_accessor<RES, DIM>(rect);
    auto in  = args.in.read_accessor<ARG, DIM>(rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{args.args};
    UnaryOpImplBody<KIND, OP_CODE, CODE, DIM>{context}(func, out, in, pitches, rect, dense);
  }

  template <Type::Code CODE, int DIM, std::enable_if_t<!UnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(UnaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND, UnaryOpCode OP_CODE>
struct MultiOutUnaryOpImpl {
  TaskContext context;
  explicit MultiOutUnaryOpImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE,
            int DIM,
            std::enable_if_t<MultiOutUnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(MultiOutUnaryOpArgs& args) const
  {
    using OP   = MultiOutUnaryOp<OP_CODE, CODE>;
    using RHS1 = typename OP::RHS1;
    using RHS2 = typename OP::RHS2;
    using LHS  = std::result_of_t<OP(RHS1, RHS2*)>;

    auto rect = args.out1.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto lhs  = args.out1.write_accessor<LHS, DIM>(rect);
    auto rhs1 = args.in.read_accessor<RHS1, DIM>(rect);
    auto rhs2 = args.out2.write_accessor<RHS2, DIM>(rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = lhs.accessor.is_dense_row_major(rect) && rhs1.accessor.is_dense_row_major(rect) &&
                 rhs2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    MultiOutUnaryOpImplBody<KIND, OP_CODE, CODE, DIM>{context}(
      func, lhs, rhs1, rhs2, pitches, rect, dense);
  }

  template <Type::Code CODE,
            int DIM,
            std::enable_if_t<!MultiOutUnaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(MultiOutUnaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct UnaryCopyImpl {
  TaskContext context;
  explicit UnaryCopyImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(UnaryOpArgs& args) const
  {
    using VAL = type_of<CODE>;
    execute_copy<VAL, DIM>(args);
  }

  template <int POINT_DIM, int DIM>
  void operator()(UnaryOpArgs& args) const
  {
    using VAL = Point<POINT_DIM>;
    execute_copy<VAL, DIM>(args);
  }

  template <typename VAL, int DIM>
  void execute_copy(UnaryOpArgs& args) const
  {
    auto rect = args.out.shape<DIM>().intersection(args.in.shape<DIM>());

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto out = args.out.write_accessor<VAL, DIM>(rect);
    auto in  = args.in.read_accessor<VAL, DIM>(rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    PointCopyImplBody<KIND, VAL, DIM>{context}(out, in, pitches, rect, dense);
  }
};

template <VariantKind KIND>
struct UnaryOpDispatch {
  TaskContext context;
  explicit UnaryOpDispatch(TaskContext context) : context(context) {}

  template <UnaryOpCode OP_CODE>
  void operator()(UnaryOpArgs& args) const
  {
    auto dim = std::max(args.in.dim(), 1);
    if ((OP_CODE == UnaryOpCode::COPY) && (args.in.code() == Type::Code::FIXED_ARRAY)) {
      auto type = args.in.type().as_fixed_array_type();
      cupynumeric::double_dispatch(dim, type.num_elements(), UnaryCopyImpl<KIND>{context}, args);
    } else {
      auto code = OP_CODE == UnaryOpCode::GETARG ? args.out.code() : args.in.code();
      legate::double_dispatch(dim, code, UnaryOpImpl<KIND, OP_CODE>{context}, args);
    }
  }
};

template <VariantKind KIND>
static void unary_op_template(TaskContext& context)
{
  auto op_code = context.scalar(0).value<UnaryOpCode>();
  switch (op_code) {
    case UnaryOpCode::FREXP: {
      MultiOutUnaryOpArgs args{context.input(0), context.output(0), context.output(1), op_code};
      auto dim = std::max(args.in.dim(), 1);
      legate::double_dispatch(
        dim, args.in.code(), MultiOutUnaryOpImpl<KIND, UnaryOpCode::FREXP>{context}, args);
      break;
    }
    case UnaryOpCode::MODF: {
      MultiOutUnaryOpArgs args{context.input(0), context.output(0), context.output(1), op_code};
      auto dim = std::max(args.in.dim(), 1);
      legate::double_dispatch(
        dim, args.in.code(), MultiOutUnaryOpImpl<KIND, UnaryOpCode::MODF>{context}, args);
      break;
    }
    default: {
      const auto num_scalars = context.num_scalars();
      std::vector<Scalar> extra_args;

      extra_args.reserve(num_scalars - 1);
      for (size_t idx = 1; idx < num_scalars; ++idx) {
        extra_args.push_back(context.scalar(idx));
      }

      UnaryOpArgs args{context.input(0), context.output(0), op_code, std::move(extra_args)};
      op_dispatch(args.op_code, UnaryOpDispatch<KIND>{context}, args);
      break;
    }
  }
}

}  // namespace cupynumeric
