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
#include <legate/utilities/typedefs.h>
#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/unary/scalar_unary_red.h"
#include "cupynumeric/unary/unary_red_util.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/execution_policy/reduction/scalar_reduction.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, UnaryRedCode OP_CODE, Type::Code CODE, int DIM, bool HAS_WHERE>
struct ScalarUnaryRed {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using LHS   = typename OP::VAL;
  using RHS   = type_of<CODE>;
  using OUT   = AccessorRD<LG_OP, true, 1>;
  using IN    = AccessorRO<RHS, DIM>;
  using WHERE = AccessorRO<bool, DIM>;

  IN in;
  const RHS* inptr;
  OUT out;
  size_t volume;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  Point<DIM> origin;
  Point<DIM> shape;
  RHS to_find;
  RHS mu;
  bool dense;
  WHERE where;
  const bool* whereptr;
  TaskContext context;

  struct DenseReduction {};
  struct SparseReduction {};

  ScalarUnaryRed(ScalarUnaryRedArgs& args, TaskContext tcontext) : dense(false), context(tcontext)
  {
    rect   = args.in.shape<DIM>();
    origin = rect.lo;
    in     = args.in.read_accessor<RHS, DIM>(rect);
    volume = pitches.flatten(rect);
    shape  = args.shape;

    out = args.out.reduce_accessor<LG_OP, true, 1>();
    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      to_find = args.args[0].value<RHS>();
    }
    if constexpr (OP_CODE == UnaryRedCode::VARIANCE) {
      mu = args.args[0].value<RHS>();
    }

    if constexpr (HAS_WHERE) {
      where = args.where.read_accessor<bool, DIM>(rect);
    }
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    if (in.accessor.is_dense_row_major(rect)) {
      dense = true;
      inptr = in.ptr(rect);
    }
    if constexpr (HAS_WHERE) {
      dense = dense && where.accessor.is_dense_row_major(rect);
      if (dense) {
        whereptr = where.ptr(rect);
      }
    }
#endif
  }

  __CUDA_HD__ void operator()(LHS& lhs, size_t idx, LHS identity, DenseReduction) const noexcept
  {
    bool mask = true;
    if constexpr (HAS_WHERE) {
      mask = whereptr[idx];
    }

    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      if (mask && (inptr[idx] == to_find)) {
        lhs = true;
      }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN ||
                         OP_CODE == UnaryRedCode::NANARGMAX || OP_CODE == UnaryRedCode::NANARGMIN) {
      auto p = pitches.unflatten(idx, origin);
      if (mask) {
        OP::template fold<true>(lhs, OP::convert(p, shape, identity, inptr[idx]));
      }
    } else if constexpr (OP_CODE == UnaryRedCode::VARIANCE) {
      if (mask) {
        OP::template fold<true>(lhs, OP::convert(inptr[idx] - mu, identity));
      }
    } else {
      if (mask) {
        OP::template fold<true>(lhs, OP::convert(inptr[idx], identity));
      }
    }
  }

  __CUDA_HD__ void operator()(LHS& lhs, size_t idx, LHS identity, SparseReduction) const noexcept
  {
    auto p    = pitches.unflatten(idx, origin);
    bool mask = true;
    if constexpr (HAS_WHERE) {
      mask = where[p];
    }

    if constexpr (OP_CODE == UnaryRedCode::CONTAINS) {
      if (mask && (in[p] == to_find)) {
        lhs = true;
      }
    } else if constexpr (OP_CODE == UnaryRedCode::ARGMAX || OP_CODE == UnaryRedCode::ARGMIN ||
                         OP_CODE == UnaryRedCode::NANARGMAX || OP_CODE == UnaryRedCode::NANARGMIN) {
      if (mask) {
        OP::template fold<true>(lhs, OP::convert(p, shape, identity, in[p]));
      }
    } else if constexpr (OP_CODE == UnaryRedCode::VARIANCE) {
      if (mask) {
        OP::template fold<true>(lhs, OP::convert(in[p] - mu, identity));
      }
    } else {
      if (mask) {
        OP::template fold<true>(lhs, OP::convert(in[p], identity));
      }
    }
  }

  void execute() const noexcept
  {
    auto identity = LG_OP::identity;
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // The constexpr if here prevents the DenseReduction from being instantiated for GPU kernels
    // which limits compile times and binary sizes.
    if constexpr (KIND != VariantKind::GPU) {
      // Check to see if this is dense or not
      if (dense) {
        return ScalarReductionPolicy<KIND, LG_OP, DenseReduction>{context}(
          volume, out, identity, *this);
      }
    }
#endif
    return ScalarReductionPolicy<KIND, LG_OP, SparseReduction>{context}(
      volume, out, identity, *this);
  }
};

template <VariantKind KIND, UnaryRedCode OP_CODE, bool HAS_WHERE>
struct ScalarUnaryRedImpl {
  TaskContext context;
  explicit ScalarUnaryRedImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(ScalarUnaryRedArgs& args) const
  {
    // The operation is always valid for contains
    if constexpr (UnaryRedOp<OP_CODE, CODE>::valid || OP_CODE == UnaryRedCode::CONTAINS) {
      ScalarUnaryRed<KIND, OP_CODE, CODE, DIM, HAS_WHERE> red(args, context);
      red.execute();
    }
  }
};

template <VariantKind KIND>
struct ScalarUnaryRedDispatch {
  TaskContext context;
  explicit ScalarUnaryRedDispatch(TaskContext context) : context(context) {}

  template <UnaryRedCode OP_CODE>
  void operator()(ScalarUnaryRedArgs& args, bool has_where) const
  {
    auto dim = std::max(1, args.in.dim());
    if (has_where) {
      double_dispatch(dim, args.in.code(), ScalarUnaryRedImpl<KIND, OP_CODE, true>{context}, args);
    } else {
      double_dispatch(dim, args.in.code(), ScalarUnaryRedImpl<KIND, OP_CODE, false>{context}, args);
    }
  }
};

template <VariantKind KIND>
static void scalar_unary_red_template(TaskContext& context)
{
  const auto num_scalars = context.num_scalars();
  auto op_code           = context.scalar(0).value<UnaryRedCode>();
  auto shape             = context.scalar(1).value<DomainPoint>();
  bool has_where         = context.scalar(2).value<bool>();

  std::vector<Scalar> extra_args;

  extra_args.reserve(num_scalars - 3);
  for (size_t idx = 3; idx < num_scalars; ++idx) {
    extra_args.emplace_back(context.scalar(idx));
  }

  // If the RHS was a scalar, use (1,) as the shape
  if (shape.dim == 0) {
    shape.dim = 1;
    shape[0]  = 1;
  }

  ScalarUnaryRedArgs args{context.reduction(0),
                          context.input(0),
                          has_where ? context.input(1) : PhysicalStore{nullptr},
                          op_code,
                          shape,
                          std::move(extra_args)};
  op_dispatch(args.op_code, ScalarUnaryRedDispatch<KIND>{context}, args, has_where);
}

}  // namespace cupynumeric
