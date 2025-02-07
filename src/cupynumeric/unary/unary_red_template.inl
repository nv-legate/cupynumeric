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
#include "cupynumeric/unary/unary_red.h"
#include "cupynumeric/unary/unary_red_util.h"
#include "cupynumeric/arg.h"
#include "cupynumeric/arg.inl"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, UnaryRedCode OP_CODE, Type::Code CODE, int DIM, bool HAS_WHERE>
struct UnaryRedImplBody;

template <VariantKind KIND, UnaryRedCode OP_CODE, bool HAS_WHERE>
struct UnaryRedImpl {
  template <Type::Code CODE,
            int DIM,
            std::enable_if_t<(DIM > 1) && UnaryRedOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(UnaryRedArgs& args) const
  {
    using OP  = UnaryRedOp<OP_CODE, CODE>;
    using RHS = type_of<CODE>;

    Pitches<DIM - 1> pitches;
    auto rect   = args.rhs.shape<DIM>();
    auto volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto rhs = args.rhs.read_accessor<RHS, DIM>(rect);

    auto lhs = args.lhs.reduce_accessor<typename OP::OP, KIND != VariantKind::GPU, DIM>(rect);

    AccessorRO<bool, DIM> where;
    if constexpr (HAS_WHERE) {
      where = args.where.read_accessor<bool, DIM>(rect);
    }
    UnaryRedImplBody<KIND, OP_CODE, CODE, DIM, HAS_WHERE>()(
      lhs, rhs, where, rect, pitches, args.collapsed_dim, volume);
  }

  template <Type::Code CODE,
            int DIM,
            std::enable_if_t<DIM <= 1 || !UnaryRedOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(UnaryRedArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND, bool HAS_WHERE>
struct UnaryRedDispatch {
  template <UnaryRedCode OP_CODE>
  void operator()(UnaryRedArgs& args) const
  {
    auto dim = std::max(1, args.rhs.dim());
    return double_dispatch(dim, args.rhs.code(), UnaryRedImpl<KIND, OP_CODE, HAS_WHERE>{}, args);
  }
};

template <VariantKind KIND>
static void unary_red_template(TaskContext& context)
{
  bool has_where = context.scalar(2).value<bool>();
  UnaryRedArgs args{context.reduction(0),
                    context.input(0),
                    has_where ? context.input(1) : legate::PhysicalStore{nullptr},
                    context.scalar(0).value<int32_t>(),
                    context.scalar(1).value<UnaryRedCode>()};
  if (has_where) {
    op_dispatch(args.op_code, UnaryRedDispatch<KIND, true>{}, args);
  } else {
    op_dispatch(args.op_code, UnaryRedDispatch<KIND, false>{}, args);
  }
}

}  // namespace cupynumeric
