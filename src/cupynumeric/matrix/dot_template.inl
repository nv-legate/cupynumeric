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
#include "cupynumeric/matrix/dot.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct DotImplBody;

template <typename VAL>
struct AccTypeOf {
  using type = VAL;
};

template <>
struct AccTypeOf<__half> {
  using type = float;
};

template <typename VAL>
using acc_type_of = typename AccTypeOf<VAL>::type;

template <VariantKind KIND>
struct DotImpl {
  TaskContext context;
  explicit DotImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(DotArgs& args) const
  {
    using VAL = type_of<CODE>;
    using ACC = acc_type_of<VAL>;

    assert(args.rhs1.dim() == 1);
    assert(args.rhs2.dim() == 1);

    auto rect = args.rhs1.shape<1>();
    auto lhs  = args.lhs.reduce_accessor<SumReduction<ACC>, true, 1>();
    auto rhs1 = args.rhs1.read_accessor<VAL, 1>(rect);
    auto rhs2 = args.rhs2.read_accessor<VAL, 1>(rect);

    if (rect.empty()) {
      return;
    }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = rhs1.accessor.is_dense_row_major(rect) && rhs2.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    DotImplBody<KIND, CODE>{context}(lhs, rhs1, rhs2, rect, dense);
  }
};

template <VariantKind KIND>
static void dot_template(TaskContext& context)
{
  auto inputs = context.inputs();
  DotArgs args{context.reduction(0), inputs[0], inputs[1]};
  type_dispatch(args.rhs1.code(), DotImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
