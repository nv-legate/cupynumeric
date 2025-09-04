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
#include "cupynumeric/ternary/where.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM>
struct WhereImplBody;

template <VariantKind KIND>
struct WhereImpl {
  TaskContext context;
  explicit WhereImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(WhereArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto out  = args.out.write_accessor<VAL, DIM>(rect);
    auto mask = args.mask.read_accessor<bool, DIM>(rect);
    auto in1  = args.in1.read_accessor<VAL, DIM>(rect);
    auto in2  = args.in2.read_accessor<VAL, DIM>(rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect) && mask.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    WhereImplBody<KIND, CODE, DIM>{context}(out, mask, in1, in2, pitches, rect, dense);
  }
};

template <VariantKind KIND>
static void where_template(TaskContext& context)
{
  auto inputs = context.inputs();
  WhereArgs args{context.output(0), inputs[0], inputs[1], inputs[2]};
  auto dim = std::max(1, args.out.dim());
  double_dispatch(dim, args.out.code(), WhereImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
