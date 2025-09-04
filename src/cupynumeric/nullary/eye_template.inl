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
#include "cupynumeric/nullary/eye.h"
#include "cupynumeric/arg.h"
#include "cupynumeric/arg.inl"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL>
struct EyeImplBody;

template <VariantKind KIND>
struct EyeImpl {
  TaskContext context;
  explicit EyeImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(EyeArgs& args) const
  {
    using VAL = type_of<CODE>;

    const auto rect = args.out.shape<2>();
    auto out        = args.out.write_accessor<VAL, 2>();
    const auto k    = args.k;

    // Solve for the start
    // y = x + k
    // x >= rect.lo[0]
    const Point<2> start1(rect.lo[0], rect.lo[0] + k);
    // y >= rect.lo[1]
    const Point<2> start2(rect.lo[1] - k, rect.lo[1]);
    // If we don't have a start point then there's nothing for us to do
    if (!rect.contains(start1) && !rect.contains(start2)) {
      return;
    }
    // Pick whichever one fits in our rect
    const Point<2> start = rect.contains(start1) ? start1 : start2;
    // Now do the same thing for the end
    // x <= rect.hi[0]
    const Point<2> stop1(rect.hi[0], rect.hi[0] + k);
    // y <= rect.hi[1]
    const Point<2> stop2(rect.hi[1] - k, rect.hi[1]);
    assert(rect.contains(stop1) || rect.contains(stop2));
    const Point<2> stop = rect.contains(stop1) ? stop1 : stop2;
    // Walk the path from the stop to the start
    const coord_t distance = (stop[0] - start[0]) + 1;
    // Should be the same along both dimensions
    assert(distance == ((stop[1] - start[1]) + 1));

    EyeImplBody<KIND, VAL>{context}(out, start, distance);
  }
};

template <VariantKind KIND>
static void eye_template(TaskContext& context)
{
  EyeArgs args{context.output(0), context.scalar(0).value<int32_t>()};
  type_dispatch(args.out.code(), EyeImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
