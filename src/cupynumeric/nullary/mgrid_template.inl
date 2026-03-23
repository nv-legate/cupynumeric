/* Copyright 2026 NVIDIA Corporation
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
#include "cupynumeric/nullary/mgrid.h"
#include "cupynumeric/pitches.h"

#include <array>

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct MGridImplBody;

template <VariantKind KIND>
struct MGridImpl {
  TaskContext context;
  explicit MGridImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()() const
  {
    using VAL = type_of<CODE>;

    assert(DIM > 1);

    legate::PhysicalStore out = context.output(0);
    const auto rect           = out.shape<DIM>();

    assert(rect.hi[0] - rect.lo[0] + 1 == DIM - 1);

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    std::array<VAL, DIM - 1> starts;
    std::array<VAL, DIM - 1> steps;

    for (int dim = 0; dim < DIM - 1; ++dim) {
      starts[dim] = context.scalar(2 * dim).template value<VAL>();
      steps[dim]  = context.scalar(2 * dim + 1).template value<VAL>();
    }

    auto out_acc = out.write_accessor<VAL, DIM>(rect);

    MGridImplBody<KIND, VAL, DIM>{context}(out_acc, pitches, rect, starts, steps);
  }
};

template <VariantKind KIND>
static void mgrid_template(TaskContext& context)
{
  auto&& out = context.output(0);

  double_dispatch(out.dim(), out.type().code(), MGridImpl<KIND>{context});
}

}  // namespace cupynumeric
