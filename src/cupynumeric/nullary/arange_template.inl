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
#include "cupynumeric/nullary/arange.h"
#include "cupynumeric/arg.h"
#include "cupynumeric/arg.inl"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL>
struct ArangeImplBody;

template <VariantKind KIND>
struct ArangeImpl {
  template <Type::Code CODE>
  void operator()(ArangeArgs& args) const
  {
    using VAL = type_of<CODE>;

    const auto rect = args.out.shape<1>();

    if (rect.empty()) {
      return;
    }

    auto out = args.out.write_accessor<VAL, 1>();

    const auto start = args.start.value<VAL>();
    const auto step  = args.step.value<VAL>();

    ArangeImplBody<KIND, VAL>{}(out, rect, start, step);
  }
};

template <VariantKind KIND>
static void arange_template(TaskContext& context)
{
  ArangeArgs args{context.output(0), context.scalar(0), context.scalar(1)};
  type_dispatch(args.out.code(), ArangeImpl<KIND>{}, args);
}

}  // namespace cupynumeric
