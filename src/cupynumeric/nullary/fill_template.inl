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
#include "cupynumeric/nullary/fill.h"
#include "cupynumeric/arg.h"
#include "cupynumeric/arg.inl"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct FillImplBody;

template <VariantKind KIND>
struct FillImpl {
  TaskContext context;
  explicit FillImpl(TaskContext context) : context(context) {}

  template <typename VAL, int DIM>
  void fill(FillArgs& args) const
  {
    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto out        = args.out.write_accessor<VAL, DIM>(rect);
    auto fill_value = args.fill_value.read_accessor<VAL, 1>();

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif
    FillImplBody<KIND, VAL, DIM>{context}(out, fill_value, pitches, rect, dense);
  }

  template <Type::Code CODE, int DIM>
  void operator()(FillArgs& args) const
  {
    using VAL = type_of<CODE>;
    fill<VAL, DIM>(args);
  }
};

template <VariantKind KIND>
static void fill_template(TaskContext& context)
{
  FillArgs args{context.output(0), context.input(0)};
  double_dispatch(
    std::max<int32_t>(args.out.dim(), 1), args.out.code(), FillImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
