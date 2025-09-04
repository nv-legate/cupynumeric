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
#include "cupynumeric/index/select.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM>
struct SelectImplBody;

template <VariantKind KIND>
struct SelectImpl {
  TaskContext context;
  explicit SelectImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(SelectArgs& args) const
  {
    using VAL     = type_of<CODE>;
    auto out_rect = args.out.shape<DIM>();

    // working set is the intersection of the shapes
    for (const auto& input : args.inputs) {
      out_rect = out_rect.intersection(input.shape<DIM>());
    }

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(out_rect);
    if (volume == 0) {
      return;
    }

    auto out = args.out.data().write_accessor<VAL, DIM>(out_rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(out_rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    std::vector<AccessorRO<bool, DIM>> condlist;
    condlist.reserve(args.inputs.size() / 2);
    for (int32_t i = 0; i < args.inputs.size() / 2; i++) {
      condlist.push_back(args.inputs[i].data().read_accessor<bool, DIM>(out_rect));
      dense = dense && condlist.back().accessor.is_dense_row_major(out_rect);
    }

    std::vector<AccessorRO<VAL, DIM>> choicelist;
    choicelist.reserve(args.inputs.size() / 2);
    for (int32_t i = args.inputs.size() / 2; i < args.inputs.size(); i++) {
      choicelist.push_back(args.inputs[i].data().read_accessor<VAL, DIM>(out_rect));
      dense = dense && choicelist.back().accessor.is_dense_row_major(out_rect);
    }

    VAL default_value = args.default_value.value<VAL>();
    SelectImplBody<KIND, CODE, DIM>{context}(
      out, condlist, choicelist, default_value, out_rect, pitches, dense);
  }
};

template <VariantKind KIND>
static void select_template(TaskContext& context)
{
  SelectArgs args{context.output(0), context.inputs(), context.scalar(0)};
  double_dispatch(args.out.dim(), args.out.type().code(), SelectImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
