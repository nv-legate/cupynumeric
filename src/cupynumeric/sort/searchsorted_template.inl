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
#include "cupynumeric/sort/searchsorted.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct SearchSortedImplBody;

template <VariantKind KIND>
struct SearchSortedImpl {
  template <Type::Code CODE, int32_t DIM>
  void operator()(SearchSortedArgs& args) const
  {
    auto rect_base       = args.input_base.shape<1>();
    auto rect_values_in  = args.input_values.shape<DIM>();
    auto rect_values_out = args.output_reduction.shape<DIM>();

    if (rect_base.empty()) {
      return;
    }
    if (rect_values_in.empty()) {
      return;
    }

    Pitches<0> pitches_base;
    Pitches<DIM - 1> pitches_values;
    size_t volume     = pitches_base.flatten(rect_base);
    size_t num_values = pitches_values.flatten(rect_values_in);
    assert(rect_values_in == rect_values_out);
    assert(num_values == pitches_values.flatten(rect_values_out));

    SearchSortedImplBody<KIND, CODE, DIM>()(args.input_base,
                                            args.input_values,
                                            args.output_reduction,
                                            rect_base,
                                            rect_values_in,
                                            pitches_values,
                                            args.left,
                                            args.is_index_space,
                                            volume,
                                            args.global_volume,
                                            num_values);
  }
};

template <VariantKind KIND>
static void searchsorted_template(TaskContext& context)
{
  SearchSortedArgs args{context.input(0),
                        context.input(1),
                        context.reduction(0),
                        context.scalar(0).value<bool>(),
                        context.scalar(1).value<int64_t>(),
                        !context.is_single_task()};

  assert(args.input_base.dim() == 1);
  assert(args.input_base.code() == args.input_values.code());
  assert(args.input_values.dim() == args.output_reduction.dim());

  double_dispatch(
    std::max(1, args.input_values.dim()), args.input_base.code(), SearchSortedImpl<KIND>{}, args);
}

}  // namespace cupynumeric
