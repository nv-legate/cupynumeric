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
#include "cupynumeric/index/advanced_indexing.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody;

template <VariantKind KIND>
struct AdvancedIndexingImpl {
  // current implementaion of the ND-output regions requires all regions
  // to have the same DIM.
  template <Type::Code CODE, int DIM>
  void operator()(AdvancedIndexingArgs& args) const
  {
    using VAL       = type_of<CODE>;
    auto input_rect = args.input_array.shape<DIM>();
    auto input_arr  = args.input_array.read_accessor<VAL, DIM>(input_rect);
    Pitches<DIM - 1> input_pitches{};
    size_t volume = input_pitches.flatten(input_rect);

    auto index_rect = args.indexing_array.shape<DIM>();
    // this task is executed only for the case when index array is a bool type
    auto index_arr = args.indexing_array.read_accessor<bool, DIM>(index_rect);
#ifdef DEBUG_CUPYNUMERIC
    // we make sure that index and input shapes are the same on the python side.
    // checking this one more time here
    assert(index_rect == input_rect);
#endif

    if (volume == 0) {
      args.output.bind_empty_data();
      return;
    }

    if (args.is_set) {
      AdvancedIndexingImplBody<KIND, CODE, DIM, Point<DIM>>{}(
        args.output, input_arr, index_arr, input_pitches, input_rect, args.key_dim);
    } else {
      AdvancedIndexingImplBody<KIND, CODE, DIM, VAL>{}(
        args.output, input_arr, index_arr, input_pitches, input_rect, args.key_dim);
    }
  }
};

template <VariantKind KIND>
static void advanced_indexing_template(TaskContext& context)
{
  // is_set flag is used to fill Point<N> field for in-place assignment operation
  bool is_set     = context.scalar(0).value<bool>();
  int64_t key_dim = context.scalar(1).value<int64_t>();
  AdvancedIndexingArgs args{context.output(0), context.input(0), context.input(1), is_set, key_dim};
  double_dispatch(
    args.input_array.dim(), args.input_array.code(), AdvancedIndexingImpl<KIND>{}, args);
}

}  // namespace cupynumeric
