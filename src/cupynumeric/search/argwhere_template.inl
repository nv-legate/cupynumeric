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
#include "cupynumeric/search/argwhere.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM>
struct ArgWhereImplBody;

template <VariantKind KIND>
struct ArgWhereImpl {
  template <Type::Code CODE, int DIM>
  void operator()(ArgWhereArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto rect_in = args.in.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect_in);

    if (volume == 0) {
      args.out.bind_empty_data();
      return;
    }

    auto in = args.in.read_accessor<VAL, DIM>(rect_in);
    ArgWhereImplBody<KIND, CODE, DIM>()(args.out, in, pitches, rect_in, volume);
  }
};

template <VariantKind KIND>
static void argwhere_template(TaskContext& context)
{
  ArgWhereArgs args{context.output(0), context.input(0)};
  double_dispatch(args.in.dim(), args.in.code(), ArgWhereImpl<KIND>{}, args);
}

}  // namespace cupynumeric
