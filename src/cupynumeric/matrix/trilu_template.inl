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
#include "cupynumeric/matrix/trilu.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int32_t DIM, bool LOWER>
struct TriluImplBody;

template <VariantKind KIND>
struct TriluImpl {
  template <Type::Code CODE, int32_t DIM, std::enable_if_t<(DIM >= 2)>* = nullptr>
  void operator()(TriluArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto shape = args.output.shape<DIM>();
    if (shape.empty()) {
      return;
    }

    auto out = args.output.write_accessor<VAL, DIM>(shape);
    auto in  = args.input.read_accessor<VAL, DIM>(shape);

    if (out.accessor.is_dense_col_major(shape)) {
      Pitches<DIM - 1, false /*C_ORDER*/> pitches{};
      size_t volume = pitches.flatten(shape);

      if (args.lower) {
        TriluImplBody<KIND, CODE, DIM, true>()(out, in, pitches, shape.lo, volume, args.k);
      } else {
        TriluImplBody<KIND, CODE, DIM, false>()(out, in, pitches, shape.lo, volume, args.k);
      }
    } else {
      Pitches<DIM - 1> pitches{};
      size_t volume = pitches.flatten(shape);

      if (args.lower) {
        TriluImplBody<KIND, CODE, DIM, true>()(out, in, pitches, shape.lo, volume, args.k);
      } else {
        TriluImplBody<KIND, CODE, DIM, false>()(out, in, pitches, shape.lo, volume, args.k);
      }
    }
  }

  template <Type::Code CODE, int32_t DIM, std::enable_if_t<(DIM < 2)>* = nullptr>
  void operator()(TriluArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void trilu_template(TaskContext& context)
{
  auto lower  = context.scalar(0).value<bool>();
  auto k      = context.scalar(1).value<int32_t>();
  auto input  = context.input(0);
  auto output = context.output(0);
  TriluArgs args{lower, k, output, input};
  double_dispatch(args.output.dim(), args.output.type().code(), TriluImpl<KIND>{}, args);
}

}  // namespace cupynumeric
