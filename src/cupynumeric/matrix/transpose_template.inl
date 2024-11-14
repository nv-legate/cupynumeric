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
#include "cupynumeric/matrix/transpose.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct TransposeImplBody;

template <VariantKind KIND>
struct TransposeImpl {
  template <Type::Code CODE>
  void operator()(TransposeArgs& args) const
  {
    using VAL = type_of<CODE>;

    const auto rect = args.out.shape<2>();
    if (rect.empty()) {
      return;
    }

    auto out = args.out.write_accessor<VAL, 2>();
    auto in  = args.in.read_accessor<VAL, 2>();

    TransposeImplBody<KIND, CODE>{}(rect, out, in);
  }
};

template <VariantKind KIND>
static void transpose_template(TaskContext& context)
{
  auto output = context.output(0);
  auto input  = context.input(0);

  TransposeArgs args{output, input};
  type_dispatch(input.type().code(), TransposeImpl<KIND>{}, args);
}

}  // namespace cupynumeric
