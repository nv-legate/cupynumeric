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
#include "cupynumeric/item/write.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct WriteImplBody;

template <VariantKind KIND>
struct WriteImpl {
  TaskContext context;
  explicit WriteImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(legate::PhysicalStore out_arr, legate::PhysicalStore in_arr) const
  {
    using VAL = type_of<CODE>;
    auto out  = out_arr.write_accessor<VAL, 1>();
    auto in   = in_arr.read_accessor<VAL, DIM>();
    WriteImplBody<KIND, VAL, DIM>{context}(out, in);
  }
};

template <VariantKind KIND>
static void write_template(TaskContext& context)
{
  auto in  = context.input(0);
  auto out = context.output(0);
  auto dim = std::max(1, in.dim());
  legate::double_dispatch(dim, out.type().code(), WriteImpl<KIND>{context}, out, in);
}

}  // namespace cupynumeric
