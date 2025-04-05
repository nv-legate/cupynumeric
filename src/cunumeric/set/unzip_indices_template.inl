/* Copyright 2022 NVIDIA Corporation
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
#include "cunumeric/set/unzip_indices.h"
#include "cunumeric/pitches.h"
#include "cunumeric/set/zip_indices.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct UniqueImplBody;

template <VariantKind KIND>
struct UnzipIndicesImpl {
  template <Type::Code CODE>
  void operator()(std::vector<Array>& outputs, Array& input)
  {
    using VAL = legate_type_of<CODE>;

    auto input_shape = input.shape<1>();
    if (input_shape.volume() == 0) return;

    auto values  = outputs[0].write_accessor<VAL, 1>(input_shape);
    auto indices = outputs[1].write_accessor<int64_t, 1>(input_shape);
    auto in      = input.read_accessor<ZippedIndex<VAL>, 1>(input_shape);

    UniqueImplBody<KIND, CODE>()(values, indices, in, input_shape);
  }
};

template <VariantKind KIND>
static void unzip_indices_template(TaskContext& context)
{
  auto& input   = context.inputs()[0];
  auto& outputs = context.outputs();

  Type::Code code{input.code()};
  assert(Type::Code::STRUCT == code);
  auto& field_type = static_cast<const StructType&>(input.type()).field_type(0);
  code             = field_type.code;

  type_dispatch(code, UnzipIndicesImpl<KIND>{}, outputs, input);
}

}  // namespace cunumeric
