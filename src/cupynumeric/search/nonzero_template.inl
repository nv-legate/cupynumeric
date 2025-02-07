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
#include "cupynumeric/search/nonzero.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct NonzeroImplBody;

template <VariantKind KIND>
struct NonzeroImpl {
  template <Type::Code CODE, int32_t DIM>
  void operator()(NonzeroArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto rect = args.input.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      for (auto& store : args.results) {
        store.bind_empty_data();
      }
      return;
    }

    auto in = args.input.read_accessor<VAL, DIM>(rect);
    NonzeroImplBody<KIND, CODE, DIM>()(args.results, in, pitches, rect, volume);
  }
};

template <VariantKind KIND>
static void nonzero_template(TaskContext& context)
{
  std::vector<legate::PhysicalStore> outputs;
  for (auto& output : context.outputs()) {
    outputs.emplace_back(output);
  }
  NonzeroArgs args{context.input(0), std::move(outputs)};
  double_dispatch(args.input.dim(), args.input.code(), NonzeroImpl<KIND>{}, args);
}

}  // namespace cupynumeric
