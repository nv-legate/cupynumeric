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
#include "cupynumeric/set/unique.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct UniqueImplBody;

template <VariantKind KIND>
struct UniqueImpl {
  template <Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore output,
                  legate::PhysicalStore input,
                  std::vector<comm::Communicator>& comms,
                  const DomainPoint& point,
                  const Domain& launch_domain) const
  {
    using VAL = type_of<CODE>;

    auto rect = input.shape<DIM>();
    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    auto in = input.read_accessor<VAL, DIM>(rect);
    UniqueImplBody<KIND, CODE, DIM>()(
      output, in, pitches, rect, volume, comms, point, launch_domain);
  }
};

template <VariantKind KIND>
static void unique_template(TaskContext& context)
{
  auto input  = context.input(0);
  auto output = context.output(0);
  auto comms  = context.communicators();
  double_dispatch(input.dim(),
                  input.type().code(),
                  UniqueImpl<KIND>{},
                  output,
                  input,
                  comms,
                  context.get_task_index(),
                  context.get_launch_domain());
}

}  // namespace cupynumeric
