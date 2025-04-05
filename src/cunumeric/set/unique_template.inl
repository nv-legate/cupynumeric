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
#include "cunumeric/set/unique.h"
#include "cunumeric/pitches.h"
#include "cunumeric/set/zip_indices.h"

namespace cunumeric {

using namespace legate;

template <typename VAL>
struct IndexEquality {
  bool operator()(const ZippedIndex<VAL>& a, const ZippedIndex<VAL>& b) const
  {
    return a.value < b.value;
  }
};

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct UniqueImplBody;

template <VariantKind KIND>
struct UniqueImpl {
  template <Type::Code CODE, int32_t DIM>
  void operator()(std::vector<Array>& outputs,
                  Array& input,
                  std::vector<comm::Communicator>& comms,
                  const DomainPoint& point,
                  const Domain& launch_domain,
                  const bool return_index,
                  std::vector<int>& parent_extents) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = input.shape<DIM>();
    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    Point<DIM> parent_point;
    if (return_index) {
      for (int i = 0; i < DIM; i++) { parent_point[i] = parent_extents[i]; }
    }

    auto in = input.read_accessor<VAL, DIM>(rect);
    UniqueImplBody<KIND, CODE, DIM>()(
      outputs, in, pitches, rect, volume, comms, point, launch_domain, return_index, parent_point);
  }
};

template <VariantKind KIND>
static void unique_template(TaskContext& context)
{
  auto& input       = context.inputs()[0];
  auto& outputs     = context.outputs();
  auto& comms       = context.communicators();
  bool return_index = context.scalars()[0].value<bool>();
  if (outputs.size() > 1) { assert(return_index); }
  std::vector<int> parent_extents(input.dim());
  if (return_index) {
    for (int i = 0; i < parent_extents.size(); i++) {
      parent_extents[i] = context.scalars()[1 + i].value<int>();
    }
  }

  double_dispatch(input.dim(),
                  input.code(),
                  UniqueImpl<KIND>{},
                  outputs,
                  input,
                  comms,
                  context.get_task_index(),
                  context.get_launch_domain(),
                  return_index,
                  parent_extents);
}

}  // namespace cunumeric
