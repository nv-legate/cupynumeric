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
#include "cupynumeric/bits/unpackbits.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, int32_t DIM, Bitorder BITORDER>
struct UnpackbitsImplBody;

template <VariantKind KIND, Bitorder BITORDER>
struct UnpackbitsImpl {
  TaskContext context;
  explicit UnpackbitsImpl(TaskContext context) : context(context) {}

  template <int32_t DIM>
  void operator()(PhysicalStore output, PhysicalStore input, uint32_t axis) const
  {
    auto out_rect = output.shape<DIM>();

    if (out_rect.empty()) {
      return;
    }

    auto in_rect = input.shape<DIM>();

    auto out = output.write_accessor<uint8_t, DIM>(out_rect);
    auto in  = input.read_accessor<uint8_t, DIM>(in_rect);

    Pitches<DIM - 1> in_pitches{};
    auto in_volume = in_pitches.flatten(in_rect);

    UnpackbitsImplBody<KIND, DIM, BITORDER>{context}(out, in, in_rect, in_pitches, in_volume, axis);
  }

  template <Type::Code CODE, int32_t DIM, std::enable_if_t<!is_integral<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore output, legate::PhysicalStore input, uint32_t axis) const
  {
    // Unreachable
    assert(false);
  }
};

template <VariantKind KIND>
static void unpackbits_template(TaskContext& context)
{
  legate::PhysicalStore output = context.output(0);
  legate::PhysicalStore input  = context.input(0);
  auto axis                    = context.scalar(0).value<uint32_t>();
  auto bitorder                = context.scalar(1).value<Bitorder>();

  switch (bitorder) {
    case Bitorder::BIG: {
      dim_dispatch(input.dim(), UnpackbitsImpl<KIND, Bitorder::BIG>{context}, output, input, axis);
      break;
    }
    case Bitorder::LITTLE: {
      dim_dispatch(
        input.dim(), UnpackbitsImpl<KIND, Bitorder::LITTLE>{context}, output, input, axis);
      break;
    }
  }
}

}  // namespace cupynumeric
