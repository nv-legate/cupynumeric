/* Copyright 2025 NVIDIA Corporation
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
#include "cupynumeric/set/in1d.h"
#include <thrust/fill.h>

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct In1dImplBody;

template <VariantKind KIND>
struct In1dImpl {
  TaskContext context;
  explicit In1dImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(legate::PhysicalStore output,
                  legate::PhysicalStore input1,
                  legate::PhysicalStore input2,
                  bool assume_unique,
                  bool invert,
                  const std::string& kind,
                  int64_t ar2_min,
                  int64_t ar2_max) const
  {
    using VAL = type_of<CODE>;

    auto rect1     = input1.shape<1>();
    auto rect2     = input2.shape<1>();
    size_t volume1 = rect1.volume();
    size_t volume2 = rect2.volume();

    auto in1 = input1.read_accessor<VAL, 1>(rect1);
    auto in2 = input2.read_accessor<VAL, 1>(rect2);
    auto out = output.write_accessor<bool, 1>(rect1);

    if (volume1 == 0) {
      return;
    }

    if (volume2 == 0) {
      thrust::fill(out.ptr(0), out.ptr(0) + volume1, invert);
      return;
    }

    In1dImplBody<KIND, CODE, 1>{context}(
      out, in1, in2, rect1, rect2, volume1, volume2, assume_unique, invert, kind, ar2_min, ar2_max);
  }
};

template <VariantKind KIND>
static void in1d_template(TaskContext& context)
{
  auto input1        = context.input(0);
  auto input2        = context.input(1);
  auto output        = context.output(0);
  auto assume_unique = context.scalars().at(0).value<bool>();
  auto invert        = context.scalars().at(1).value<bool>();
  auto kind          = context.scalars().at(2).value<std::string>();
  auto ar2_min       = context.scalars().at(3).value<int64_t>();
  auto ar2_max       = context.scalars().at(4).value<int64_t>();

  type_dispatch(input1.type().code(),
                In1dImpl<KIND>{context},
                output,
                input1,
                input2,
                assume_unique,
                invert,
                kind,
                ar2_min,
                ar2_max);
}

}  // namespace cupynumeric
