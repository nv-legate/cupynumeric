/* Copyright 2026 NVIDIA Corporation
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
#include "cupynumeric/matrix/vander.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

// Sequential multiplication (base^exp) so that floating-point rounding matches
// NumPy's multiply.accumulate implementation of vander bit-for-bit. Works for
// integer, floating and complex value types on both host and device.
template <typename VAL>
__CUDA_HD__ inline VAL vander_pow(VAL base, int64_t exp)
{
  VAL result = static_cast<VAL>(1);
  for (int64_t k = 0; k < exp; ++k) {
    result = static_cast<VAL>(result * base);
  }
  return result;
}

template <VariantKind KIND, Type::Code CODE>
struct VanderImplBody;

template <VariantKind KIND>
struct VanderImpl {
  TaskContext context;
  explicit VanderImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(VanderArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto shape = args.output.shape<2>();
    if (shape.empty()) {
      return;
    }

    auto out = args.output.write_accessor<VAL, 2>(shape);
    // The input is a 1-D vector that has been promoted to (M, N) so that
    // element (i, j) reads back x[i] regardless of how columns are partitioned.
    auto in = args.input.read_accessor<VAL, 2>(shape);

    Pitches<1> pitches{};
    size_t volume = pitches.flatten(shape);

    VanderImplBody<KIND, CODE>{context}(
      out, in, pitches, shape.lo, volume, args.N, args.increasing);
  }
};

template <VariantKind KIND>
static void vander_template(TaskContext& context)
{
  auto increasing = context.scalar(0).value<bool>();
  auto N          = context.scalar(1).value<int64_t>();
  auto input      = context.input(0);
  auto output     = context.output(0);
  VanderArgs args{increasing, N, output, input};
  type_dispatch(args.output.type().code(), VanderImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
