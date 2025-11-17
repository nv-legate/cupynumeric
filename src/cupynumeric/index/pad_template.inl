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
#include "cupynumeric/index/pad.h"
#include "cupynumeric/pitches.h"

#include <cassert>

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM>
struct PadImplBody;

template <VariantKind KIND>
struct PadImpl {
  TaskContext context;
  explicit PadImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(PadArgs& args) const
  {
    using VAL        = type_of<CODE>;
    auto output_rect = args.output.shape<DIM>();

    if (output_rect.empty()) {
      return;
    }

    // Output is accessed as read-write (input already copied to center)
    auto output_arr = args.output.read_write_accessor<VAL, DIM>(output_rect);

    // Create accessor for constant value (if CONSTANT mode)
    if (args.mode == PadMode::CONSTANT) {
      auto const_rect = args.constant_input.shape<1>();
      auto const_acc  = args.constant_input.read_accessor<VAL, 1>(const_rect);
      PadImplBody<KIND, CODE, DIM>{context}(output_arr,
                                            args.mode,
                                            args.pad_width,
                                            args.inner_shape,
                                            args.constant_rows,
                                            args.constant_cols,
                                            const_acc,
                                            output_rect);
    } else {
      // EDGE mode - reads from center region of output
      PadImplBody<KIND, CODE, DIM>{context}(
        output_arr, args.mode, args.pad_width, args.inner_shape, output_rect);
    }
  }
};

template <VariantKind KIND>
static void pad_template(TaskContext& context)
{
  auto mode = static_cast<PadMode>(context.scalar(0).value<int32_t>());

  // Determine number of dimensions by checking output
  size_t num_dims = context.output(0).dim();

  // Parse scalars: mode + global_shape + pad_width_pairs + [constant_value]
  // scalar(0) = mode
  // scalar(1..num_dims) = global_shape
  // scalar(num_dims+1 .. num_dims+1+num_dims*2-1) = pad_width pairs
  // [optional] scalar or input(1) = constant_value

  std::vector<int64_t> inner_shape_vec;
  for (size_t i = 0; i < num_dims; ++i) {
    inner_shape_vec.push_back(context.scalar(1 + i).value<int64_t>());
  }

  size_t pad_width_start = 1 + num_dims;
  std::vector<std::pair<int64_t, int64_t>> pad_width_vec;
  for (size_t i = 0; i < num_dims; ++i) {
    auto left  = context.scalar(pad_width_start + i * 2).value<int64_t>();
    auto right = context.scalar(pad_width_start + i * 2 + 1).value<int64_t>();
    pad_width_vec.push_back({left, right});
  }

  auto constant_rows = context.scalar(pad_width_start + num_dims * 2).value<int64_t>();
  auto constant_cols = context.scalar(pad_width_start + num_dims * 2 + 1).value<int64_t>();

  auto num_inputs = context.num_inputs();

  PhysicalStore const_input{nullptr};
  if (mode == PadMode::CONSTANT) {
    assert(num_inputs > 0);
    // The constant value is expected to be the last input (handles optional extras)
    const_input = context.input(num_inputs - 1);
  }

  PadArgs args{context.output(0),
               const_input,
               mode,
               Span<const std::pair<int64_t, int64_t>>(pad_width_vec.data(), pad_width_vec.size()),
               Span<const int64_t>(inner_shape_vec.data(), inner_shape_vec.size()),
               constant_rows,
               constant_cols};

  double_dispatch(args.output.dim(), args.output.code(), PadImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
