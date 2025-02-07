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
#include "cupynumeric/fft/fft.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/fft/fft_util.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND,
          CuPyNumericFFTType FFT_TYPE,
          Type::Code CODE_OUT,
          Type::Code CODE_IN,
          int32_t DIM>
struct FFTImplBody;

template <VariantKind KIND, CuPyNumericFFTType FFT_TYPE>
struct FFTImpl {
  template <Type::Code CODE_IN,
            int32_t DIM,
            std::enable_if_t<(FFT<FFT_TYPE, CODE_IN>::valid)>* = nullptr>
  void operator()(FFTArgs& args) const
  {
    using INPUT_TYPE  = type_of<CODE_IN>;
    using OUTPUT_TYPE = type_of<FFT<FFT_TYPE, CODE_IN>::CODE_OUT>;

    auto in_rect  = args.input.shape<DIM>();
    auto out_rect = args.output.shape<DIM>();
    if (in_rect.empty() || out_rect.empty()) {
      return;
    }

    auto input  = args.input.read_accessor<INPUT_TYPE, DIM>(in_rect);
    auto output = args.output.write_accessor<OUTPUT_TYPE, DIM>(out_rect);

    FFTImplBody<KIND, FFT_TYPE, FFT<FFT_TYPE, CODE_IN>::CODE_OUT, CODE_IN, DIM>()(
      output, input, out_rect, in_rect, args.axes, args.direction, args.operate_over_axes);
  }

  // Filter valid types
  template <Type::Code CODE_IN,
            int32_t DIM,
            std::enable_if_t<(!FFT<FFT_TYPE, CODE_IN>::valid)>* = nullptr>
  void operator()(FFTArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct FFTDispatch {
  template <CuPyNumericFFTType FFT_TYPE>
  void operator()(FFTArgs& args) const
  {
    // Not expecting changing dimensions, at least for now
    assert(args.input.dim() == args.output.dim());

    double_dispatch(args.input.dim(), args.input.code(), FFTImpl<KIND, FFT_TYPE>{}, args);
  }
};

template <VariantKind KIND>
static void fft_template(TaskContext& context)
{
  FFTArgs args;

  args.output = context.output(0);
  args.input  = context.input(0);
  // Scalar arguments. Pay attention to indexes / ranges when adding or reordering arguments
  args.type              = context.scalar(0).value<CuPyNumericFFTType>();
  args.direction         = context.scalar(1).value<CuPyNumericFFTDirection>();
  args.operate_over_axes = context.scalar(2).value<bool>();

  const auto num_scalars = context.num_scalars();
  for (uint32_t i = 3; i < num_scalars; ++i) {
    args.axes.push_back(context.scalar(i).value<int64_t>());
  }

  fft_dispatch(args.type, FFTDispatch<KIND>{}, args);
}
}  // namespace cupynumeric
