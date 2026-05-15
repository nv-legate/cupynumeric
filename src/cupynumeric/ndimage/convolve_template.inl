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
#include "cupynumeric/ndimage/convolve.h"

#include <cstdint>

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct NdimageConvolveImplBody;

template <VariantKind KIND>
struct NdimageConvolveDispatch {
  TaskContext context;
  explicit NdimageConvolveDispatch(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(NdimageConvolveArgs& args) const
  {
    using VAL                    = type_of<CODE>;
    PhysicalStore& output_store  = args.output;
    PhysicalStore& input_store   = args.input;
    PhysicalStore& weights_store = args.weights;

    auto out_rect     = output_store.shape<DIM>();
    auto input_rect   = input_store.shape<DIM>();
    auto weights_rect = weights_store.shape<DIM>();

    // Because the input may span multiple tiles across GPUs,
    // we need to ensure the weights aren't larger than the input tile
    // so that we can satisfy certain boundary modes.
    if (!context.is_single_task()) {
      for (int dim = 0; dim < DIM; ++dim) {
        int32_t weight_dim_size = weights_rect.hi[dim] - weights_rect.lo[dim] + 1;
        int32_t output_dim_size = out_rect.hi[dim] - out_rect.lo[dim] + 1;

        assert(weight_dim_size <= output_dim_size);
      }
    }

    if (out_rect.empty()) {
      return;
    }

    auto output     = output_store.write_accessor<VAL, DIM>(out_rect);
    auto input      = input_store.read_accessor<VAL, DIM>(input_rect);
    auto weights    = weights_store.read_accessor<VAL, DIM>(weights_rect);
    auto fill_value = args.fill_value.value<VAL>();
    Point<DIM> origins;
    for (int dim = 0; dim < DIM; ++dim) {
      origins[dim] = args.origins[dim];
    }

    NdimageConvolveImplBody<KIND, VAL, DIM>{context}(
      output, input, weights, input_rect, out_rect, weights_rect, args.mode, fill_value, origins);
  }
};

template <VariantKind KIND>
static void ndimage_convolve_template(TaskContext& context)
{
  assert(context.output(0).dim() > 0);

  std::vector<int64_t> origins;
  origins.reserve(context.input(0).dim());
  for (int32_t dim = 0; dim < context.input(0).dim(); ++dim) {
    origins.push_back(context.scalar(2 + dim).value<int64_t>());
  }

  NdimageConvolveArgs args{
    context.output(0),
    context.input(2),
    context.input(0),
    static_cast<CuPyNumericNdimageConvolveMode>(context.scalar(0).value<std::int32_t>()),
    context.scalar(1),
    std::move(origins),
  };

  double_dispatch(
    args.output.dim(), args.output.code(), NdimageConvolveDispatch<KIND>{context}, args);
}

}  // namespace cupynumeric
