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
#include "cupynumeric/index/zip.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, int DIM, int N>
struct ZipImplBody;

template <VariantKind KIND>
struct ZipImpl {
  template <int DIM, int N>
  void operator()(ZipArgs& args) const
  {
    using VAL     = int64_t;
    auto out_rect = args.out.shape<DIM>();
    auto out      = args.out.write_accessor<Point<N>, DIM>(out_rect);
    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(out_rect);
    if (volume == 0) {
      return;
    }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    bool dense = out.accessor.is_dense_row_major(out_rect);
#else
    bool dense = false;
#endif
    std::vector<AccessorRO<VAL, DIM>> index_arrays;
    for (uint32_t i = 0; i < args.inputs.size(); i++) {
      index_arrays.push_back(args.inputs[i].read_accessor<VAL, DIM>(args.inputs[i].shape<DIM>()));
      dense = dense && index_arrays[i].accessor.is_dense_row_major(out_rect);
    }

    ZipImplBody<KIND, DIM, N>()(out,
                                index_arrays,
                                out_rect,
                                pitches,
                                dense,
                                args.key_dim,
                                args.start_index,
                                args.shape,
                                std::make_index_sequence<N>());
  }
};

template <VariantKind KIND>
static void zip_template(TaskContext& context)
{
  // Here `N` is the number of dimensions of the input array and the number
  // of dimensions of the Point<N> field
  // key_dim - is the number of dimensions of the index arrays before
  // they were broadcasted to the shape of the input array (shape of
  // all index arrays should be the same))
  // start index - is the index from which first index array was passed
  // DIM - dimension of the output array
  //
  // for the example:
  // x.shape = (2,3,4,5)
  // ind1.shape = (6,7,8)
  // ind2.shape = (6,7,8)
  // y = x[:,ind1,ind2,:]
  // y.shape == (2,6,7,8,5)
  // out.shape == (2,6,7,8,5)
  // index_arrays = [ind1', ind2']
  // ind1' == ind1 promoted to (2,6,7,8,5)
  // ind2' == ind2 promoted to (2,6,7,8,5)
  // DIM = 5
  // N = 4
  // key_dim = 3
  // start_index = 1

  int64_t N           = context.scalar(0).value<int64_t>();
  int64_t key_dim     = context.scalar(1).value<int64_t>();
  int64_t start_index = context.scalar(2).value<int64_t>();
  auto shape          = context.scalar(3).value<DomainPoint>();
  std::vector<legate::PhysicalStore> inputs;
  for (auto& input : context.inputs()) {
    inputs.emplace_back(input);
  }
  ZipArgs args{context.output(0), std::move(inputs), N, key_dim, start_index, shape};
  int dim = args.inputs[0].dim();
  // if scalar passed as an input, convert it to the array size 1
  if (dim == 0) {
    dim = 1;
  }

  double_dispatch(dim, N, ZipImpl<KIND>{}, args);
}

}  // namespace cupynumeric
