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
#include "cupynumeric/matrix/tile.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <int32_t OUT_DIM, int32_t IN_DIM>
__CUDA_HD__ inline Point<IN_DIM> get_tile_point(const Point<OUT_DIM>& point,
                                                const Point<IN_DIM>& strides)
{
  Point<IN_DIM> result;
  for (int32_t out_idx = OUT_DIM - 1, in_idx = IN_DIM - 1; in_idx >= 0; --out_idx, --in_idx) {
    result[in_idx] = point[out_idx] % strides[in_idx];
  }
  return result;
}

template <VariantKind KIND, typename VAL, int32_t OUT_DIM, int32_t IN_DIM>
struct TileImplBody;

template <VariantKind KIND, typename VAL>
struct TileImpl {
  TaskContext context;
  explicit TileImpl(TaskContext context) : context(context) {}

  template <int32_t OUT_DIM, int32_t IN_DIM, std::enable_if_t<IN_DIM <= OUT_DIM>* = nullptr>
  void operator()(TileArgs& args) const
  {
    const auto out_rect = args.out.shape<OUT_DIM>();
    Pitches<OUT_DIM - 1> out_pitches;
    auto out_volume = out_pitches.flatten(out_rect);

    if (out_volume == 0) {
      return;
    }

    const auto in_rect       = args.in.shape<IN_DIM>();
    Point<IN_DIM> in_strides = in_rect.hi + Point<IN_DIM>::ONES();

    auto out = args.out.write_accessor<VAL, OUT_DIM>();
    auto in  = args.in.read_accessor<VAL, IN_DIM>();

    TileImplBody<KIND, VAL, OUT_DIM, IN_DIM>{context}(
      out_rect, out_pitches, out_volume, in_strides, out, in);
  }

  template <int32_t OUT_DIM, int32_t IN_DIM, std::enable_if_t<!(IN_DIM <= OUT_DIM)>* = nullptr>
  void operator()(TileArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct TileDispatch {
  TaskContext context;
  explicit TileDispatch(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(TileArgs& args) const
  {
    using VAL = type_of<CODE>;
    double_dispatch(args.out.dim(), args.in.dim(), TileImpl<KIND, VAL>{context}, args);
  }
};

template <VariantKind KIND>
static void tile_template(TaskContext& context)
{
  TileArgs args{context.input(0), context.output(0)};
  type_dispatch(args.in.code(), TileDispatch<KIND>{context}, args);
}

}  // namespace cupynumeric
