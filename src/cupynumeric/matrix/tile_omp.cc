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

#include "cupynumeric/matrix/tile.h"
#include "cupynumeric/matrix/tile_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int32_t OUT_DIM, int32_t IN_DIM>
struct TileImplBody<VariantKind::OMP, VAL, OUT_DIM, IN_DIM> {
  TaskContext context;
  explicit TileImplBody(TaskContext context) : context(context) {}

  void operator()(const Rect<OUT_DIM>& out_rect,
                  const Pitches<OUT_DIM - 1>& out_pitches,
                  size_t out_volume,
                  const Point<IN_DIM>& in_strides,
                  const AccessorWO<VAL, OUT_DIM>& out,
                  const AccessorRO<VAL, IN_DIM>& in) const
  {
#pragma omp parallel for
    for (size_t out_idx = 0; out_idx < out_volume; ++out_idx) {
      const auto out_point = out_pitches.unflatten(out_idx, out_rect.lo);
      const auto in_point  = get_tile_point(out_point, in_strides);
      out[out_point]       = in[in_point];
    }
  }
};

/*static*/ void TileTask::omp_variant(TaskContext context)
{
  tile_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
