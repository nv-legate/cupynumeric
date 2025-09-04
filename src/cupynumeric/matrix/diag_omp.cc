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

#include "cupynumeric/matrix/diag.h"
#include "cupynumeric/matrix/diag_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct DiagImplBody<VariantKind::OMP, CODE, DIM, true> {
  TaskContext context;
  explicit DiagImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorRD<SumReduction<VAL>, true, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const coord_t& start,
                  const Pitches<DIM - 1>& m_pitches,
                  const Rect<DIM>& m_shape,
                  const size_t naxes,
                  const coord_t distance) const
  {
    size_t skip_size = 1;

    for (size_t i = 0; i < naxes; i++) {
      auto diff = 1 + m_shape.hi[DIM - i - 1] - m_shape.lo[DIM - i - 1];
      if (diff != 0) {
        skip_size *= diff;
      }
    }
    const size_t volume = m_shape.volume();
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; idx += skip_size) {
      Point<DIM> in_p  = m_pitches.unflatten(idx, m_shape.lo);
      Point<DIM> out_p = in_p;
      for (coord_t d = 0; d < distance; ++d) {
        for (size_t i = DIM - naxes; i < DIM; i++) {
          in_p[i]  = start + d;
          out_p[i] = start + d;
        }
        auto v = in[in_p];
        out.reduce(out_p, v);
      }
    }
  }
};

// not extract (create a new 2D matrix with diagonal from vector)
template <Type::Code CODE>
struct DiagImplBody<VariantKind::OMP, CODE, 2, false> {
  TaskContext context;
  explicit DiagImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorRO<VAL, 2>& in,
                  const AccessorRW<VAL, 2>& out,
                  const Point<2>& start,
                  const coord_t distance)
  {
#pragma omp parallel for schedule(static)
    for (coord_t idx = 0; idx < distance; idx++) {
      Point<2> in_p(start[0] + idx, start[1] + idx);
      Point<2> out_p(start[0] + idx, start[1] + idx);
      out[out_p] = in[in_p];
    }
  }
};

/*static*/ void DiagTask::omp_variant(TaskContext context)
{
  diag_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
