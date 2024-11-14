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

#include "cupynumeric/transform/flip.h"
#include "cupynumeric/transform/flip_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int32_t DIM>
struct FlipImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  legate::Span<const int32_t> axes) const

  {
    const size_t volume = rect.volume();
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      auto q = p;
      for (uint32_t idx = 0; idx < axes.size(); ++idx) {
        q[axes[idx]] = rect.hi[axes[idx]] - q[axes[idx]];
      }
      out[p] = in[q];
    }
  }
};

/*static*/ void FlipTask::omp_variant(TaskContext context)
{
  flip_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
