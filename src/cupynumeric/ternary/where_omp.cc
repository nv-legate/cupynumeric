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

#include "cupynumeric/ternary/where.h"
#include "cupynumeric/ternary/where_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct WhereImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<bool, DIM> mask,
                  AccessorRO<VAL, DIM> in1,
                  AccessorRO<VAL, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      size_t volume = rect.volume();
      auto outptr   = out.ptr(rect);
      auto maskptr  = mask.ptr(rect);
      auto in1ptr   = in1.ptr(rect);
      auto in2ptr   = in2.ptr(rect);
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        outptr[idx] = maskptr[idx] ? in1ptr[idx] : in2ptr[idx];
      }
    } else {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        out[point] = mask[point] ? in1[point] : in2[point];
      }
    }
  }
};

/*static*/ void WhereTask::omp_variant(TaskContext context)
{
  where_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
