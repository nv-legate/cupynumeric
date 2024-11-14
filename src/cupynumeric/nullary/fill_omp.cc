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

#include "cupynumeric/nullary/fill.h"
#include "cupynumeric/nullary/fill_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename T>
struct Identity {
  constexpr T operator()(const T& in) const { return in; }
};

template <typename VAL, int32_t DIM>
struct FillImplBody<VariantKind::OMP, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, 1> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    auto fill_value = in[0];
    size_t volume   = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        outptr[idx] = fill_value;
      }
    } else {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = fill_value;
      }
    }
  }
};

/*static*/ void FillTask::omp_variant(TaskContext context)
{
  fill_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
