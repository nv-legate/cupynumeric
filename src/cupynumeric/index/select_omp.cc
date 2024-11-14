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

#include "cupynumeric/index/select.h"
#include "cupynumeric/index/select_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct SelectImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const std::vector<AccessorRO<bool, DIM>>& condlist,
                  const std::vector<AccessorRO<VAL, DIM>>& choicelist,
                  VAL default_val,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    uint32_t narrays    = condlist.size();
#ifdef DEBUG_CUPYNUMERIC
    assert(narrays == choicelist.size());
#endif

    if (dense) {
      auto outptr = out.ptr(rect);
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        outptr[idx] = default_val;
      }
      for (int32_t c = (narrays - 1); c >= 0; c--) {
        auto condptr   = condlist[c].ptr(rect);
        auto choiseptr = choicelist[c].ptr(rect);
#pragma omp parallel for schedule(static)
        for (int32_t idx = 0; idx < volume; idx++) {
          if (condptr[idx]) {
            outptr[idx] = choiseptr[idx];
          }
        }
      }
    } else {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = default_val;
      }
      for (int32_t c = (narrays - 1); c >= 0; c--) {
#pragma omp parallel for schedule(static)
        for (int32_t idx = 0; idx < volume; idx++) {
          auto p = pitches.unflatten(idx, rect.lo);
          if (condlist[c][p]) {
            out[p] = choicelist[c][p];
          }
        }
      }
    }
  }
};

/*static*/ void SelectTask::omp_variant(TaskContext context)
{
  select_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
