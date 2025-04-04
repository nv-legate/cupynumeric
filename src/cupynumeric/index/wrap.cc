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

#include "cupynumeric/index/wrap.h"
#include "cupynumeric/index/wrap_template.inl"

namespace cupynumeric {

using namespace legate;

template <int DIM>
struct WrapImplBody<VariantKind::CPU, DIM> {
  template <typename IND>
  void operator()(const AccessorWO<Point<DIM>, 1>& out,
                  const Pitches<0>& pitches_out,
                  const Rect<1>& rect_out,
                  const Pitches<DIM - 1>& pitches_base,
                  const Rect<DIM>& rect_base,
                  const bool dense,
                  const bool check_bounds,
                  const IND& indices) const
  {
    const int64_t start    = rect_out.lo[0];
    const int64_t end      = rect_out.hi[0];
    const auto volume_base = rect_base.volume();
    if (dense) {
      auto outptr = out.ptr(rect_out);
      for (int64_t i = start; i <= end; i++) {
        if (check_bounds) {
          check_idx(i, volume_base, indices);
        }
        const int64_t input_idx = compute_idx(i, volume_base, indices);
        auto point              = pitches_base.unflatten(input_idx, rect_base.lo);
        outptr[i - start]       = point;
      }
    } else {
      for (int64_t i = start; i <= end; i++) {
        if (check_bounds) {
          check_idx(i, volume_base, indices);
        }
        const int64_t input_idx = compute_idx(i, volume_base, indices);
        auto point              = pitches_base.unflatten(input_idx, rect_base.lo);
        out[i]                  = point;
      }
    }  // else
  }
};

/*static*/ void WrapTask::cpu_variant(TaskContext context)
{
  wrap_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  WrapTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
