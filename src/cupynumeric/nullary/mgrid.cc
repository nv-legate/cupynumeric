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

#include "cupynumeric/nullary/mgrid.h"
#include "cupynumeric/nullary/mgrid_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int32_t DIM>
struct MGridImplBody<VariantKind::CPU, VAL, DIM> {
  TaskContext context;
  explicit MGridImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const std::array<VAL, DIM - 1>& starts,
                  const std::array<VAL, DIM - 1>& steps) const
  {
    const size_t volume = rect.volume();

    for (size_t idx = 0; idx < volume; ++idx) {
      const auto point = pitches.unflatten(idx, rect.lo);
      const auto d     = point[0];

      out[point] = starts[d] + steps[d] * static_cast<VAL>(point[d + 1]);
    }
  }
};

/*static*/ void MGridTask::cpu_variant(TaskContext context)
{
  mgrid_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  MGridTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
