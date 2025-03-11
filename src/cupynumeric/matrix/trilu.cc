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

#include "cupynumeric/matrix/trilu.h"
#include "cupynumeric/matrix/trilu_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int32_t DIM, bool LOWER>
struct TriluImplBody<VariantKind::CPU, CODE, DIM, LOWER> {
  using VAL = type_of<CODE>;

  template <bool C_ORDER>
  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1, C_ORDER>& pitches,
                  const Point<DIM>& lo,
                  size_t volume,
                  int32_t k) const
  {
    if (LOWER) {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, lo);
        if (p[DIM - 2] + k >= p[DIM - 1]) {
          out[p] = in[p];
        } else {
          out[p] = 0;
        }
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, lo);
        if (p[DIM - 2] + k <= p[DIM - 1]) {
          out[p] = in[p];
        } else {
          out[p] = 0;
        }
      }
    }
  }
};

/*static*/ void TriluTask::cpu_variant(TaskContext context)
{
  trilu_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  TriluTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
