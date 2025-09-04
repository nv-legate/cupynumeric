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

#include "cupynumeric/bits/packbits.h"
#include "cupynumeric/bits/packbits_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int32_t DIM, Bitorder BITORDER>
struct PackbitsImplBody<VariantKind::CPU, CODE, DIM, BITORDER> {
  TaskContext context;
  explicit PackbitsImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorWO<uint8_t, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const Rect<DIM>& aligned_rect,
                  const Rect<DIM>& unaligned_rect,
                  const Pitches<DIM - 1>& aligned_pitches,
                  const Pitches<DIM - 1>& unaligned_pitches,
                  size_t aligned_volume,
                  size_t unaligned_volume,
                  int64_t in_hi_axis,
                  uint32_t axis) const
  {
    Pack<BITORDER, true /* ALIGNED */> op{};
    Pack<BITORDER, false /* ALIGNED */> op_unaligned{};

    for (size_t idx = 0; idx < aligned_volume; ++idx) {
      auto out_p = aligned_pitches.unflatten(idx, aligned_rect.lo);
      out[out_p] = op(in, out_p, in_hi_axis, axis);
    }
    for (size_t idx = 0; idx < unaligned_volume; ++idx) {
      auto out_p = unaligned_pitches.unflatten(idx, unaligned_rect.lo);
      out[out_p] = op_unaligned(in, out_p, in_hi_axis, axis);
    }
  }
};

/*static*/ void PackbitsTask::cpu_variant(TaskContext context)
{
  packbits_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  PackbitsTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
