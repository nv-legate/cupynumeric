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

#include "cupynumeric/bits/unpackbits.h"
#include "cupynumeric/bits/unpackbits_template.inl"

namespace cupynumeric {

using namespace legate;

template <int32_t DIM, Bitorder BITORDER>
struct UnpackbitsImplBody<VariantKind::OMP, DIM, BITORDER> {
  TaskContext context;
  explicit UnpackbitsImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorWO<uint8_t, DIM>& out,
                  const AccessorRO<uint8_t, DIM>& in,
                  const Rect<DIM>& in_rect,
                  const Pitches<DIM - 1>& in_pitches,
                  size_t in_volume,
                  uint32_t axis) const
  {
    Unpack<BITORDER> op{};
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < in_volume; ++idx) {
      auto in_p = in_pitches.unflatten(idx, in_rect.lo);
      op(out, in, in_p, axis);
    }
  }
};

/*static*/ void UnpackbitsTask::omp_variant(TaskContext context)
{
  unpackbits_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
