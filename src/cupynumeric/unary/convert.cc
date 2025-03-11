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

#include "cupynumeric/unary/convert.h"
#include "cupynumeric/unary/convert_template.inl"

namespace cupynumeric {

using namespace legate;

template <ConvertCode NAN_OP, Type::Code DST_TYPE, Type::Code SRC_TYPE, int DIM>
struct ConvertImplBody<VariantKind::CPU, NAN_OP, DST_TYPE, SRC_TYPE, DIM> {
  using OP  = ConvertOp<NAN_OP, DST_TYPE, SRC_TYPE>;
  using SRC = type_of<SRC_TYPE>;
  using DST = type_of<DST_TYPE>;

  void operator()(OP func,
                  AccessorWO<DST, DIM> out,
                  AccessorRO<SRC, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        outptr[idx] = func(inptr[idx]);
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = func(in[p]);
      }
    }
  }
};

/*static*/ void ConvertTask::cpu_variant(TaskContext context)
{
  convert_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  ConvertTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
