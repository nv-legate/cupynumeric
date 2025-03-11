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

#include "cupynumeric/binary/binary_red.h"
#include "cupynumeric/binary/binary_red_template.inl"

namespace cupynumeric {

using namespace legate;

template <BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryRedImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = BinaryOp<OP_CODE, CODE>;
  using ARG = type_of<CODE>;

  template <typename AccessorRD>
  void operator()(OP func,
                  AccessorRD out,
                  AccessorRO<ARG, DIM> in1,
                  AccessorRO<ARG, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume = rect.volume();
    if (dense) {
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        if (!func(in1ptr[idx], in2ptr[idx])) {
          out.reduce(0, false);
          return;
        }
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto point = pitches.unflatten(idx, rect.lo);
        if (!func(in1[point], in2[point])) {
          out.reduce(0, false);
          return;
        }
      }
    }

    out.reduce(0, true);
  }
};

/*static*/ void BinaryRedTask::cpu_variant(TaskContext context)
{
  binary_red_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
const auto reg_ = []() -> char {
  BinaryRedTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
