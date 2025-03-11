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

#include "cupynumeric/binary/binary_op.h"
#include "cupynumeric/binary/binary_op_template.inl"

namespace cupynumeric {

using namespace legate;

template <BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP   = BinaryOp<OP_CODE, CODE>;
  using RHS1 = type_of<CODE>;
  using RHS2 = rhs2_of_binary_op<OP_CODE, CODE>;
  using LHS  = std::result_of_t<OP(RHS1, RHS2)>;

  void operator()(OP func,
                  AccessorWO<LHS, DIM> out,
                  AccessorRO<RHS1, DIM> in1,
                  AccessorRO<RHS2, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        outptr[idx] = func(in1ptr[idx], in2ptr[idx]);
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = func(in1[p], in2[p]);
      }
    }
  }
};

/*static*/ void BinaryOpTask::cpu_variant(TaskContext context)
{
  binary_op_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  BinaryOpTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
