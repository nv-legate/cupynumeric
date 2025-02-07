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

#include "cupynumeric/unary/unary_op.h"
#include "cupynumeric/unary/unary_op_template.inl"

namespace cupynumeric {

using namespace legate;

template <UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct UnaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = UnaryOp<OP_CODE, CODE>;
  using ARG = typename OP::T;
  using RES = std::result_of_t<OP(ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in,
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

template <typename VAL, int DIM>
struct PointCopyImplBody<VariantKind::CPU, VAL, DIM> {
  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        outptr[idx] = inptr[idx];
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        out[p] = in[p];
      }
    }
  }
};

template <UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct MultiOutUnaryOpImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP   = MultiOutUnaryOp<OP_CODE, CODE>;
  using RHS1 = typename OP::RHS1;
  using RHS2 = typename OP::RHS2;
  using LHS  = std::result_of_t<OP(RHS1, RHS2*)>;

  void operator()(OP func,
                  AccessorWO<LHS, DIM> lhs,
                  AccessorRO<RHS1, DIM> rhs1,
                  AccessorWO<RHS2, DIM> rhs2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto lhsptr  = lhs.ptr(rect);
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        lhsptr[idx] = func(rhs1ptr[idx], &rhs2ptr[idx]);
      }
    } else {
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        lhs[p] = func(rhs1[p], rhs2.ptr(p));
      }
    }
  }
};

/*static*/ void UnaryOpTask::cpu_variant(TaskContext context)
{
  unary_op_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { UnaryOpTask::register_variants(); }
}  // namespace

}  // namespace cupynumeric
