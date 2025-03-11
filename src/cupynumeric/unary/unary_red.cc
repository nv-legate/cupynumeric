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

#include "cupynumeric/unary/unary_red.h"
#include "cupynumeric/unary/unary_red_template.inl"

namespace cupynumeric {

using namespace legate;

template <UnaryRedCode OP_CODE, Type::Code CODE, int DIM, bool HAS_WHERE>
struct UnaryRedImplBody<VariantKind::CPU, OP_CODE, CODE, DIM, HAS_WHERE> {
  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using RHS   = type_of<CODE>;

  void operator()(AccessorRD<LG_OP, true, DIM> lhs,
                  AccessorRO<RHS, DIM> rhs,
                  AccessorRO<bool, DIM> where,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      bool mask  = true;
      if constexpr (HAS_WHERE) {
        mask = where[point];
      }
      if (mask) {
        auto identity = LG_OP::identity;
        lhs.reduce(point, OP::convert(point, collapsed_dim, identity, rhs[point]));
      }
    }
  }
};

/*static*/ void UnaryRedTask::cpu_variant(TaskContext context)
{
  unary_red_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  UnaryRedTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
