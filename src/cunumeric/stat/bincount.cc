/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/stat/bincount.h"
#include "cunumeric/stat/bincount_template.inl"

namespace cunumeric {

using namespace legate;

template <Type::Code CODE>
struct BincountImplBody<VariantKind::CPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorRD<SumReduction<int64_t>, true, 1> lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    for (size_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
      auto value = rhs[idx];
      assert(lhs_rect.contains(value));
      lhs.reduce(value, 1);
    }
  }

  void operator()(AccessorRD<SumReduction<double>, true, 1> lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const AccessorRO<double, 1>& weights,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    for (size_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
      auto value = rhs[idx];
      assert(lhs_rect.contains(value));
      lhs.reduce(value, weights[idx]);
    }
  }
};

/*static*/ void BincountTask::cpu_variant(TaskContext& context)
{
  bincount_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { BincountTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
