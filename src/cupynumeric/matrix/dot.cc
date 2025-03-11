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

#include "cupynumeric/matrix/dot.h"
#include "cupynumeric/matrix/dot_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct DotImplBody<VariantKind::CPU, CODE> {
  using VAL = type_of<CODE>;
  using ACC = acc_type_of<VAL>;

  template <typename AccessorRD>
  void operator()(AccessorRD out,
                  const AccessorRO<VAL, 1>& rhs1,
                  const AccessorRO<VAL, 1>& rhs2,
                  const Rect<1>& rect,
                  bool dense)
  {
    const auto volume = rect.volume();
    if (dense) {
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
      for (size_t idx = 0; idx < volume; ++idx) {
        const auto prod = static_cast<ACC>(rhs1ptr[idx]) * static_cast<ACC>(rhs2ptr[idx]);
        out.reduce(0, prod);
      }
    } else {
      for (coord_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
        const auto prod = static_cast<ACC>(rhs1[idx]) * static_cast<ACC>(rhs2[idx]);
        out.reduce(0, prod);
      }
    }
  }
};

/*static*/ void DotTask::cpu_variant(TaskContext context)
{
  dot_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  DotTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
