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
#include "cupynumeric/omp_help.h"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct DotImplBody<VariantKind::OMP, CODE> {
  TaskContext context;
  explicit DotImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;
  using ACC = acc_type_of<VAL>;

  template <typename AccessorRD>
  void operator()(AccessorRD out,
                  const AccessorRO<VAL, 1>& rhs1,
                  const AccessorRO<VAL, 1>& rhs2,
                  const Rect<1>& rect,
                  bool dense)
  {
    const auto volume      = rect.volume();
    const auto max_threads = omp_get_max_threads();
    ThreadLocalStorage<ACC> locals(max_threads);
    for (auto idx = 0; idx < max_threads; ++idx) {
      locals[idx] = SumReduction<ACC>::identity;
    }

    if (dense) {
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          const auto prod = static_cast<ACC>(rhs1ptr[idx]) * static_cast<ACC>(rhs2ptr[idx]);
          SumReduction<ACC>::template fold<true>(locals[tid], prod);
        }
      }
    } else {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (coord_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
          const auto prod = static_cast<ACC>(rhs1[idx]) * static_cast<ACC>(rhs2[idx]);
          SumReduction<ACC>::template fold<true>(locals[tid], prod);
        }
      }
    }

    for (auto idx = 0; idx < max_threads; ++idx) {
      out.reduce(0, locals[idx]);
    }
  }
};

/*static*/ void DotTask::omp_variant(TaskContext context)
{
  dot_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
