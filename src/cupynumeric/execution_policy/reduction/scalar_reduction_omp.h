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

#pragma once

#include "cupynumeric/execution_policy/reduction/scalar_reduction.h"
#include "cupynumeric/omp_help.h"

#include <omp.h>

namespace cupynumeric {

template <class LG_OP, class Tag>
struct ScalarReductionPolicy<VariantKind::OMP, LG_OP, Tag> {
  legate::TaskContext context;
  ScalarReductionPolicy(legate::TaskContext tcontext) : context(tcontext) {}

  template <class AccessorRD, class LHS, class Kernel>
  void operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    const auto max_threads = omp_get_max_threads();
    ThreadLocalStorage<LHS> locals(max_threads);
    for (auto idx = 0; idx < max_threads; ++idx) {
      locals[idx] = identity;
    }
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        kernel(locals[tid], idx, identity, Tag{});
      }
    }
    for (auto idx = 0; idx < max_threads; ++idx) {
      out.reduce(0, locals[idx]);
    }
  }
};

}  // namespace cupynumeric
