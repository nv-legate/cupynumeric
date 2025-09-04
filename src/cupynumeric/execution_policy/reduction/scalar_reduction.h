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

#include "cupynumeric/cupynumeric_task.h"

namespace cupynumeric {

template <VariantKind KIND, class LG_OP, class Tag = void>
struct ScalarReductionPolicy {
  legate::TaskContext context;
  ScalarReductionPolicy(legate::TaskContext tcontext) : context(tcontext) {}

  // No C++-20 yet. This is just here to illustrate the expected concept
  // that all kernels passed to this execution should have.
  struct KernelConcept {
    // Every operator should take a scalar LHS as the
    // target of the reduction and an index represeting the point
    // in the iteration space being added into the reduction.
    template <class LHS>
    void operator()(LHS& lhs, size_t idx)
    {
      // LHS <- op[idx]
    }
  };
};

template <class LG_OP, class Tag>
struct ScalarReductionPolicy<VariantKind::CPU, LG_OP, Tag> {
  legate::TaskContext context;
  ScalarReductionPolicy(legate::TaskContext tcontext) : context(tcontext) {}

  template <class AccessorRD, class LHS, class Kernel>
  void operator()(size_t volume, AccessorRD& out, const LHS& identity, Kernel&& kernel)
  {
    auto result = identity;
    for (size_t idx = 0; idx < volume; ++idx) {
      kernel(result, idx, identity, Tag{});
    }
    out.reduce(0, result);
  }
};

}  // namespace cupynumeric
