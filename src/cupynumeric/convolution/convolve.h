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

// We'll make some assumptions here about cache size
// that should hold up against most CPUs out there today
// Most L1 caches are 32-48KB so we'll go with the lower bound
#define L1_CACHE_SIZE 32768
// Most L2 caches are at least 256KB
#define L2_CACHE_SIZE 262144
// Most caches have 64B lines
#define CACHE_LINE_SIZE 64

namespace cupynumeric {

struct ConvolveArgs {
  legate::PhysicalStore out{nullptr};
  legate::PhysicalStore filter{nullptr};
  std::vector<legate::PhysicalStore> inputs;
  legate::Domain root_domain;
  CuPyNumericConvolveMethod method;
};

class ConvolveTask : public CuPyNumericTask<ConvolveTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_CONVOLVE}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

 public:
  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace cupynumeric
