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

enum class PartitionStrategy : int { Batched = 0, Unbatched };

struct MatMulArgs {
  legate::PhysicalStore lhs;
  legate::PhysicalStore rhs1;
  legate::PhysicalStore rhs2;
  PartitionStrategy partition_type;
};

class MatMulTask : public CuPyNumericTask<MatMulTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_MATMUL}};

  // Only the CPU implementation needs temporary allocations due to lack of float16 support
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
  static constexpr auto OMP_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

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
