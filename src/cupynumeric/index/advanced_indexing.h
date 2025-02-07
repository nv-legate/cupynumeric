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

struct AdvancedIndexingArgs {
  legate::PhysicalStore output;
  legate::PhysicalStore input_array;
  legate::PhysicalStore indexing_array;
  const bool is_set;
  const int64_t key_dim;
};

class AdvancedIndexingTask : public CuPyNumericTask<AdvancedIndexingTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{CUPYNUMERIC_ADVANCED_INDEXING};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
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

template <typename T, int DIM>
constexpr void fill_out(legate::Point<DIM>& out, legate::Point<DIM>& p, T&)
{
  out = p;
}

template <typename T, int DIM>
constexpr void fill_out(T& out, legate::Point<DIM>& p, const T& in)
{
  out = in;
}

}  // namespace cupynumeric
