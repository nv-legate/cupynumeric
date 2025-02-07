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

struct ZipArgs {
  legate::PhysicalStore out;
  std::vector<legate::PhysicalStore> inputs;
  const int64_t N;
  const int64_t key_dim;
  const int64_t start_index;
  const legate::DomainPoint shape;
};

class ZipTask : public CuPyNumericTask<ZipTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{CUPYNUMERIC_ZIP};

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

constexpr legate::coord_t compute_idx(legate::coord_t index, legate::coord_t extent)
{
  legate::coord_t new_index = index < 0 ? index + extent : index;
  if (new_index < 0 || new_index >= extent) {
    throw legate::TaskException("index is out of bounds in index array");
  }
  return new_index;
}

constexpr std::pair<legate::coord_t, bool> compute_idx_omp(legate::coord_t index,
                                                           legate::coord_t extent)
{
  legate::coord_t new_index = index < 0 ? index + extent : index;
  bool out_of_bounds        = (new_index < 0 || new_index >= extent);
  return {new_index, out_of_bounds};
}

constexpr legate::coord_t compute_idx_cuda(legate::coord_t index, legate::coord_t extent)
{
  legate::coord_t new_index = index < 0 ? index + extent : index;
  return new_index;
}

}  // namespace cupynumeric
