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

struct WrapArgs {
  legate::PhysicalStore out{nullptr};  // Array with Point<N> type that is used to
                                       // copy information from original array to the
                                       //  `wrapped` one
  const legate::DomainPoint shape;     // shape of the original array
  const bool has_input;
  const bool check_bounds;
  legate::PhysicalStore in{nullptr};
};

class WrapTask : public CuPyNumericTask<WrapTask> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_WRAP}};

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

__CUDA_HD__ inline int64_t compute_idx(const int64_t i, const int64_t volume, const bool&)
{
  return i % volume;
}

__CUDA_HD__ inline int64_t compute_idx(const int64_t i,
                                       const int64_t volume,
                                       const legate::AccessorRO<int64_t, 1>& indices)
{
  int64_t idx   = indices[i];
  int64_t index = idx < 0 ? idx + volume : idx;
  return index;
}

inline void check_idx(const int64_t i,
                      const int64_t volume,
                      const legate::AccessorRO<int64_t, 1>& indices)
{
  int64_t idx   = indices[i];
  int64_t index = idx < 0 ? idx + volume : idx;
  if (index < 0 || index >= volume) {
    throw legate::TaskException("index is out of bounds in index array");
  }
}
inline void check_idx(const int64_t i, const int64_t volume, const bool&)
{
  // don't do anything when wrapping indices
}

inline bool check_idx_omp(const int64_t i,
                          const int64_t volume,
                          const legate::AccessorRO<int64_t, 1>& indices)
{
  int64_t idx   = indices[i];
  int64_t index = idx < 0 ? idx + volume : idx;
  return (index < 0 || index >= volume);
}
inline bool check_idx_omp(const int64_t i, const int64_t volume, const bool&) { return false; }

}  // namespace cupynumeric
