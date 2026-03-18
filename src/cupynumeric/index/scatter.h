/* Copyright 2026 NVIDIA Corporation
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

// Task-based scatter (indirect copy) for single-GPU advanced indexing.
// Replaces legate_runtime.issue_scatter() with a kernel-based implementation.
//
// Semantics: out[index[p]] = src[p] for all points p in the output domain.
//
// - Output 0: result array  (type T,              dimensionality OUT_DIM)
// - Input 0:  source array  (type T,              dimensionality SRC_DIM)
// - Input 1:  index array   (type Point<OUT_DIM>, dimensionality SRC_DIM)

class ScatterTask : public CuPyNumericTask<ScatterTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_SCATTER}};

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
