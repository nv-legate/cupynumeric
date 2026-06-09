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

#include <cstdint>
#include <vector>

namespace cupynumeric {

struct NdimageFourierGaussianParams {
  int64_t n{-1};
  int32_t axis{-1};
  int64_t extents[LEGATE_MAX_DIM]{};
  double sigmas[LEGATE_MAX_DIM]{};
};

struct NdimageFourierGaussianArgs {
  legate::PhysicalStore output{nullptr};
  legate::PhysicalStore input{nullptr};
  NdimageFourierGaussianParams params;
};

class NdimageFourierGaussianTask : public CuPyNumericTask<NdimageFourierGaussianTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_NDIMAGE_FOURIER_GAUSSIAN}};

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
