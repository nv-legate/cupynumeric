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
#include "cupynumeric/index/zip.h"

#include <vector>

namespace cupynumeric {

struct ZipGatherArgs {
  legate::PhysicalStore out;
  legate::PhysicalStore source;
  std::vector<legate::PhysicalStore> inputs;
  const int64_t key_dim;
  const int64_t start_index;
  const legate::DomainPoint shape;
  const bool check_bounds;
};

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
namespace detail {
// Runs the fused zip+gather GPU kernel directly from `args`, bypassing the
// task's scalar/input layout. Used by the all2all gather task when the
// partitioner sequentializes the launch (single rank owns all the data): the
// NCCL shuffle is unnecessary, so we do the same local zip+gather as
// CUPYNUMERIC_ZIPGATHER. Defined in zipgather.cu so the GPU ImplBody
// instantiations are reused.
void launch_local_zipgather_gpu(legate::TaskContext& context, ZipGatherArgs& args);
}  // namespace detail
#endif

class ZipGatherTask : public CuPyNumericTask<ZipGatherTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_ZIPGATHER}};

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
