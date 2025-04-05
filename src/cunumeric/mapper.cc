/* Copyright 2021-2022 NVIDIA Corporation
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

#include "env_defaults.h"
#include "cunumeric/mapper.h"

using namespace legate;
using namespace legate::mapping;

namespace cunumeric {

CuNumericMapper::CuNumericMapper()
  : min_gpu_chunk(
      extract_env("CUNUMERIC_MIN_GPU_CHUNK", MIN_GPU_CHUNK_DEFAULT, MIN_GPU_CHUNK_TEST)),
    min_cpu_chunk(
      extract_env("CUNUMERIC_MIN_CPU_CHUNK", MIN_CPU_CHUNK_DEFAULT, MIN_CPU_CHUNK_TEST)),
    min_omp_chunk(extract_env("CUNUMERIC_MIN_OMP_CHUNK", MIN_OMP_CHUNK_DEFAULT, MIN_OMP_CHUNK_TEST))
{
}

void CuNumericMapper::set_machine(const legate::mapping::MachineQueryInterface* m) { machine = m; }

TaskTarget CuNumericMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return *options.begin();
}

Scalar CuNumericMapper::tunable_value(TunableID tunable_id)
{
  switch (tunable_id) {
    case CUNUMERIC_TUNABLE_NUM_GPUS: {
      int32_t num_gpus = machine->gpus().size() * machine->total_nodes();
      return Scalar(num_gpus);
    }
    case CUNUMERIC_TUNABLE_NUM_PROCS: {
      int32_t num_procs = 0;
      if (!machine->gpus().empty())
        num_procs = machine->gpus().size() * machine->total_nodes();
      else if (!machine->omps().empty())
        num_procs = machine->omps().size() * machine->total_nodes();
      else
        num_procs = machine->cpus().size() * machine->total_nodes();
      return Scalar(num_procs);
    }
    case CUNUMERIC_TUNABLE_MAX_EAGER_VOLUME: {
      int32_t eager_volume = 0;
      // TODO: make these profile guided
      if (!machine->gpus().empty())
        eager_volume = min_gpu_chunk;
      else if (!machine->omps().empty())
        eager_volume = min_omp_chunk;
      else
        eager_volume = min_cpu_chunk;
      return Scalar(eager_volume);
    }
    default: break;
  }
  LEGATE_ABORT;  // unknown tunable value
}

std::vector<StoreMapping> CuNumericMapper::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  switch (task.task_id()) {
    case CUNUMERIC_CONVOLVE: {
      std::vector<StoreMapping> mappings;
      auto& inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.push_back(StoreMapping::default_mapping(inputs[1], options.front()));
      auto& input_mapping = mappings.back();
      for (uint32_t idx = 2; idx < inputs.size(); ++idx)
        input_mapping.stores.push_back(inputs[idx]);
      return std::move(mappings);
    }
    case CUNUMERIC_FFT: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.push_back(StoreMapping::default_mapping(outputs[0], options.front()));
      mappings.back().policy.exact = true;
      mappings.back().policy.ordering.c_order();
      return std::move(mappings);
    }
    case CUNUMERIC_TRANSPOSE_COPY_2D: {
      std::vector<StoreMapping> mappings;
      auto& outputs = task.outputs();
      mappings.push_back(StoreMapping::default_mapping(outputs[0], options.front()));
      mappings.back().policy.ordering.set_fortran_order();
      mappings.back().policy.exact = true;
      return std::move(mappings);
    }
    case CUNUMERIC_MATMUL:
    case CUNUMERIC_MATVECMUL:
    case CUNUMERIC_UNIQUE_REDUCE:
    case CUNUMERIC_UNZIP_INDICES: {
      // TODO: Our actual requirements are a little less strict than this; we require each array or
      // vector to have a stride of 1 on at least one dimension.
      std::vector<StoreMapping> mappings;
      auto& inputs     = task.inputs();
      auto& reductions = task.reductions();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.exact = true;
      }
      for (auto& reduction : reductions) {
        mappings.push_back(StoreMapping::default_mapping(reduction, options.front()));
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    case CUNUMERIC_POTRF:
    case CUNUMERIC_TRSM:
    case CUNUMERIC_SOLVE:
    case CUNUMERIC_SYRK:
    case CUNUMERIC_GEMM: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.ordering.set_fortran_order();
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.ordering.set_fortran_order();
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    // CHANGE: If this code is changed, make sure all layouts are
    // consistent with those assumed in batched_cholesky.cu, etc
    case CUNUMERIC_BATCHED_CHOLESKY: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      mappings.reserve(inputs.size() + outputs.size());
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.exact = true;
        mappings.back().policy.ordering.set_c_order();
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.exact = true;
        mappings.back().policy.ordering.set_c_order();
      }
      return std::move(mappings);
    }
    case CUNUMERIC_TRILU: {
      if (task.scalars().size() == 2) return {};
      // If we're here, this task was the post-processing for Cholesky.
      // So we will request fortran ordering
      std::vector<StoreMapping> mappings;
      auto& input = task.inputs().front();
      mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      mappings.back().policy.ordering.set_fortran_order();
      mappings.back().policy.exact = true;
      return std::move(mappings);
    }
    case CUNUMERIC_SEARCHSORTED: {
      std::vector<StoreMapping> mappings;
      auto& inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.back().policy.exact = true;
      return std::move(mappings);
    }
    case CUNUMERIC_SORT: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.ordering.set_c_order();
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.ordering.set_c_order();
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    case CUNUMERIC_SCAN_LOCAL: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.ordering.set_c_order();
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.ordering.set_c_order();
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    case CUNUMERIC_SCAN_GLOBAL: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.ordering.set_c_order();
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.ordering.set_c_order();
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    case CUNUMERIC_BITGENERATOR: {
      std::vector<StoreMapping> mappings;
      auto& inputs  = task.inputs();
      auto& outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
        mappings.back().policy.exact = true;
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
        mappings.back().policy.exact = true;
      }
      return std::move(mappings);
    }
    default: {
      return {};
    }
  }
  assert(false);
  return {};
}

}  // namespace cunumeric
