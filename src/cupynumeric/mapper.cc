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

#include "cupynumeric/mapper.h"

using namespace legate;
using namespace legate::mapping;

namespace cupynumeric {

Scalar CuPyNumericMapper::tunable_value(TunableID tunable_id)
{
  LEGATE_ABORT("cuPyNumeric does not use any tunable values");
}

std::vector<StoreMapping> CuPyNumericMapper::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  switch (static_cast<std::int64_t>(task.task_id())) {
    case CUPYNUMERIC_CONVOLVE: {
      std::vector<StoreMapping> mappings;
      auto inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0].data(), options.front()));
      mappings.push_back(StoreMapping::default_mapping(inputs[1].data(), options.front()));
      auto& input_mapping = mappings.back();
      for (uint32_t idx = 2; idx < inputs.size(); ++idx) {
        input_mapping.add_store(inputs[idx].data());
      }
      return mappings;
    }
    case CUPYNUMERIC_FFT: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0].data(), options.front()));
      mappings.push_back(
        StoreMapping::default_mapping(outputs[0].data(), options.front(), true /*exact*/));
      return mappings;
    }
    case CUPYNUMERIC_TRANSPOSE_COPY_2D: {
      std::vector<StoreMapping> mappings;
      auto output = task.output(0);
      mappings.push_back(StoreMapping::default_mapping(output.data(), options.front()));
      mappings.back().policy().ordering.set_fortran_order();
      mappings.back().policy().exact = true;
      return std::move(mappings);
    }
    case CUPYNUMERIC_MATMUL: {
      std::vector<StoreMapping> mappings;
      auto inputA = task.input(1);
      auto inputB = task.input(2);

      mappings.push_back(
        StoreMapping::default_mapping(inputA.data(), options.front(), true /*exact*/));
      mappings.back().policy().redundant = true;
      mappings.push_back(
        StoreMapping::default_mapping(inputB.data(), options.front(), true /*exact*/));
      mappings.back().policy().redundant = true;

      auto outputC = task.output(0);
      mappings.push_back(
        StoreMapping::default_mapping(outputC.data(), options.front(), true /*exact*/));

      return mappings;
    }
    case CUPYNUMERIC_MATVECMUL:
    case CUPYNUMERIC_UNIQUE_REDUCE: {
      // TODO: Our actual requirements are a little less strict than this; we require each array or
      // vector to have a stride of 1 on at least one dimension.
      std::vector<StoreMapping> mappings;
      auto inputs     = task.inputs();
      auto reductions = task.reductions();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
      }
      for (auto& reduction : reductions) {
        mappings.push_back(
          StoreMapping::default_mapping(reduction.data(), options.front(), true /*exact*/));
      }
      return mappings;
    }
    case CUPYNUMERIC_POTRF:
    case CUPYNUMERIC_QR:
    case CUPYNUMERIC_TRSM:
    case CUPYNUMERIC_SOLVE:
    case CUPYNUMERIC_SVD:
    case CUPYNUMERIC_SYRK:
    case CUPYNUMERIC_GEMM:
    case CUPYNUMERIC_MP_POTRF:
    case CUPYNUMERIC_MP_SOLVE: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
        mappings.back().policy().ordering.set_fortran_order();
      }
      for (auto& output : outputs) {
        mappings.push_back(
          StoreMapping::default_mapping(output.data(), options.front(), true /*exact*/));
        mappings.back().policy().ordering.set_fortran_order();
      }
      return mappings;
    }
    // CHANGE: If this code is changed, make sure all layouts are
    // consistent with those assumed in batched_cholesky.cu, etc
    case CUPYNUMERIC_BATCHED_CHOLESKY: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      mappings.reserve(inputs.size() + outputs.size());
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input.data(), options.front()));
        mappings.back().policy().exact = true;
        mappings.back().policy().ordering.set_c_order();
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output.data(), options.front()));
        mappings.back().policy().exact = true;
        mappings.back().policy().ordering.set_c_order();
      }
      return std::move(mappings);
    }
    case CUPYNUMERIC_TRILU: {
      if (task.scalars().size() == 2) {
        return {};
      }
      // If we're here, this task was the post-processing for Cholesky.
      // So we will request fortran ordering
      std::vector<StoreMapping> mappings;
      auto input = task.input(0);
      mappings.push_back(
        StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
      mappings.back().policy().ordering.set_fortran_order();
      return mappings;
    }
    case CUPYNUMERIC_SEARCHSORTED: {
      std::vector<StoreMapping> mappings;
      auto inputs = task.inputs();
      mappings.push_back(
        StoreMapping::default_mapping(inputs[0].data(), options.front(), true /*exact*/));
      return mappings;
    }
    case CUPYNUMERIC_SORT: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
      }
      for (auto& output : outputs) {
        mappings.push_back(
          StoreMapping::default_mapping(output.data(), options.front(), true /*exact*/));
      }
      return mappings;
    }
    case CUPYNUMERIC_SCAN_LOCAL: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
      }
      for (auto& output : outputs) {
        mappings.push_back(
          StoreMapping::default_mapping(output.data(), options.front(), true /*exact*/));
      }
      return mappings;
    }
    case CUPYNUMERIC_SCAN_GLOBAL: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
      }
      for (auto& output : outputs) {
        mappings.push_back(
          StoreMapping::default_mapping(output.data(), options.front(), true /*exact*/));
      }
      return mappings;
    }
    case CUPYNUMERIC_BITGENERATOR: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input.data(), options.front(), true /*exact*/));
      }
      for (auto& output : outputs) {
        mappings.push_back(
          StoreMapping::default_mapping(output.data(), options.front(), true /*exact*/));
      }
      return mappings;
    }
    default: {
      return {};
    }
  }
  assert(false);
  return {};
}

}  // namespace cupynumeric
