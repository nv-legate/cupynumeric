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
#include "legate/utilities/assert.h"

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
  const auto task_id = static_cast<CuPyNumericOpCode>(task.task_id());

  switch (task_id) {
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
    case CUPYNUMERIC_GEEV:
    case CUPYNUMERIC_SYEV: {
      std::vector<StoreMapping> mappings;
      auto input_a   = task.input(0);
      auto output_ew = task.output(0);

      auto dimensions = input_a.dim();

      // last 2 (matrix) dimensions col-major
      // batch dimensions 0, ..., dim-3 row-major
      std::vector<int32_t> dim_order;
      dim_order.push_back(dimensions - 2);
      dim_order.push_back(dimensions - 1);
      for (int32_t i = dimensions - 3; i >= 0; i--) {
        dim_order.push_back(i);
      }

      mappings.push_back(
        StoreMapping::default_mapping(input_a.data(), options.front(), true /*exact*/));
      mappings.back().policy().ordering.set_custom_order(dim_order);

      // eigenvalue computation is optional
      if (task.outputs().size() > 1) {
        auto output_ev = task.output(1);
        mappings.push_back(
          StoreMapping::default_mapping(output_ev.data(), options.front(), true /*exact*/));
        mappings.back().policy().ordering.set_custom_order(dim_order);
      }

      // remove last dimension for eigenvalues
      dim_order.erase(std::next(dim_order.begin()));
      mappings.push_back(
        StoreMapping::default_mapping(output_ew.data(), options.front(), true /*exact*/));
      mappings.back().policy().ordering.set_custom_order(dim_order);

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
  LEGATE_ABORT("Unsupported task id: " + std::to_string(task_id));
  return {};
}

namespace {

// Use an accessor type with the maximum number of dimensions for the size approximation
using ACC_TYPE = legate::AccessorRO<std::int8_t, LEGATE_MAX_DIM>;

[[nodiscard]] constexpr std::size_t aligned_size(std::size_t size, std::size_t alignment)
{
  return (size + alignment - 1) / alignment * alignment;
}

constexpr std::size_t DEFAULT_ALIGNMENT = 16;

}  // namespace

std::optional<std::size_t> CuPyNumericMapper::allocation_pool_size(
  const legate::mapping::Task& task, legate::mapping::StoreTarget memory_kind)
{
  const auto task_id = static_cast<CuPyNumericOpCode>(task.task_id());

  switch (task_id) {
    case CUPYNUMERIC_ADVANCED_INDEXING: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_ARGWHERE: {
      auto&& input  = task.input(0);
      auto in_count = input.domain().get_volume();
      auto out_size = in_count * input.dim() * sizeof(std::int64_t);
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          return out_size;
        }
        case legate::mapping::StoreTarget::FBMEM: {
          return out_size + in_count * sizeof(std::int64_t);
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          return 0;
        }
      }
    }
    case CUPYNUMERIC_BATCHED_CHOLESKY: [[fallthrough]];
    case CUPYNUMERIC_GEEV: [[fallthrough]];
    case CUPYNUMERIC_POTRF: [[fallthrough]];
    // FIXME(wonchanl): These tasks actually don't need unbound pools on CPUs. They are being used
    // only to finish up the first implementation quickly
    case CUPYNUMERIC_QR: [[fallthrough]];
    case CUPYNUMERIC_SOLVE: [[fallthrough]];
    case CUPYNUMERIC_SVD: [[fallthrough]];
    case CUPYNUMERIC_SYEV: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return aligned_size(sizeof(std::int32_t), DEFAULT_ALIGNMENT);
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_BINARY_RED: {
      return memory_kind == legate::mapping::StoreTarget::FBMEM
               ? aligned_size(sizeof(bool), DEFAULT_ALIGNMENT)
               : 0;
    }
    case CUPYNUMERIC_CHOOSE: {
      return memory_kind == legate::mapping::StoreTarget::ZCMEM
               ? sizeof(ACC_TYPE) * task.num_inputs()
               : 0;
    }
    case CUPYNUMERIC_CONTRACT: {
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          auto&& lhs = task.reduction(0);
          if (lhs.type().code() != legate::Type::Code::FLOAT16) {
            return 0;
          }
          constexpr auto compute_buffer_size = [](auto&& arr) {
            return aligned_size(arr.domain().get_volume() * sizeof(float), DEFAULT_ALIGNMENT);
          };
          return compute_buffer_size(lhs) + compute_buffer_size(task.input(0)) +
                 compute_buffer_size(task.input(1));
        }
        case legate::mapping::StoreTarget::FBMEM: {
          return std::nullopt;
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          return 0;
        }
      }
    }
    case CUPYNUMERIC_CONVOLVE: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_DOT: {
      return memory_kind == legate::mapping::StoreTarget::FBMEM
               ? aligned_size(task.reduction(0).type().size(), DEFAULT_ALIGNMENT)
               : 0;
    }
    case CUPYNUMERIC_FFT: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_FLIP: {
      return memory_kind == legate::mapping::StoreTarget::ZCMEM
               ? sizeof(std::int32_t) * task.scalar(0).values<std::int32_t>().size()
               : 0;
    }
    case CUPYNUMERIC_HISTOGRAM: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_MATMUL: [[fallthrough]];
    case CUPYNUMERIC_MATVECMUL: {
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          const auto rhs1_idx = task.num_inputs() - 2;
          const auto rhs2_idx = task.num_inputs() - 1;
          auto&& rhs1         = task.input(rhs1_idx);
          if (rhs1.type().code() != legate::Type::Code::FLOAT16) {
            return 0;
          }
          constexpr auto compute_buffer_size = [](auto&& arr) {
            return aligned_size(arr.domain().get_volume() * sizeof(float), DEFAULT_ALIGNMENT);
          };
          return compute_buffer_size(rhs1) + compute_buffer_size(task.input(rhs2_idx));
        }
        // The GPU implementation needs no temporary allocations
        case legate::mapping::StoreTarget::FBMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::ZCMEM: {
          LEGATE_ABORT("GPU tasks shouldn't reach here");
          return 0;
        }
      }
    }
    case CUPYNUMERIC_MP_POTRF:
    case CUPYNUMERIC_MP_SOLVE: {
      switch (memory_kind) {
        case legate::mapping::StoreTarget::FBMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::ZCMEM: {
          return std::nullopt;
        }
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          LEGATE_ABORT("CPU tasks shouldn't reach here");
          return 0;
        }
      }
    }
    case CUPYNUMERIC_NONZERO: {
      auto&& input      = task.input(0);
      auto&& output     = task.output(0);
      auto in_count     = input.domain().get_volume();
      auto max_out_size = in_count * output.type().size() * input.dim();
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          return max_out_size;
        }
        case legate::mapping::StoreTarget::FBMEM: {
          return std::nullopt;
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          // The doubling here shouldn't be necessary, but the memory fragmentation seems to be
          // causing allocation failures even though there's enough space.
          return input.dim() * sizeof(std::int64_t*) * 2;
        }
      }
    }
    case CUPYNUMERIC_REPEAT: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        if (const auto scalar_repeats = task.scalar(1).value<bool>(); scalar_repeats) {
          return 0;
        }
        const auto axis      = task.scalar(0).value<std::uint32_t>();
        const auto in_domain = task.input(0).domain();
        const auto lo        = in_domain.lo();
        const auto hi        = in_domain.hi();
        const auto extent    = std::max(hi[axis] - lo[axis] + 1, legate::coord_t{0});
        return aligned_size(extent * sizeof(std::int64_t), DEFAULT_ALIGNMENT);
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_SCALAR_UNARY_RED: {
      return memory_kind == legate::mapping::StoreTarget::FBMEM
               ? aligned_size(task.reduction(0).type().size(), DEFAULT_ALIGNMENT)
               : 0;
    }
    case CUPYNUMERIC_SCAN_LOCAL: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      const auto output = task.output(0);
      const auto domain = output.domain();
      const auto ndim   = domain.dim;
      auto tmp_volume   = std::size_t{1};
      for (std::int32_t dim = 0; dim < ndim; ++dim) {
        tmp_volume *=
          std::max<>(legate::coord_t{0}, domain.rect_data[dim + ndim] - domain.rect_data[dim] + 1);
      }
      return aligned_size(tmp_volume * output.type().size(), output.type().alignment());
    }
    case CUPYNUMERIC_SELECT: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return aligned_size(sizeof(ACC_TYPE) * task.num_inputs(), DEFAULT_ALIGNMENT);
      }
      return 0;
    }
    case CUPYNUMERIC_SORT: {
      // There can be up to seven buffers on the zero-copy memory holding pointers and sizes
      auto compute_zc_alloc_size = [&]() -> std::optional<std::size_t> {
        return task.is_single_task() ? 0
                                     : 7 * task.get_launch_domain().get_volume() * sizeof(void*);
      };
      return memory_kind == legate::mapping::StoreTarget::ZCMEM ? compute_zc_alloc_size()
                                                                : std::nullopt;
    }
    case CUPYNUMERIC_UNIQUE: {
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          auto&& input = task.input(0);
          return input.domain().get_volume() * input.type().size();
        }
        case legate::mapping::StoreTarget::FBMEM: {
          return std::nullopt;
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          return task.get_launch_domain().get_volume() * sizeof(std::size_t);
        }
      }
    }
    case CUPYNUMERIC_UNIQUE_REDUCE: {
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          auto inputs       = task.inputs();
          auto elem_type    = inputs.front().type();
          auto total_volume = std::size_t{0};

          for (auto&& input : inputs) {
            total_volume += input.domain().get_volume();
          }
          return aligned_size(total_volume * elem_type.size(), elem_type.alignment());
        }
        // The GPU implementation needs no temporary allocations
        case legate::mapping::StoreTarget::FBMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::ZCMEM: {
          LEGATE_ABORT("GPU tasks shouldn't reach here");
          return 0;
        }
      }
    }
    case CUPYNUMERIC_WRAP: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return aligned_size(sizeof(bool), DEFAULT_ALIGNMENT);
    }
    case CUPYNUMERIC_ZIP: {
      using ACC = legate::AccessorRO<std::int8_t, LEGATE_MAX_DIM>;
      return memory_kind == legate::mapping::StoreTarget::ZCMEM
               ? (task.num_inputs() * sizeof(ACC_TYPE) + 15)
               : 0;
    }
  }
  LEGATE_ABORT("Unsupported task id: " + std::to_string(task_id));
  return {};
}

}  // namespace cupynumeric
