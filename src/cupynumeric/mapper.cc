/* Copyright 2025 NVIDIA Corporation
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

#include <cupynumeric/stat/histogramdd.h>

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
    case CUPYNUMERIC_NDIMAGE_FOURIER_FILTER: {
      std::vector<StoreMapping> mappings;
      auto inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      return mappings;
    }
    case CUPYNUMERIC_CONVOLVE: {
      std::vector<StoreMapping> mappings;
      auto inputs = task.inputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.push_back(StoreMapping::default_mapping(inputs[1], options.front()));
      auto& input_mapping = mappings.back();
      for (uint32_t idx = 2; idx < inputs.size(); ++idx) {
        input_mapping.add_store(inputs[idx]);
      }
      return mappings;
    }
    case CUPYNUMERIC_FFT: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      mappings.push_back(StoreMapping::default_mapping(inputs[0], options.front()));
      mappings.push_back(
        StoreMapping::default_mapping(outputs[0], options.front(), true /*exact*/));
      return mappings;
    }
    case CUPYNUMERIC_TRANSPOSE_COPY_2D: {
      std::vector<StoreMapping> mappings;
      auto output = task.output(0);

      mappings.push_back(StoreMapping::default_mapping(
        output, options.front(), true /*exact*/, DimOrdering::fortran_order()));
      return mappings;
    }
    case CUPYNUMERIC_HISTOGRAMDD: {
      return HistogramDDTask::store_mappings(task, options);
    }
    case CUPYNUMERIC_MATMUL: {
      std::vector<StoreMapping> mappings;
      auto scalars = task.scalars();

      // if scalar parameter is passed _and_ it is 1
      // then use unbatched matmul mapping;
      // otherwise use batched matmul;
      //
      assert(scalars.size() == 0 || scalars[0].value<int>() == 0 || scalars[0].value<int>() == 1);
      if (scalars.size() == 0 || scalars[0].value<int>() == 0) {
        //
        // batched matmul:
        //
        auto inputA = task.input(1);
        auto inputB = task.input(2);

        mappings.push_back(StoreMapping::default_mapping(inputA, options.front(), true /*exact*/));
        mappings.back().policy().redundant = true;
        mappings.push_back(StoreMapping::default_mapping(inputB, options.front(), true /*exact*/));
        mappings.back().policy().redundant = true;

        auto outputC = task.output(0);

        mappings.push_back(StoreMapping::default_mapping(
          outputC, options.front(), true /*exact*/, DimOrdering::c_order()));

        return mappings;
      } else {
        //
        // unbatched matmul:
        //
        auto inputs     = task.inputs();
        auto reductions = task.reductions();
        for (auto& input : inputs) {
          mappings.push_back(StoreMapping::default_mapping(input, options.front(), true /*exact*/));
        }
        for (auto& reduction : reductions) {
          mappings.push_back(StoreMapping::default_mapping(
            reduction, options.front(), true /*exact*/, DimOrdering::c_order()));
        }
        return mappings;
      }
    }
    case CUPYNUMERIC_MATVECMUL:
    case CUPYNUMERIC_UNIQUE_REDUCE: {
      // TODO: Our actual requirements are a little less strict than this; we require each array or
      // vector to have a stride of 1 on at least one dimension.
      std::vector<StoreMapping> mappings;
      auto inputs     = task.inputs();
      auto reductions = task.reductions();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front(), true /*exact*/));
      }
      for (auto& reduction : reductions) {
        mappings.push_back(
          StoreMapping::default_mapping(reduction, options.front(), true /*exact*/));
      }
      return mappings;
    }
    case CUPYNUMERIC_QR:
    case CUPYNUMERIC_SVD:
    case CUPYNUMERIC_SYRK:
    case CUPYNUMERIC_GEMM:
    case CUPYNUMERIC_MP_POTRF:
    case CUPYNUMERIC_MP_QR:
    case CUPYNUMERIC_MP_SOLVE: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(
          input, options.front(), true /*exact*/, DimOrdering::fortran_order()));
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(
          output, options.front(), true /*exact*/, DimOrdering::fortran_order()));
      }
      return mappings;
    }
    case CUPYNUMERIC_POTRF:
    case CUPYNUMERIC_POTRS:
    case CUPYNUMERIC_TRSM:
    case CUPYNUMERIC_SOLVE: {
      std::vector<StoreMapping> mappings;
      auto dimensions = task.input(0).dim();

      // last 2 (matrix) dimensions col-major
      // batch dimensions 0, ..., dim-3 row-major
      std::vector<int32_t> dim_order;
      dim_order.push_back(dimensions - 2);
      dim_order.push_back(dimensions - 1);
      for (int32_t i = dimensions - 3; i >= 0; i--) {
        dim_order.push_back(i);
      }
      auto ordering = DimOrdering::custom_order(dim_order);

      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(
          StoreMapping::default_mapping(input, options.front(), true /*exact*/, ordering));
      }
      for (auto& output : outputs) {
        mappings.push_back(
          StoreMapping::default_mapping(output, options.front(), true /*exact*/, ordering));
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
      auto ordering = DimOrdering::custom_order(dim_order);

      mappings.push_back(
        StoreMapping::default_mapping(input_a, options.front(), true /*exact*/, ordering));

      // eigenvalue computation is optional
      if (task.outputs().size() > 1) {
        auto output_ev = task.output(1);
        mappings.push_back(
          StoreMapping::default_mapping(output_ev, options.front(), true /*exact*/, ordering));
      }

      // remove last dimension for eigenvalues
      dim_order.erase(std::next(dim_order.begin()));
      ordering = DimOrdering::custom_order(dim_order);
      mappings.push_back(
        StoreMapping::default_mapping(output_ew, options.front(), true /*exact*/, ordering));

      return mappings;
    }
    case CUPYNUMERIC_TRILU: {
      if (task.scalars().size() == 2) {
        return {};
      }
      // If we're here, this task was the post-processing for Cholesky.
      // So we will request fortran ordering
      std::vector<StoreMapping> mappings;
      auto input = task.input(0);
      mappings.push_back(StoreMapping::default_mapping(
        input, options.front(), true /*exact*/, DimOrdering::fortran_order()));
      return mappings;
    }
    case CUPYNUMERIC_SEARCHSORTED: {
      std::vector<StoreMapping> mappings;
      auto inputs    = task.inputs();
      auto reduction = task.reduction(0);

      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(
          input, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      mappings.push_back(StoreMapping::default_mapping(
        reduction, options.front(), true /*exact*/, DimOrdering::c_order()));
      return mappings;
    }
    case CUPYNUMERIC_SORT: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(
          input, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(
          output, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      return mappings;
    }
    case CUPYNUMERIC_SCAN_LOCAL: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(
          input, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(
          output, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      return mappings;
    }
    case CUPYNUMERIC_ALL2ALL_GATHER:
    case CUPYNUMERIC_ALL2ALL_SCATTER: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front()));
      }
      return mappings;
    }
    case CUPYNUMERIC_SCAN_GLOBAL: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(
          input, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(
          output, options.front(), true /*exact*/, DimOrdering::c_order()));
      }
      return mappings;
    }
    case CUPYNUMERIC_BITGENERATOR: {
      std::vector<StoreMapping> mappings;
      auto inputs  = task.inputs();
      auto outputs = task.outputs();
      for (auto& input : inputs) {
        mappings.push_back(StoreMapping::default_mapping(input, options.front(), true /*exact*/));
      }
      for (auto& output : outputs) {
        mappings.push_back(StoreMapping::default_mapping(output, options.front(), true /*exact*/));
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

// Compute the total batch size for a given array and number of fixed dimensions
// Fixed dimensions are the trailing dimensions that are not part of the batch
// For example, for batched matrix operations, num_fixed_dims would be 2 (the matrix dimensions)
template <typename Array>
[[nodiscard]] std::int64_t compute_batchsize(const Array& array, std::int32_t num_fixed_dims)
{
  auto domain                  = array.domain();
  std::int64_t batchsize_total = 1;
  auto dim                     = domain.dim;

  for (std::int32_t i = 0; i < dim - num_fixed_dims; ++i) {
    auto extent = domain.rect_data[i + dim] - domain.rect_data[i] + 1;
    batchsize_total *= extent;
  }

  return batchsize_total;
}

}  // namespace

std::optional<std::size_t> CuPyNumericMapper::allocation_pool_size(
  const legate::mapping::Task& task, legate::mapping::StoreTarget memory_kind)
{
  const auto task_id = static_cast<CuPyNumericOpCode>(task.task_id());

  switch (task_id) {
    case CUPYNUMERIC_ADVANCED_INDEXING: {
      auto&& input      = task.input(0);
      auto input_volume = input.domain().get_volume();

      // In the worst case (all boolean indices are true) the output has at
      // most input_volume elements.  element_size already encodes the full
      // per-element size: sizeof(VAL) for a GET and sizeof(Point<DIM>) for a
      // SET (the is_set flag is a task scalar, not reflected in the store
      // dimensionality visible at mapping time).  There is no extra dimension
      // factor — multiplying by input_dim would overestimate by DIM× for GET
      // and DIM²× for SET.
      auto element_size = task.output(0).type().size();
      auto max_out_size = aligned_size(input_volume * element_size, DEFAULT_ALIGNMENT);

      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: {
          return max_out_size;
        }
        case legate::mapping::StoreTarget::FBMEM: {
          // TEMPORARY WORKAROUND: drop the FBMEM upper-bound pool request so
          // Legate allocates the actual buffers at task execution time. The
          // upper-bound calculation below assumes 100% mask density
          // (max_out_size = input_volume * element_size), which significantly
          // overestimates real usage for typical sparse masks and causes OOM
          // on large microbenchmarks even when the actual peak fits.
          //
          // Restore the original calculation once the scan-buffer / output
          // sizing is fixed in advanced_indexing.cu (tiled scan or
          // count-then-copy implementation).
          //
          // Original implementation (restore when structural fix lands):
          //   // Add extra buffer for intermediate offset array calculation
          //   const auto offset_size =
          //     aligned_size(input_volume * sizeof(std::uint64_t), DEFAULT_ALIGNMENT);
          //
          //   // Consider min buffer for thrust scan temps (necessary for smaller task inputs)
          //   constexpr std::size_t MIN_THRUST_SCAN_TEMP_SIZE = 2048;
          //
          //   return MIN_THRUST_SCAN_TEMP_SIZE + offset_size + max_out_size;
          return std::nullopt;
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          return 0;
        }
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
          auto reduction_size = aligned_size(sizeof(std::int64_t), DEFAULT_ALIGNMENT);
          return out_size + in_count * sizeof(std::int64_t) + reduction_size;
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          return 0;
        }
      }
    }
    case CUPYNUMERIC_TRSM: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        auto a_array                 = task.input(0);
        std::int64_t batchsize_total = compute_batchsize(a_array, 2);
        return batchsize_total > 1
                 ? aligned_size(batchsize_total * sizeof(std::int32_t), DEFAULT_ALIGNMENT)
                 : 0;
      }
      return 0;
    }
    case CUPYNUMERIC_GEEV: [[fallthrough]];
    // FIXME(wonchanl): These tasks actually don't need unbound pools on CPUs. They are being used
    // only to finish up the first implementation quickly
    case CUPYNUMERIC_QR: [[fallthrough]];
    case CUPYNUMERIC_SVD: [[fallthrough]];
    case CUPYNUMERIC_SYEV: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        auto a_array                 = task.input(0);
        std::int64_t batchsize_total = compute_batchsize(a_array, 2);
        return aligned_size(batchsize_total * sizeof(std::int32_t), DEFAULT_ALIGNMENT);
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_POTRF: [[fallthrough]];
    case CUPYNUMERIC_POTRS: [[fallthrough]];
    case CUPYNUMERIC_SOLVE: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        auto a_array                 = task.input(0);
        std::int64_t batchsize_total = compute_batchsize(a_array, 2);
        // additional space for batchsize*2*pointers to store the pointers to the matrices and the
        // right-hand sides
        return aligned_size(batchsize_total * (sizeof(std::int32_t) + 2 * sizeof(void*)),
                            DEFAULT_ALIGNMENT);
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
    case CUPYNUMERIC_NDIMAGE_FOURIER_FILTER:
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
    case CUPYNUMERIC_GATHER: {
      return 0;
    }
    case CUPYNUMERIC_SCATTER: {
      return 0;
    }
    case CUPYNUMERIC_HISTOGRAM: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_HISTOGRAMDD: {
      return HistogramDDTask::allocation_pool_size(task, memory_kind);
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
    case CUPYNUMERIC_MGRID: {
      return 0;
    }
    case CUPYNUMERIC_MP_QR:
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
      auto&& input  = task.input(0);
      auto&& output = task.output(0);
      switch (memory_kind) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::FBMEM: {
          // allocate maximum available memory to prevent fragmentation
          // https://github.com/nv-legate/legate/issues/985
          return std::nullopt;
        }
        case legate::mapping::StoreTarget::ZCMEM: {
          // The doubling here shouldn't be necessary, but the memory fragmentation seems to be
          // causing allocation failures even though there's enough space.
          return input.dim() * sizeof(std::int64_t*) * 2;
        }
      }
    }
    case CUPYNUMERIC_PAD: {
      // PAD task doesn't need special allocation
      return 0;
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
      // Two Z_COPY_MEM buffers: AccessorRO array (always) + const int64_t* array (dense path).
      return memory_kind == legate::mapping::StoreTarget::ZCMEM
               ? aligned_size(task.num_inputs() * sizeof(ACC_TYPE), DEFAULT_ALIGNMENT) +
                   aligned_size(task.num_inputs() * sizeof(const int64_t*), DEFAULT_ALIGNMENT)
               : 0;
    }
    case CUPYNUMERIC_ZIPGATHER: {
      // Two scratch buffers:
      //  - FBMEM: AccessorRO table for the sparse / general path, copied H2D
      //    before the kernel launch.
      //  - ZCMEM: const int64_t* pointer table for the dense path, written
      //    directly by the host and read once on the device.
      const auto num_index_arrays = task.num_inputs() - 1;
      if (memory_kind == legate::mapping::StoreTarget::FBMEM) {
        return aligned_size(num_index_arrays * sizeof(ACC_TYPE), DEFAULT_ALIGNMENT);
      }
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return aligned_size(num_index_arrays * sizeof(const int64_t*), DEFAULT_ALIGNMENT);
      }
      return 0;
    }
    case CUPYNUMERIC_ZIPSCATTER: {
      // Mirrors CUPYNUMERIC_ZIPGATHER but the result is re-added as the last
      // input for dependency tracking, so index_arrays = num_inputs() - 2.
      const auto num_index_arrays = task.num_inputs() - 2;
      if (memory_kind == legate::mapping::StoreTarget::FBMEM) {
        return aligned_size(num_index_arrays * sizeof(ACC_TYPE), DEFAULT_ALIGNMENT);
      }
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return aligned_size(num_index_arrays * sizeof(const int64_t*), DEFAULT_ALIGNMENT);
      }
      return 0;
    }
    case CUPYNUMERIC_IN1D: {
      if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
        return 0;
      }
      return std::nullopt;
    }
    case CUPYNUMERIC_ALL2ALL_GATHER:
    case CUPYNUMERIC_ALL2ALL_SCATTER: {
      const auto num_ranks = task.get_launch_domain().get_volume();

      // Single-rank tasks dispatch to the local GatherTask / ScatterTask
      // kernels, which need no shuffle scratch.
      if (num_ranks <= 1) {
        return 0;
      }

      const auto elem_size      = task.input(0).type().size();
      const auto staging_factor = task.scalar(0).value<double>();
      const auto global_index   = task.scalar(1).value<std::uint64_t>();

      if (!(staging_factor > 0.0)) {
        LEGATE_ABORT("CUPYNUMERIC_ALL2ALL_STAGING_FACTOR must be a positive finite value, got ",
                     staging_factor);
      }

      // `avg_per_rank_count` is the partitioned-case basis (ceil(V / N));
      // `local_index_count` is this task's actual slice and equals the
      // full global volume when the partitioner replicates the indices.
      const std::size_t avg_per_rank_count =
        static_cast<std::size_t>((global_index + num_ranks - 1) / num_ranks);
      const std::size_t local_index_count =
        static_cast<std::size_t>(task.input(1).domain().get_volume());
      const std::size_t max_staging_bytes = std::max<std::size_t>(
        elem_size,
        static_cast<std::size_t>(staging_factor * static_cast<double>(avg_per_rank_count) *
                                 static_cast<double>(elem_size)));

      const auto per_rank_u64 = aligned_size(num_ranks * sizeof(std::uint64_t), DEFAULT_ALIGNMENT);

      switch (memory_kind) {
        case legate::mapping::StoreTarget::ZCMEM: {
          // Partition rects + allreduce buf + three num_ranks-sized
          // `ShuffleDescriptor` uint64 arrays + the two GPU-visible
          // num_ranks-sized `RoundSchedule` tables (within_round_send_prefix,
          // peer_src_offsets). The other RoundSchedule tables are
          // host-only std::vector and don't consume ZCMEM.
          constexpr std::size_t rect_bytes_max = sizeof(legate::Rect<LEGATE_MAX_DIM>);
          const auto partition_rects_bytes =
            aligned_size(num_ranks * rect_bytes_max, DEFAULT_ALIGNMENT);
          const auto local_rect_bytes = aligned_size(rect_bytes_max, DEFAULT_ALIGNMENT);
          const auto allreduce_buf_bytes =
            aligned_size(2 * sizeof(std::uint64_t), DEFAULT_ALIGNMENT);
          constexpr std::size_t kNumShuffleDescriptorU64Buffers = 3;
          constexpr std::size_t kNumRoundScheduleZcmemTables    = 2;
          return partition_rects_bytes + local_rect_bytes +
                 kNumShuffleDescriptorU64Buffers * per_rank_u64 +
                 kNumRoundScheduleZcmemTables * per_rank_u64 + allreduce_buf_bytes;
        }
        case legate::mapping::StoreTarget::FBMEM: {
          // Per-rank FBMEM terms include local request buffers, small metadata,
          // CUB temp, three round-offset buffers, and two staging buffers.
          //
          // CUB histogram temp upper bound: each SM keeps a privatized
          // `num_bins * sizeof(uint32_t)` counters. `num_bins` here is
          // `num_ranks` (one bin per peer). We cannot query the actual SM
          // count from the mapper, so use a conservative upper bound.
          constexpr std::size_t MAX_POSSIBLE_SMS = 4096;

          const auto local_terms =
            aligned_size(local_index_count * sizeof(std::uint64_t), DEFAULT_ALIGNMENT) +
            aligned_size(local_index_count * sizeof(int), DEFAULT_ALIGNMENT);
          const auto staging_per_buffer =
            aligned_size(std::max(max_staging_bytes, num_ranks * elem_size), DEFAULT_ALIGNMENT);
          // `round_request_positions` and `round_{send,recv}_offsets` are
          // laid out as `num_ranks * max_elems_per_peer` slots, so cap by
          // `min(num_ranks * avg_per_rank_count, max_staging_bytes / elem_size)`.
          const auto round_offset_buf_bound = aligned_size(
            std::min<std::size_t>(num_ranks * avg_per_rank_count,
                                  max_staging_bytes / std::max<std::size_t>(elem_size, 1)) *
              sizeof(std::uint64_t),
            DEFAULT_ALIGNMENT);
          // Overestimate sizeof(LinearizedRectInfo<DIM>) using the
          // largest-DIM Point + Pitches representable.
          constexpr std::size_t rect_info_bytes_max =
            sizeof(legate::Point<LEGATE_MAX_DIM>) + LEGATE_MAX_DIM * sizeof(std::int64_t);
          const auto cub_temp_budget = num_ranks * sizeof(std::uint32_t) * MAX_POSSIBLE_SMS;
          const auto small_terms =
            aligned_size(num_ranks * rect_info_bytes_max, DEFAULT_ALIGNMENT) + per_rank_u64 +
            cub_temp_budget;
          // Round-local FB buffers in `local_gather_and_exchange`:
          //   - 3 uint64 offset buffers sized `num_ranks * max_elems_per_peer`:
          //     `round_request_positions_buf`, `round_send_offsets`,
          //     `round_recv_offsets` (each bounded by `round_offset_buf_bound`).
          //   - 2 byte staging buffers sized `num_ranks * max_elems_per_peer * elem_size`:
          //     `send_staging`, `recv_staging` (each bounded by `staging_per_buffer`).
          constexpr std::size_t kNumRoundOffsetBuffers = 3;
          constexpr std::size_t kNumStagingBuffers     = 2;
          return local_terms + kNumRoundOffsetBuffers * round_offset_buf_bound +
                 kNumStagingBuffers * staging_per_buffer + small_terms;
        }
      }
      // SYSMEM / SOCKETMEM are not queried: only the GPU variant sets
      // `with_has_allocations(true)` (see all2all_gather.h / all2all_scatter.h).
      return std::nullopt;
    }
  }
  LEGATE_ABORT("Unsupported task id: " + std::to_string(task_id));
  return {};
}

}  // namespace cupynumeric
