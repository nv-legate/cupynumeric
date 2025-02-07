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

#include "cupynumeric/sort/searchsorted.h"
#include "cupynumeric/sort/searchsorted_template.inl"
#include <cub/thread/thread_search.cuh>

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  searchsorted_kernel_min(AccessorRD<MinReduction<int64_t>, false, DIM> output_reduction,
                          AccessorRO<VAL, 1> sorted_array,
                          AccessorRO<VAL, DIM> values,
                          const Point<DIM> lo,
                          const Pitches<DIM - 1> pitches,
                          const size_t volume,
                          const size_t num_values,
                          const size_t global_offset)
{
  const size_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (v_idx >= num_values) {
    return;
  }

  auto v_point        = pitches.unflatten(v_idx, lo);
  int64_t lower_bound = cub::LowerBound(sorted_array.ptr(global_offset), volume, values[v_point]);

  if (lower_bound < volume) {
    output_reduction.reduce(v_point, lower_bound + global_offset);
  }
}

template <typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  searchsorted_kernel_max(AccessorRD<MaxReduction<int64_t>, false, DIM> output_reduction,
                          AccessorRO<VAL, 1> sorted_array,
                          AccessorRO<VAL, DIM> values,
                          const Point<DIM> lo,
                          const Pitches<DIM - 1> pitches,
                          const size_t volume,
                          const size_t num_values,
                          const size_t global_offset)
{
  const size_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (v_idx >= num_values) {
    return;
  }

  auto v_point        = pitches.unflatten(v_idx, lo);
  int64_t upper_bound = cub::UpperBound(sorted_array.ptr(global_offset), volume, values[v_point]);

  if (upper_bound > 0) {
    output_reduction.reduce(v_point, upper_bound + global_offset);
  }
}

template <Type::Code CODE, int32_t DIM>
struct SearchSortedImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(const PhysicalStore& input_array,
                  const PhysicalStore& input_values,
                  const PhysicalStore& output_positions,
                  const Rect<1>& rect_base,
                  const Rect<DIM>& rect_values,
                  const Pitches<DIM - 1> pitches,
                  const bool left,
                  const bool is_index_space,
                  const size_t volume,
                  const int64_t global_volume,
                  const size_t num_values)
  {
    auto input   = input_array.read_accessor<VAL, 1>(rect_base);
    auto input_v = input_values.read_accessor<VAL, DIM>(rect_values);
    assert(input.accessor.is_dense_arbitrary(rect_base));
    size_t offset = rect_base.lo[0];

    const size_t num_blocks_desired = (num_values + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream                     = get_cached_stream();
    if (left) {
      auto output_reduction =
        output_positions.reduce_accessor<MinReduction<int64_t>, false, DIM>(rect_values);
      searchsorted_kernel_min<VAL><<<num_blocks_desired, THREADS_PER_BLOCK, 0, stream>>>(
        output_reduction, input, input_v, rect_values.lo, pitches, volume, num_values, offset);
    } else {
      auto output_reduction =
        output_positions.reduce_accessor<MaxReduction<int64_t>, false, DIM>(rect_values);
      searchsorted_kernel_max<VAL><<<num_blocks_desired, THREADS_PER_BLOCK, 0, stream>>>(
        output_reduction, input, input_v, rect_values.lo, pitches, volume, num_values, offset);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SearchSortedTask::gpu_variant(TaskContext context)
{
  searchsorted_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
