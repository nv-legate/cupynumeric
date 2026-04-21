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

#include "cupynumeric/index/advanced_indexing.h"
#include "cupynumeric/index/advanced_indexing_template.inl"
#include "cupynumeric/utilities/thrust_util.h"
#include "cupynumeric/utilities/thrust_allocator.h"
#include "cupynumeric/cuda_help.h"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

namespace cupynumeric {

template <typename VAL, int DIM, typename OUT_TYPE>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  getitem_masked(const size_t volume,
                 const AccessorRO<VAL, DIM> input,
                 const AccessorRO<bool, DIM> index,
                 Pitches<DIM - 1> in_pitches,
                 Point<DIM> in_lo,
                 uint64_t* offsets,
                 Buffer<OUT_TYPE, DIM> out,
                 Pitches<DIM - 1> out_pitches,
                 Point<DIM> out_lo)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) {
    return;
  }
  auto in_p = in_pitches.unflatten(tid, in_lo);
  if (index[in_p]) {
    auto out_p = out_pitches.unflatten(offsets[tid] - 1, out_lo);
    fill_out(out[out_p], in_p, input[in_p]);
  }
}

template <typename VAL, int DIM, typename OUT_TYPE>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  getitem_masked_dense(const size_t volume,
                       const VAL* input,
                       const bool* index,
                       Pitches<DIM - 1> in_pitches,
                       Point<DIM> in_lo,
                       uint64_t* offsets,
                       Buffer<OUT_TYPE, DIM> out,
                       Pitches<DIM - 1> out_pitches,
                       Point<DIM> out_lo)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) {
    return;
  }
  auto in_p = in_pitches.unflatten(tid, in_lo);
  if (index[tid]) {
    auto out_p = out_pitches.unflatten(offsets[tid] - 1, out_lo);
    fill_out(out[out_p], in_p, input[tid]);
  }
}

template <Type::Code CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::GPU, CODE, DIM, OUT_TYPE> {
  TaskContext context;
  explicit AdvancedIndexingImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(PhysicalStore& out_arr,
                  const AccessorRO<VAL, DIM>& input,
                  const AccessorRO<bool, DIM>& index,
                  const Pitches<DIM - 1>& in_pitches,
                  const Rect<DIM>& rect,
                  const size_t key_dim) const
  {
    const size_t volume = rect.volume();
    const auto in_lo    = rect.lo;

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool index_dense = index.accessor.is_dense_row_major(rect);
    bool dense       = input.accessor.is_dense_row_major(rect) && index_dense;
#else
    // No dense execution if we're doing bounds checks
    bool index_dense = false;
    bool dense       = false;
#endif

    auto buffer          = create_buffer<uint64_t, 1>(volume, legate::Memory::Kind::GPU_FB_MEM);
    uint64_t* buffer_ptr = buffer.ptr(Rect<1>{Point<1>{0}, Point<1>{volume}});

    // perform an inclusive scan to compute indices of elements to copy to output
    auto stream  = context.get_task_stream();
    auto alloc   = ThrustAllocator(legate::Memory::Kind::GPU_FB_MEM);
    auto exe_pol = DEFAULT_POLICY(alloc).on(stream);

    if (index_dense) {
      auto index_ptr   = index.ptr(rect);
      auto cast_lambda = [] __host__ __device__(bool x) -> uint64_t {
        return static_cast<uint64_t>(x);
      };
      auto cast_iter = thrust::make_transform_iterator(index_ptr, cast_lambda);

      thrust::inclusive_scan(exe_pol, cast_iter, cast_iter + volume, buffer_ptr);
    } else {
      auto get_index_lambda =
        [index, in_pitches, in_lo] __host__ __device__(uint64_t i) -> uint64_t {
        return static_cast<uint64_t>(index[in_pitches.unflatten(i, in_lo)]);
      };
      auto get_index_iter = thrust::make_transform_iterator(
        thrust::make_counting_iterator<uint64_t>(0), get_index_lambda);

      thrust::inclusive_scan(exe_pol, get_index_iter, get_index_iter + volume, buffer_ptr);
    }

    // Get the last element of the buffer to compute the size
    uint64_t count;
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      &count, buffer_ptr + volume - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

    // Compute the outer dimension of the output buffer
    uint64_t outer_dim_size = count;
    for (size_t i = key_dim; i < DIM; i++) {
      outer_dim_size /= (1 + rect.hi[i] - rect.lo[i]);
    }

    // Allocate output buffer
    Point<DIM> extents;
    extents[0] = outer_dim_size;
    for (size_t i = 0; i < DIM - key_dim; i++) {
      size_t j       = key_dim + i;
      extents[i + 1] = 1 + rect.hi[j] - rect.lo[j];
    }
    for (size_t i = DIM - key_dim + 1; i < DIM; i++) {
      extents[i] = 1;
    }

    auto out      = out_arr.create_output_buffer<OUT_TYPE, DIM>(extents, true);
    auto out_rect = out.get_bounds();
    auto out_lo   = out_rect.lo;

    Pitches<DIM - 1> out_pitches{};
    out_pitches.flatten(out_rect);

    const size_t blocks       = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const bool any_index_true = count > 0;

    if (any_index_true && dense) {
      auto input_ptr = input.ptr(rect);
      auto index_ptr = index.ptr(rect);

      getitem_masked_dense<VAL, DIM, OUT_TYPE><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, input_ptr, index_ptr, in_pitches, in_lo, buffer_ptr, out, out_pitches, out_lo);
    } else if (any_index_true) {
      getitem_masked<VAL, DIM, OUT_TYPE><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, input, index, in_pitches, in_lo, buffer_ptr, out, out_pitches, out_lo);
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void AdvancedIndexingTask::gpu_variant(TaskContext context)
{
  advanced_indexing_template<VariantKind::GPU>(context);
}
}  // namespace cupynumeric
