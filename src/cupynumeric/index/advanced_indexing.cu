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

#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace cupynumeric {

namespace {

template <typename VAL, int IN_DIM, int OUT_DIM, typename OUT_TYPE, typename OFFSET_T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  getitem_masked(const size_t volume,
                 const AccessorRO<VAL, IN_DIM> input,
                 const AccessorRO<bool, IN_DIM> index,
                 Pitches<IN_DIM - 1> in_pitches,
                 Point<IN_DIM> in_lo,
                 OFFSET_T* offsets,
                 Buffer<OUT_TYPE, OUT_DIM> out,
                 Pitches<OUT_DIM - 1> out_pitches,
                 Point<OUT_DIM> out_lo)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }
  auto in_p = in_pitches.unflatten(tid, in_lo);
  if (index[in_p]) {
    auto out_p = out_pitches.unflatten(offsets[tid] - 1, out_lo);
    fill_out(out[out_p], in_p, input[in_p]);
  }
}

template <typename VAL, int IN_DIM, int OUT_DIM, typename OUT_TYPE, typename OFFSET_T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  getitem_masked_dense(const size_t volume,
                       const VAL* input,
                       const bool* index,
                       Pitches<IN_DIM - 1> in_pitches,
                       Point<IN_DIM> in_lo,
                       OFFSET_T* offsets,
                       Buffer<OUT_TYPE, OUT_DIM> out,
                       Pitches<OUT_DIM - 1> out_pitches,
                       Point<OUT_DIM> out_lo)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }
  auto in_p = in_pitches.unflatten(tid, in_lo);
  if (index[tid]) {
    auto out_p = out_pitches.unflatten(offsets[tid] - 1, out_lo);
    fill_out(out[out_p], in_p, input[tid]);
  }
}

// OFFSET_T is the per-element prefix-sum scratch: it holds counts of true mask entries, which are
// <= volume, so the caller passes uint32 when the tile fits in 2^32 elements (halving this
// full-volume buffer) and uint64 otherwise. Returns the offsets buffer and the total count of
// selected elements (the last prefix sum).
template <typename OFFSET_T, int DIM>
std::pair<OFFSET_T*, uint64_t> calculate_offsets(cudaStream_t stream,
                                                 const size_t volume,
                                                 const bool index_dense,
                                                 const AccessorRO<bool, DIM>& index,
                                                 const Pitches<DIM - 1>& in_pitches,
                                                 const Rect<DIM>& rect)
{
  const auto& in_lo    = rect.lo;
  auto buffer          = create_buffer<OFFSET_T, 1>(volume, legate::Memory::Kind::GPU_FB_MEM);
  OFFSET_T* buffer_ptr = buffer.ptr(0);

  // perform an inclusive scan to compute indices of elements to copy to output
  auto alloc   = ThrustAllocator(legate::Memory::Kind::GPU_FB_MEM);
  auto exe_pol = DEFAULT_POLICY(alloc).on(stream);

  if (index_dense) {
    auto index_ptr   = index.ptr(rect);
    auto cast_lambda = [] __host__ __device__(bool x) -> OFFSET_T {
      return static_cast<OFFSET_T>(x);
    };
    auto cast_iter = thrust::make_transform_iterator(index_ptr, cast_lambda);

    thrust::inclusive_scan(exe_pol, cast_iter, cast_iter + volume, buffer_ptr);
  } else {
    auto get_index_lambda = [index, in_pitches, in_lo] __host__ __device__(uint64_t i) -> OFFSET_T {
      return static_cast<OFFSET_T>(index[in_pitches.unflatten(i, in_lo)]);
    };
    auto get_index_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<uint64_t>(0), get_index_lambda);

    thrust::inclusive_scan(exe_pol, get_index_iter, get_index_iter + volume, buffer_ptr);
  }

  // Get the last element of the buffer to compute the number of selected elements
  OFFSET_T last_offset;
  CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
    &last_offset, buffer_ptr + volume - 1, sizeof(OFFSET_T), cudaMemcpyDeviceToHost, stream));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  // It's safe to "leak" the buffer, as the runtime destroys the buffer once the task is done.
  return {buffer_ptr, last_offset};
}

template <typename VAL, int IN_DIM, int OUT_DIM, typename OUT_TYPE, typename OFFSET_T>
void launch_getitem(cudaStream_t stream,
                    const size_t volume,
                    const bool dense,
                    const AccessorRO<VAL, IN_DIM>& input,
                    const AccessorRO<bool, IN_DIM>& index,
                    const Pitches<IN_DIM - 1>& in_pitches,
                    const Rect<IN_DIM>& rect,
                    OFFSET_T* offsets,
                    Buffer<OUT_TYPE, OUT_DIM>& out,
                    const Pitches<OUT_DIM - 1>& out_pitches,
                    const Point<OUT_DIM>& out_lo)
{
  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (dense) {
    getitem_masked_dense<VAL, IN_DIM, OUT_DIM, OUT_TYPE>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume,
                                                 input.ptr(rect),
                                                 index.ptr(rect),
                                                 in_pitches,
                                                 rect.lo,
                                                 offsets,
                                                 out,
                                                 out_pitches,
                                                 out_lo);
  } else {
    getitem_masked<VAL, IN_DIM, OUT_DIM, OUT_TYPE><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, input, index, in_pitches, rect.lo, offsets, out, out_pitches, out_lo);
  }
}

}  // namespace

template <Type::Code CODE, int IN_DIM, int OUT_DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::GPU, CODE, IN_DIM, OUT_DIM, OUT_TYPE> {
  TaskContext context;
  explicit AdvancedIndexingImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  static constexpr auto KEY_DIM = IN_DIM - OUT_DIM + 1;

  void operator()(PhysicalStore& out_arr,
                  const AccessorRO<VAL, IN_DIM>& input,
                  const AccessorRO<bool, IN_DIM>& index,
                  const Pitches<IN_DIM - 1>& in_pitches,
                  const Rect<IN_DIM>& rect,
                  const size_t volume,
                  const size_t /*skip_size*/) const
  {
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool index_dense = index.accessor.is_dense_row_major(rect);
    bool dense       = input.accessor.is_dense_row_major(rect) && index_dense;
#else
    // No dense execution if we're doing bounds checks
    bool index_dense = false;
    bool dense       = false;
#endif

    auto stream = context.get_task_stream();

    // volume == 0 is handled in advanced_indexing_template.inl by binding empty output data, so
    // the offsets[volume - 1] read in calculate_offsets is always valid here.
    LEGATE_ASSERT(volume > 0);

    // The full-volume prefix-sum scratch holds counts <= volume; use uint32 when the tile fits in
    // 2^32 elements (halving the buffer), else uint64. The output index type is unaffected.
    const bool narrow_offsets =
      volume <= static_cast<size_t>(std::numeric_limits<std::uint32_t>::max());

    std::uint32_t* offsets32 = nullptr;
    std::uint64_t* offsets64 = nullptr;
    uint64_t count           = 0;
    if (narrow_offsets) {
      std::tie(offsets32, count) =
        calculate_offsets<std::uint32_t>(stream, volume, index_dense, index, in_pitches, rect);
    } else {
      std::tie(offsets64, count) =
        calculate_offsets<std::uint64_t>(stream, volume, index_dense, index, in_pitches, rect);
    }

    // Compute the outer dimension of the output buffer
    uint64_t outer_dim_size = count;
    for (int32_t i = KEY_DIM; i < IN_DIM; i++) {
      outer_dim_size /= (1 + rect.hi[i] - rect.lo[i]);
    }

    // Allocate output buffer
    Point<OUT_DIM> extents;

    extents[0] = outer_dim_size;
    for (int32_t i = 1; i < OUT_DIM; i++) {
      size_t j = KEY_DIM + i - 1;

      extents[i] = (rect.hi[j] - rect.lo[j]) + 1;
    }

    auto out      = out_arr.create_output_buffer<OUT_TYPE, OUT_DIM>(extents, true);
    auto out_rect = out.get_bounds();
    auto out_lo   = out_rect.lo;

    Pitches<OUT_DIM - 1> out_pitches{};
    out_pitches.flatten(out_rect);

    if (count > 0) {
      if (narrow_offsets) {
        launch_getitem(stream,
                       volume,
                       dense,
                       input,
                       index,
                       in_pitches,
                       rect,
                       offsets32,
                       out,
                       out_pitches,
                       out_lo);
      } else {
        launch_getitem(stream,
                       volume,
                       dense,
                       input,
                       index,
                       in_pitches,
                       rect,
                       offsets64,
                       out,
                       out_pitches,
                       out_lo);
      }
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void AdvancedIndexingTask::gpu_variant(TaskContext context)
{
  advanced_indexing_template<VariantKind::GPU>(context);
}
}  // namespace cupynumeric
