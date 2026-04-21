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

#include "cupynumeric/index/gather.h"
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

// Dense fast-path: both source and index arrays are contiguous row-major.
template <int32_t SRC_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gather_kernel_dense(char* out_bytes,
                      const char* src_bytes,
                      size_t elem_size,
                      Point<SRC_DIM> src_byte_strides,
                      const Point<SRC_DIM>* idx_ptr,
                      size_t volume)
{
  const size_t tid = global_tid_1d();

  if (tid >= volume) {
    return;
  }

  memcpy(out_bytes + tid * elem_size, src_bytes + idx_ptr[tid].dot(src_byte_strides), elem_size);
}

template <int32_t OUT_DIM, int32_t SRC_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  gather_kernel_sparse(char* out_bytes,
                       const char* src_bytes,
                       size_t elem_size,
                       Point<SRC_DIM> src_byte_strides,
                       AccessorRO<Point<SRC_DIM>, OUT_DIM> idx_acc,
                       Pitches<OUT_DIM - 1> out_pitches,
                       Point<OUT_DIM> out_lo,
                       size_t volume)
{
  const size_t tid = global_tid_1d();

  if (tid >= volume) {
    return;
  }

  const auto p = out_pitches.unflatten(tid, out_lo);

  memcpy(out_bytes + tid * elem_size, src_bytes + idx_acc[p].dot(src_byte_strides), elem_size);
}

template <int32_t OUT_DIM, int32_t SRC_DIM>
static void gather_impl(char* out_bytes,
                        const char* src_bytes,
                        size_t elem_size,
                        Point<SRC_DIM> src_byte_strides,
                        AccessorRO<Point<SRC_DIM>, OUT_DIM> idx_acc,
                        bool dense,
                        Rect<OUT_DIM> out_rect,
                        Rect<OUT_DIM> idx_rect,
                        cudaStream_t stream)
{
  Pitches<OUT_DIM - 1> out_pitches;
  const size_t volume = out_pitches.flatten(out_rect);

  if (volume == 0) {
    return;
  }

  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (dense) {
    gather_kernel_dense<SRC_DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out_bytes, src_bytes, elem_size, src_byte_strides, idx_acc.ptr(idx_rect), volume);
  } else {
    gather_kernel_sparse<OUT_DIM, SRC_DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out_bytes, src_bytes, elem_size, src_byte_strides, idx_acc, out_pitches, out_rect.lo, volume);
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

struct GatherDimDispatchGPU {
  PhysicalStore output;
  PhysicalStore source;
  PhysicalStore indices;
  size_t elem_size;
  cudaStream_t stream;

  template <int32_t OUT_DIM, int32_t SRC_DIM>
  void operator()()
  {
    const auto out_rect = output.shape<OUT_DIM>();
    const auto idx_rect = indices.shape<OUT_DIM>();
    LEGATE_ASSERT(out_rect == idx_rect);
    const auto src_rect = source.shape<SRC_DIM>();

    const auto idx_acc = indices.read_accessor<Point<SRC_DIM>, OUT_DIM>(idx_rect);

    // Obtain raw byte pointer and strides for the source store.
    size_t src_bstrides[SRC_DIM];
    const auto src_byte_acc = source.read_accessor<int8_t, SRC_DIM, false>(src_rect);
    const char* src_bytes = reinterpret_cast<const char*>(src_byte_acc.ptr(src_rect, src_bstrides));

    // Obtain raw byte pointer and strides for the output store.
    size_t out_bstrides[OUT_DIM];
    const auto out_byte_acc = output.write_accessor<int8_t, OUT_DIM, false>(out_rect);
    char* out_bytes         = reinterpret_cast<char*>(out_byte_acc.ptr(out_rect, out_bstrides));

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Manual density check for source (int8_t strides aren't equivalent to
    // FieldAccessor::is_dense_row_major when elem_size > 1).
    auto is_dense_row_major = [&]() {
      size_t expected = elem_size;
      for (int d = SRC_DIM - 1; d >= 0; --d) {
        if (src_bstrides[d] != expected) {
          return false;
        }
        expected *= (src_rect.hi[d] - src_rect.lo[d] + 1);
      }
      return true;
    };
    const bool dense = is_dense_row_major() && idx_acc.accessor.is_dense_row_major(idx_rect);
#else
    const bool dense = false;
#endif

    gather_impl<OUT_DIM, SRC_DIM>(out_bytes,
                                  src_bytes,
                                  elem_size,
                                  Point<SRC_DIM>{src_bstrides},
                                  idx_acc,
                                  dense,
                                  out_rect,
                                  idx_rect,
                                  stream);
  }
};

void GatherTask::gpu_variant(TaskContext context)
{
  auto output       = context.output(0);
  auto source       = context.input(0);
  auto indices      = context.input(1);
  const auto stream = context.get_task_stream();

  const auto src_dim     = source.dim();
  const auto out_dim     = output.dim();
  const size_t elem_size = source.type().size();

  // legate::double_dispatch(dim1, dim2, f) calls f.operator<dim1, dim2>()
  // so pass (out_dim, src_dim) to get operator<OUT_DIM=out_dim, SRC_DIM=src_dim>.
  cupynumeric::double_dispatch(
    out_dim,
    src_dim,
    GatherDimDispatchGPU{
      std::move(output), std::move(source), std::move(indices), elem_size, stream});
}

}  // namespace cupynumeric
