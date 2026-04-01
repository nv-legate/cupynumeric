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

#include "cupynumeric/index/scatter.h"
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/unary/unary_op.h"

namespace cupynumeric {

using namespace legate;

template <int32_t OUT_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scatter_kernel_dense(char* out,
                       const Point<OUT_DIM> out_strides,
                       const char* src,
                       const Point<OUT_DIM>* indices,
                       const size_t elem_size,
                       size_t volume)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }

  const auto out_idx = indices[tid].dot(out_strides);

  memcpy(out + out_idx, src + elem_size * tid, elem_size);
}

template <int32_t OUT_DIM, int32_t SRC_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scatter_kernel(char* out,
                 const Point<OUT_DIM> out_strides,
                 const char* src,
                 const Point<SRC_DIM> src_strides,
                 AccessorRO<Point<OUT_DIM>, SRC_DIM> indices,
                 Pitches<SRC_DIM - 1> pitches,
                 Point<SRC_DIM> lo,
                 const size_t elem_size,
                 size_t volume)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }

  auto p       = pitches.unflatten(tid, lo);
  auto out_idx = indices[p].dot(out_strides);
  auto src_idx = p.dot(src_strides);

  memcpy(out + out_idx, src + src_idx, elem_size);
}

template <int32_t DIM>
bool is_dense_row_major(const size_t strides[DIM], const Rect<DIM>& rect, size_t elem_size)
{
  size_t expected = elem_size;
  for (int d = DIM - 1; d >= 0; --d) {
    if (strides[d] != expected) {
      return false;
    }
    expected *= (rect.hi[d] - rect.lo[d] + 1);
  }
  return true;
}

template <int32_t OUT_DIM, int32_t SRC_DIM>
static void scatter_impl(PhysicalStore output,
                         PhysicalStore source,
                         PhysicalStore indices,
                         cudaStream_t stream)
{
  auto out_rect = output.shape<OUT_DIM>();
  auto src_rect = source.shape<SRC_DIM>();
  auto idx_rect = indices.shape<SRC_DIM>();

  assert(src_rect == idx_rect);

  // Using char to avoid needing to template on type information
  auto out_acc = output.write_accessor<char, OUT_DIM, false>();
  auto src_acc = source.read_accessor<char, SRC_DIM, false>();
  auto idx_acc = indices.read_accessor<Point<OUT_DIM>, SRC_DIM>();

  // Note: strides are already appropriately scaled to underlying type of output
  // so no need to scale them according to elem_size
  size_t out_strides[OUT_DIM];
  auto out_ptr = reinterpret_cast<char*>(out_acc.ptr(out_rect, out_strides));

  size_t src_strides[SRC_DIM];
  auto src_ptr = reinterpret_cast<const char*>(src_acc.ptr(src_rect, src_strides));

  const size_t elem_size = source.type().size();
  assert(elem_size == output.type().size());

  Pitches<SRC_DIM - 1> src_pitches;
  size_t volume = src_pitches.flatten(src_rect);
  if (volume == 0) {
    return;
  }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
  // Check to see if this is dense or not
  bool dense = is_dense_row_major<OUT_DIM>(out_strides, out_rect, elem_size) &&
               is_dense_row_major<SRC_DIM>(src_strides, src_rect, elem_size) &&
               idx_acc.accessor.is_dense_row_major(idx_rect);
#else
  // No dense execution if we're doing bounds checks
  bool dense = false;
#endif

  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (dense) {
    auto idx_ptr = idx_acc.ptr(idx_rect);

    scatter_kernel_dense<OUT_DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out_ptr, Point<OUT_DIM>{out_strides}, src_ptr, idx_ptr, elem_size, volume);
  } else {
    scatter_kernel<OUT_DIM, SRC_DIM>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out_ptr,
                                                 Point<OUT_DIM>{out_strides},
                                                 src_ptr,
                                                 Point<SRC_DIM>{src_strides},
                                                 idx_acc,
                                                 src_pitches,
                                                 src_rect.lo,
                                                 elem_size,
                                                 volume);
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

struct ScatterDimDispatchGPU {
  PhysicalStore output;
  PhysicalStore source;
  PhysicalStore indices;
  cudaStream_t stream;

  template <int32_t OUT_DIM, int32_t SRC_DIM>
  void operator()()
  {
    scatter_impl<OUT_DIM, SRC_DIM>(output, source, indices, stream);
  }
};

void ScatterTask::gpu_variant(TaskContext context)
{
  PhysicalStore output  = context.output(0);
  PhysicalStore source  = context.input(0);
  PhysicalStore indices = context.input(1);
  auto stream           = context.get_task_stream();
  auto src_dim          = source.dim();
  auto out_dim          = output.dim();
  ScatterDimDispatchGPU impl{std::move(output), std::move(source), std::move(indices), stream};

  cupynumeric::double_dispatch(src_dim, out_dim, impl);
}

}  // namespace cupynumeric