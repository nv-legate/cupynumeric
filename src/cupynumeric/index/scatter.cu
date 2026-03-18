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

template <typename T, int32_t OUT_DIM, int32_t SRC_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scatter_kernel_dense(T* out,
                       const Point<OUT_DIM> out_strides,
                       const T* src,
                       const Point<OUT_DIM>* indices,
                       size_t volume)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }

  out[indices[tid].dot(out_strides)] = src[tid];
}

template <typename T, int32_t OUT_DIM, int32_t SRC_DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scatter_kernel(AccessorWO<T, OUT_DIM> out,
                 AccessorRO<T, SRC_DIM> src,
                 AccessorRO<Point<OUT_DIM>, SRC_DIM> indices,
                 Pitches<SRC_DIM - 1> pitches,
                 Point<SRC_DIM> lo,
                 size_t volume)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }

  auto p          = pitches.unflatten(tid, lo);
  out[indices[p]] = src[p];
}

template <typename T, int32_t OUT_DIM, int32_t SRC_DIM>
static void scatter_impl(PhysicalStore output,
                         PhysicalStore source,
                         PhysicalStore indices,
                         cudaStream_t stream)
{
  auto out_rect = output.shape<OUT_DIM>();
  auto src_rect = source.shape<SRC_DIM>();
  auto idx_rect = indices.shape<SRC_DIM>();

  assert(src_rect == idx_rect);

  auto out_acc = output.write_accessor<T, OUT_DIM>();
  auto src_acc = source.read_accessor<T, SRC_DIM>();
  auto idx_acc = indices.read_accessor<Point<OUT_DIM>, SRC_DIM>();

  Pitches<SRC_DIM - 1> src_pitches;
  size_t volume = src_pitches.flatten(src_rect);
  if (volume == 0) {
    return;
  }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
  // Check to see if this is dense or not
  bool dense = out_acc.accessor.is_dense_row_major(out_rect) &&
               src_acc.accessor.is_dense_row_major(src_rect) &&
               idx_acc.accessor.is_dense_row_major(idx_rect);
#else
  // No dense execution if we're doing bounds checks
  bool dense = false;
#endif

  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (dense) {
    size_t out_strides[OUT_DIM];
    auto outptr = out_acc.ptr(out_rect, out_strides);
    auto srcptr = src_acc.ptr(src_rect);
    auto idxptr = idx_acc.ptr(idx_rect);

    scatter_kernel_dense<T, OUT_DIM, SRC_DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      outptr, Point<OUT_DIM>{out_strides}, srcptr, idxptr, volume);
  } else {
    scatter_kernel<T, OUT_DIM, SRC_DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out_acc, src_acc, idx_acc, src_pitches, src_rect.lo, volume);
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <typename T>
struct ScatterDimDispatchGPU {
  PhysicalStore output;
  PhysicalStore source;
  PhysicalStore indices;
  cudaStream_t stream;

  template <int32_t OUT_DIM, int32_t SRC_DIM>
  void operator()()
  {
    scatter_impl<T, OUT_DIM, SRC_DIM>(output, source, indices, stream);
  }
};

struct ScatterTypeDispatchGPU {
  cudaStream_t stream;

  template <Type::Code CODE>
  void operator()(PhysicalStore output, PhysicalStore source, PhysicalStore indices) const
  {
    using T = type_of<CODE>;
    ScatterDimDispatchGPU<T> impl{output, source, indices, stream};
    cupynumeric::double_dispatch(source.dim(), output.dim(), impl);
  }
};

void ScatterTask::gpu_variant(TaskContext context)
{
  PhysicalStore output  = context.output(0);
  PhysicalStore source  = context.input(0);
  PhysicalStore indices = context.input(1);
  auto stream           = context.get_task_stream();

  type_dispatch(source.type().code(), ScatterTypeDispatchGPU{stream}, output, source, indices);
}

}  // namespace cupynumeric