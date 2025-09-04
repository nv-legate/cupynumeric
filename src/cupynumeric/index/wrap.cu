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

#include "cupynumeric/index/wrap.h"
#include "cupynumeric/index/wrap_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename Output>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  check_kernel(Output out,
               const AccessorRO<int64_t, 1> indices,
               const int64_t start,
               const int64_t volume,
               const int64_t volume_base,
               const int64_t iters)
{
  bool value = false;
  for (size_t i = 0; i < iters; i++) {
    const auto idx = (i * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= volume) {
      break;
    }
    auto index_tmp = indices[idx + start];
    int64_t index  = index_tmp < 0 ? index_tmp + volume_base : index_tmp;
    bool val       = (index < 0 || index >= volume_base);
    SumReduction<bool>::fold<true>(value, val);
  }
  reduce_output(out, value);
}

template <int DIM, typename IND>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  wrap_kernel(const AccessorWO<Point<DIM>, 1> out,
              const int64_t start,
              const int64_t volume,
              const Pitches<0> pitches_out,
              const Point<1> out_lo,
              const Pitches<DIM - 1> pitches_base,
              const Point<DIM> base_lo,
              const size_t volume_base,
              const IND indices)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  const int64_t input_idx = compute_idx((idx + start), volume_base, indices);
  auto out_p              = pitches_out.unflatten(idx, out_lo);
  auto p                  = pitches_base.unflatten(input_idx, base_lo);
  out[out_p]              = p;
}

template <int DIM, typename IND>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  wrap_kernel_dense(Point<DIM>* out,
                    const int64_t start,
                    const int64_t volume,
                    const Pitches<DIM - 1> pitches_base,
                    const Point<DIM> base_lo,
                    const size_t volume_base,
                    const IND indices)
{
  const auto idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  const int64_t input_idx = compute_idx((idx + start), volume_base, indices);
  auto p                  = pitches_base.unflatten(input_idx, base_lo);
  out[idx]                = p;
}

// don't do anything when indices is a boolean
void check_out_of_bounds(const bool& indices,
                         const int64_t start,
                         const int64_t volume,
                         const int64_t volume_base,
                         cudaStream_t stream)
{
}

void check_out_of_bounds(const AccessorRO<int64_t, 1>& indices,
                         const int64_t start,
                         const int64_t volume,
                         const int64_t volume_base,
                         cudaStream_t stream)
{
  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(bool);
  DeviceScalarReductionBuffer<SumReduction<bool>> out_of_bounds(stream);

  if (blocks >= MAX_REDUCTION_CTAS) {
    const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
    check_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
      out_of_bounds, indices, start, volume, volume_base, iters);
  } else {
    check_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
      out_of_bounds, indices, start, volume, volume_base, 1);
  }
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  bool res = out_of_bounds.read(stream);
  if (res) {
    throw legate::TaskException("index is out of bounds in index array");
  }
}

template <int DIM>
struct WrapImplBody<VariantKind::GPU, DIM> {
  TaskContext context;
  explicit WrapImplBody(TaskContext context) : context(context) {}

  template <typename IND>
  void operator()(const AccessorWO<Point<DIM>, 1>& out,
                  const Pitches<0>& pitches_out,
                  const Rect<1>& rect_out,
                  const Pitches<DIM - 1>& pitches_base,
                  const Rect<DIM>& rect_base,
                  const bool dense,
                  const bool check_bounds,
                  const IND& indices) const
  {
    auto stream            = context.get_task_stream();
    const int64_t start    = rect_out.lo[0];
    const int64_t volume   = rect_out.volume();
    const auto volume_base = rect_base.volume();
    const size_t blocks    = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (check_bounds) {
      check_out_of_bounds(indices, start, volume, volume_base, stream);
    }

    if (dense) {
      auto outptr = out.ptr(rect_out);
      wrap_kernel_dense<DIM, IND><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        outptr, start, volume, pitches_base, rect_base.lo, volume_base, indices);
    } else {
      wrap_kernel<DIM, IND><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out,
                                                                      start,
                                                                      volume,
                                                                      pitches_out,
                                                                      rect_out.lo,
                                                                      pitches_base,
                                                                      rect_base.lo,
                                                                      volume_base,
                                                                      indices);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void WrapTask::gpu_variant(TaskContext context)
{
  wrap_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
