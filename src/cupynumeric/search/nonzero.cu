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

#include "cupynumeric/search/nonzero.h"
#include "cupynumeric/search/nonzero_template.inl"
#include "cupynumeric/search/nonzero.cuh"

namespace cupynumeric {

template <typename Pitches, typename Point, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  nonzero_kernel(size_t volume,
                 AccessorRO<VAL, DIM> in,
                 Pitches pitches,
                 Point origin,
                 Buffer<int64_t> offsets,
                 Buffer<int64_t*> p_results)
{
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) {
    return;
  }

  auto point = pitches.unflatten(tid, origin);
  if (in[point] != VAL(0)) {
    auto offset = offsets[tid];
    for (int32_t dim = 0; dim < DIM; ++dim) {
      p_results[dim][offset] = point[dim];
    }
  }
}

template <Type::Code CODE, int32_t DIM>
struct NonzeroImplBody<VariantKind::GPU, CODE, DIM> {
  TaskContext context;
  explicit NonzeroImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void populate_nonzeros(const AccessorRO<VAL, DIM>& in,
                         const Pitches<DIM - 1>& pitches,
                         const Rect<DIM>& rect,
                         const size_t volume,
                         std::vector<Buffer<int64_t>>& results,
                         Buffer<int64_t>& offsets,
                         cudaStream_t stream)
  {
    auto ndims     = static_cast<int32_t>(results.size());
    auto p_results = create_buffer<int64_t*>(ndims, legate::Memory::Kind::Z_COPY_MEM);
    for (int32_t dim = 0; dim < ndims; ++dim) {
      p_results[dim] = results[dim].ptr(0);
    }

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    nonzero_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, in, pitches, rect.lo, offsets, p_results);
  }

  void operator()(std::vector<Array>& outputs,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume)
  {
    auto stream = context.get_task_stream();

    auto offsets =
      create_buffer<std::int64_t>(volume, legate::Memory::Kind::GPU_FB_MEM, sizeof(std::int64_t));
    auto size = compute_offsets(in, pitches, rect, volume, offsets, stream);

    std::vector<Buffer<int64_t>> results;
    for (auto& output : outputs) {
      results.push_back(output.create_output_buffer<int64_t, 1>(Point<1>(size), true));
    }

    if (size > 0) {
      populate_nonzeros(in, pitches, rect, volume, results, offsets, stream);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void NonzeroTask::gpu_variant(TaskContext context)
{
  nonzero_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
