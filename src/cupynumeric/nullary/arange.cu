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

#include "cupynumeric/nullary/arange.h"
#include "cupynumeric/nullary/arange_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) arange_kernel(
  const AccessorWO<VAL, 1> out, const coord_t lo, const VAL start, const VAL step, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) {
    return;
  }
  const auto p = lo + offset;
  out[p]       = static_cast<VAL>(p) * step + start;
}

template <typename VAL>
struct ArangeImplBody<VariantKind::GPU, VAL> {
  void operator()(const AccessorWO<VAL, 1>& out,
                  const Rect<1>& rect,
                  const VAL start,
                  const VAL step) const
  {
    const auto distance = rect.hi[0] - rect.lo[0] + 1;
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    arange_kernel<VAL>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, rect.lo[0], start, step, distance);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ArangeTask::gpu_variant(TaskContext context)
{
  arange_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
