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

#include "cupynumeric/nullary/eye.h"
#include "cupynumeric/nullary/eye_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  eye_kernel(const AccessorWO<VAL, 2> out, const Point<2> start, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max) {
    return;
  }
  out[start[0] + offset][start[1] + offset] = 1;
}

template <typename VAL>
struct EyeImplBody<VariantKind::GPU, VAL> {
  void operator()(const AccessorWO<VAL, 2>& out,
                  const Point<2>& start,
                  const coord_t distance) const
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    eye_kernel<VAL><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, start, distance);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void EyeTask::gpu_variant(TaskContext context)
{
  eye_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
