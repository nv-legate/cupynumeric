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

#pragma once

#include "cupynumeric/execution_policy/reduction/scalar_reduction.h"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {
namespace scalar_reduction_impl {

template <class AccessorRD, class Kernel, class LHS, class Tag>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  scalar_unary_red_kernel(
    size_t volume, size_t iters, AccessorRD out, Kernel kernel, LHS identity, Tag tag)
{
  auto value = identity;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      kernel(value, offset, identity, tag);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Buffer, typename RedAcc>
static __global__ void __launch_bounds__(1, 1) copy_kernel(Buffer result, RedAcc out)
{
  out.reduce(0, result.read());
}

}  // namespace scalar_reduction_impl

template <class LG_OP, class Tag>
struct ScalarReductionPolicy<VariantKind::GPU, LG_OP, Tag> {
  TaskContext context;
  ScalarReductionPolicy(TaskContext tcontext) : context(tcontext) {}

  template <class AccessorRD, class LHS, class Kernel>
  void __attribute__((visibility("hidden"))) operator()(size_t volume,
                                                        AccessorRD& out,
                                                        const LHS& identity,
                                                        Kernel&& kernel)
  {
    if (0 == volume) {
      return;
    }

    auto stream = context.get_task_stream();

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeviceScalarReductionBuffer<LG_OP> result(stream);
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(LHS);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      scalar_reduction_impl::
        scalar_unary_red_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
          volume, iters, result, std::forward<Kernel>(kernel), identity, Tag{});
    } else {
      scalar_reduction_impl::
        scalar_unary_red_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
          volume, 1, result, std::forward<Kernel>(kernel), identity, Tag{});
    }
    scalar_reduction_impl::copy_kernel<<<1, 1, 0, stream>>>(result, out);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

}  // namespace cupynumeric
