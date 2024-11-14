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

#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/execution_policy/indexing/parallel_loop.h"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <class KERNEL, class Tag>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  parallel_loop_kernel(const size_t volume, KERNEL kernel, Tag tag)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  kernel(idx, tag);
}

template <class Tag>
struct ParallelLoopPolicy<VariantKind::GPU, Tag> {
  template <class RECT, class KERNEL>
  void operator()(const RECT& rect, KERNEL&& kernel)
  {
    const size_t volume = rect.volume();
    if (0 == volume) {
      return;
    }
    auto stream         = get_cached_stream();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    parallel_loop_kernel<<<blocks, THREADS_PER_BLOCK, 1, stream>>>(
      volume, std::forward<KERNEL>(kernel), Tag{});

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

}  // namespace cupynumeric
