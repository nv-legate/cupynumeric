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

#include "cupynumeric/nullary/mgrid.h"
#include "cupynumeric/nullary/mgrid_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume,
                 AccessorWO<VAL, DIM> out,
                 const Pitches<DIM - 1> pitches,
                 const Rect<DIM> rect,
                 const std::array<VAL, DIM - 1> starts,
                 const std::array<VAL, DIM - 1> steps)
{
  const size_t idx = global_tid_1d();

  if (idx >= volume) {
    return;
  }

  const auto point = pitches.unflatten(idx, rect.lo);
  const auto d     = point[0];

  out[point] = starts[d] + steps[d] * static_cast<VAL>(point[d + 1]);
}

template <typename VAL, int DIM>
struct MGridImplBody<VariantKind::GPU, VAL, DIM> {
  TaskContext context;
  explicit MGridImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const std::array<VAL, DIM - 1>& starts,
                  const std::array<VAL, DIM - 1>& steps) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();

    generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, out, pitches, rect, starts, steps);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void MGridTask::gpu_variant(TaskContext context)
{
  mgrid_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
