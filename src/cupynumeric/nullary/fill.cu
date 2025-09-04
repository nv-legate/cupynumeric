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

#include "cupynumeric/nullary/fill.h"
#include "cupynumeric/nullary/fill_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename ARG, typename ReadAcc>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, ARG* out, ReadAcc fill_value)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  out[idx] = fill_value[0];
}

template <typename WriteAcc, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, WriteAcc out, ReadAcc fill_value, Pitches pitches, Rect rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = fill_value[0];
}

template <typename VAL, int32_t DIM>
struct FillImplBody<VariantKind::GPU, VAL, DIM> {
  TaskContext context;
  explicit FillImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, 1> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();
    if (dense) {
      auto outptr = out.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, outptr, in);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, out, in, pitches, rect);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void FillTask::gpu_variant(TaskContext context)
{
  fill_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
