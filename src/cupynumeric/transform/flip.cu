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

#include "cupynumeric/transform/flip.h"
#include "cupynumeric/transform/flip_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename WriteAcc, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  flip_kernel(const size_t volume,
              WriteAcc out,
              ReadAcc in,
              Pitches pitches,
              Rect rect,
              Buffer<int32_t, 1> axes,
              const uint32_t num_axes)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto p = pitches.unflatten(idx, rect.lo);
  auto q = p;
  for (uint32_t idx = 0; idx < num_axes; ++idx) {
    q[axes[idx]] = rect.hi[axes[idx]] - q[axes[idx]];
  }
  out[p] = in[q];
}

template <Type::Code CODE, int32_t DIM>
struct FlipImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  legate::Span<const int32_t> axes) const

  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto num_axes       = axes.size();
    auto gpu_axes = create_buffer<int32_t>(num_axes, Memory::Kind::Z_COPY_MEM, sizeof(int32_t));
    for (uint32_t idx = 0; idx < num_axes; ++idx) {
      gpu_axes[idx] = axes[idx];
    }
    auto stream = get_cached_stream();
    flip_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, out, in, pitches, rect, gpu_axes, num_axes);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void FlipTask::gpu_variant(TaskContext context)
{
  flip_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
