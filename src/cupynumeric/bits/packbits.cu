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

#include "cupynumeric/bits/packbits.h"
#include "cupynumeric/bits/packbits_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename PackOP, typename WriteAcc, typename ReadAcc, typename Pitches, typename Point>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume,
                 PackOP pack,
                 WriteAcc out,
                 ReadAcc in,
                 Pitches pitches,
                 Point lo,
                 int64_t in_hi_axis,
                 uint32_t axis)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto out_p = pitches.unflatten(idx, lo);
  out[out_p] = pack(in, out_p, in_hi_axis, axis);
}

template <Type::Code CODE, int32_t DIM, Bitorder BITORDER>
struct PackbitsImplBody<VariantKind::GPU, CODE, DIM, BITORDER> {
  TaskContext context;
  explicit PackbitsImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorWO<uint8_t, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const Rect<DIM>& aligned_rect,
                  const Rect<DIM>& unaligned_rect,
                  const Pitches<DIM - 1>& aligned_pitches,
                  const Pitches<DIM - 1>& unaligned_pitches,
                  size_t aligned_volume,
                  size_t unaligned_volume,
                  int64_t in_hi_axis,
                  uint32_t axis) const
  {
    Pack<BITORDER, true /* ALIGNED */> op{};
    Pack<BITORDER, false /* ALIGNED */> op_unaligned{};

    auto stream = context.get_task_stream();
    if (aligned_volume > 0) {
      const size_t blocks = (aligned_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        aligned_volume, op, out, in, aligned_pitches, aligned_rect.lo, in_hi_axis, axis);
    }
    if (unaligned_volume > 0) {
      const size_t blocks = (unaligned_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(unaligned_volume,
                                                               op_unaligned,
                                                               out,
                                                               in,
                                                               unaligned_pitches,
                                                               unaligned_rect.lo,
                                                               in_hi_axis,
                                                               axis);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void PackbitsTask::gpu_variant(TaskContext context)
{
  packbits_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
