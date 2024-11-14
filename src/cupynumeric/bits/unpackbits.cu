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

#include "cupynumeric/bits/unpackbits.h"
#include "cupynumeric/bits/unpackbits_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename UnpackOP, typename WriteAcc, typename ReadAcc, typename Pitches, typename Point>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume,
                 UnpackOP unpack,
                 WriteAcc out,
                 ReadAcc in,
                 Pitches in_pitches,
                 Point in_lo,
                 uint32_t axis)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto in_p = in_pitches.unflatten(idx, in_lo);
  unpack(out, in, in_p, axis);
}

template <int32_t DIM, Bitorder BITORDER>
struct UnpackbitsImplBody<VariantKind::GPU, DIM, BITORDER> {
  void operator()(const AccessorWO<uint8_t, DIM>& out,
                  const AccessorRO<uint8_t, DIM>& in,
                  const Rect<DIM>& in_rect,
                  const Pitches<DIM - 1>& in_pitches,
                  size_t in_volume,
                  uint32_t axis) const
  {
    Unpack<BITORDER> unpack{};
    auto stream         = get_cached_stream();
    const size_t blocks = (in_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      in_volume, unpack, out, in, in_pitches, in_rect.lo, axis);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void UnpackbitsTask::gpu_variant(TaskContext context)
{
  unpackbits_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
