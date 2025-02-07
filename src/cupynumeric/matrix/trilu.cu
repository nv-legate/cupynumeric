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

#include "cupynumeric/matrix/trilu.h"
#include "cupynumeric/matrix/trilu_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int32_t DIM, bool LOWER, bool C_ORDER>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  trilu_kernel(AccessorWO<VAL, DIM> out,
               AccessorRO<VAL, DIM> in,
               Pitches<DIM - 1, C_ORDER> pitches,
               Point<DIM> lo,
               size_t volume,
               int32_t k)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }

  if (LOWER) {
    auto p = pitches.unflatten(idx, lo);
    if (p[DIM - 2] + k >= p[DIM - 1]) {
      out[p] = in[p];
    } else {
      out[p] = 0;
    }
  } else {
    auto p = pitches.unflatten(idx, lo);
    if (p[DIM - 2] + k <= p[DIM - 1]) {
      out[p] = in[p];
    } else {
      out[p] = 0;
    }
  }
}

template <Type::Code CODE, int32_t DIM, bool LOWER>
struct TriluImplBody<VariantKind::GPU, CODE, DIM, LOWER> {
  using VAL = type_of<CODE>;

  template <bool C_ORDER>
  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1, C_ORDER>& pitches,
                  const Point<DIM>& lo,
                  size_t volume,
                  int32_t k) const
  {
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    trilu_kernel<VAL, DIM, LOWER, C_ORDER>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, in, pitches, lo, volume, k);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void TriluTask::gpu_variant(TaskContext context)
{
  trilu_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
