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

#include "cupynumeric/ternary/where.h"
#include "cupynumeric/ternary/where_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, VAL* out, const bool* mask, const VAL* in1, const VAL* in2)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  out[idx] = mask[idx] ? in1[idx] : in2[idx];
}

template <typename WriteAcc, typename MaskAcc, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) generic_kernel(
  size_t volume, WriteAcc out, MaskAcc mask, ReadAcc in1, ReadAcc in2, Pitches pitches, Rect rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = mask[point] ? in1[point] : in2[point];
}

template <Type::Code CODE, int DIM>
struct WhereImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<bool, DIM> mask,
                  AccessorRO<VAL, DIM> in1,
                  AccessorRO<VAL, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    if (dense) {
      size_t volume = rect.volume();
      auto outptr   = out.ptr(rect);
      auto maskptr  = mask.ptr(rect);
      auto in1ptr   = in1.ptr(rect);
      auto in2ptr   = in2.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, outptr, maskptr, in1ptr, in2ptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, out, mask, in1, in2, pitches, rect);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void WhereTask::gpu_variant(TaskContext context)
{
  where_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
