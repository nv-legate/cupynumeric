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

#include "cupynumeric/binary/binary_red.h"
#include "cupynumeric/binary/binary_red_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename Function, typename RES, typename ARG>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES out, const ARG* in1, const ARG* in2)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  if (!func(in1[idx], in2[idx])) {
    out.reduce<false /*EXCLUSIVE*/>(false);
  }
}

template <typename Function, typename RES, typename ReadAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) generic_kernel(
  size_t volume, Function func, RES out, ReadAcc in1, ReadAcc in2, Pitches pitches, Rect rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  if (!func(in1[point], in2[point])) {
    out.reduce<false /*EXCLUSIVE*/>(false);
  }
}

template <typename Buffer, typename RedAcc>
static __global__ void __launch_bounds__(1, 1) copy_kernel(Buffer result, RedAcc out)
{
  out.reduce(0, result.read());
}

template <BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryRedImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  using OP  = BinaryOp<OP_CODE, CODE>;
  using ARG = type_of<CODE>;

  template <typename AccessorRD>
  void operator()(OP func,
                  AccessorRD out,
                  AccessorRO<ARG, DIM> in1,
                  AccessorRO<ARG, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = get_cached_stream();
    DeviceScalarReductionBuffer<ProdReduction<bool>> result(stream);
    if (dense) {
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, func, result, in1ptr, in2ptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, func, result, in1, in2, pitches, rect);
    }

    copy_kernel<<<1, 1, 0, stream>>>(result, out);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void BinaryRedTask::gpu_variant(TaskContext context)
{
  binary_red_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
