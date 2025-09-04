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

#include "cupynumeric/unary/convert.h"
#include "cupynumeric/unary/convert_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename Function, typename ARG, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES* out, const ARG* in)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  out[idx] = func(in[idx]);
}

template <typename Function, typename ReadAcc, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, Function func, WriteAcc out, ReadAcc in, Pitches pitches, Rect rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in[point]);
}

template <ConvertCode NAN_OP, Type::Code DST_TYPE, Type::Code SRC_TYPE, int DIM>
struct ConvertImplBody<VariantKind::GPU, NAN_OP, DST_TYPE, SRC_TYPE, DIM> {
  TaskContext context;
  explicit ConvertImplBody(TaskContext context) : context(context) {}

  using OP  = ConvertOp<NAN_OP, DST_TYPE, SRC_TYPE>;
  using SRC = type_of<SRC_TYPE>;
  using DST = type_of<DST_TYPE>;

  void operator()(OP func,
                  AccessorWO<DST, DIM> out,
                  AccessorRO<SRC, DIM> in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, func, outptr, inptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, func, out, in, pitches, rect);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ConvertTask::gpu_variant(TaskContext context)
{
  convert_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
