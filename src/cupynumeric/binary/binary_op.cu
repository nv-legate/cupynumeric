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

#include "cupynumeric/binary/binary_op.h"
#include "cupynumeric/binary/binary_op_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename Function, typename LHS, typename RHS1, typename RHS2>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, LHS* out, const RHS1* in1, const RHS2* in2)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  out[idx] = func(in1[idx], in2[idx]);
}

template <typename Function,
          typename WriteAcc,
          typename ReadAcc1,
          typename ReadAcc2,
          typename Pitches,
          typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume,
                 Function func,
                 WriteAcc out,
                 ReadAcc1 in1,
                 ReadAcc2 in2,
                 Pitches pitches,
                 Rect rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in1[point], in2[point]);
}

template <BinaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct BinaryOpImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  TaskContext context;
  explicit BinaryOpImplBody(TaskContext context) : context(context) {}

  using OP   = BinaryOp<OP_CODE, CODE>;
  using RHS1 = type_of<CODE>;
  using RHS2 = rhs2_of_binary_op<OP_CODE, CODE>;
  using LHS  = std::result_of_t<OP(RHS1, RHS2)>;

  void operator()(OP func,
                  AccessorWO<LHS, DIM> out,
                  AccessorRO<RHS1, DIM> in1,
                  AccessorRO<RHS2, DIM> in2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    size_t volume       = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, func, outptr, in1ptr, in2ptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, func, out, in1, in2, pitches, rect);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void BinaryOpTask::gpu_variant(TaskContext context)
{
  binary_op_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
