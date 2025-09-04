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

#include "cupynumeric/unary/unary_op.h"
#include "cupynumeric/unary/unary_op_template.inl"

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

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_copy_kernel(size_t volume, VAL* out, const VAL* in)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  out[idx] = in[idx];
}

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_copy_kernel(size_t volume,
                      AccessorWO<VAL, DIM> out,
                      AccessorRO<VAL, DIM> in,
                      Pitches<DIM - 1> pitches,
                      Rect<DIM> rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = in[point];
}

template <UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct UnaryOpImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  TaskContext context;
  explicit UnaryOpImplBody(TaskContext context) : context(context) {}

  using OP  = UnaryOp<OP_CODE, CODE>;
  using ARG = typename OP::T;
  using RES = std::result_of_t<OP(ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRO<ARG, DIM> in,
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

template <typename VAL, int DIM>
struct PointCopyImplBody<VariantKind::GPU, VAL, DIM> {
  TaskContext context;
  explicit PointCopyImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> in,
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
      dense_copy_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, outptr, inptr);
    } else {
      generic_copy_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(volume, out, in, pitches, rect);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <typename Function, typename LHS, typename RHS1, typename RHS2>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel_multiout(size_t volume, Function func, LHS* lhs, const RHS1* rhs1, RHS2* rhs2)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  lhs[idx] = func(rhs1[idx], &rhs2[idx]);
}

template <typename Function,
          typename LHSAcc,
          typename RHS1Acc,
          typename RHS2Acc,
          typename Pitches,
          typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel_multiout(size_t volume,
                          Function func,
                          LHSAcc lhs,
                          RHS1Acc rhs1,
                          RHS2Acc rhs2,
                          Pitches pitches,
                          Rect rect)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto point = pitches.unflatten(idx, rect.lo);
  lhs[point] = func(rhs1[point], rhs2.ptr(point));
}

template <UnaryOpCode OP_CODE, Type::Code CODE, int DIM>
struct MultiOutUnaryOpImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  TaskContext context;
  explicit MultiOutUnaryOpImplBody(TaskContext context) : context(context) {}

  using OP   = MultiOutUnaryOp<OP_CODE, CODE>;
  using RHS1 = typename OP::RHS1;
  using RHS2 = typename OP::RHS2;
  using LHS  = std::result_of_t<OP(RHS1, RHS2*)>;

  void operator()(OP func,
                  AccessorWO<LHS, DIM> lhs,
                  AccessorRO<RHS1, DIM> rhs1,
                  AccessorWO<RHS2, DIM> rhs2,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();
    if (dense) {
      auto lhsptr  = lhs.ptr(rect);
      auto rhs1ptr = rhs1.ptr(rect);
      auto rhs2ptr = rhs2.ptr(rect);
      dense_kernel_multiout<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, func, lhsptr, rhs1ptr, rhs2ptr);
    } else {
      generic_kernel_multiout<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        volume, func, lhs, rhs1, rhs2, pitches, rect);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void UnaryOpTask::gpu_variant(TaskContext context)
{
  unary_op_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
