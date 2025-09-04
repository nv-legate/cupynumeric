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

#include "cupynumeric/nullary/window.h"
#include "cupynumeric/nullary/window_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <WindowOpCode OP_CODE>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(WindowOp<OP_CODE> gen, int64_t volume, double* out, int64_t lo)
{
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) {
    return;
  }
  out[idx] = gen(idx + lo);
}

template <WindowOpCode OP_CODE>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(WindowOp<OP_CODE> gen, int64_t volume, AccessorWO<double, 1> out, int64_t lo)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) {
    return;
  }
  idx += lo;
  Point<1> point(idx);
  out[point] = gen(idx);
}

template <WindowOpCode OP_CODE>
struct WindowImplBody<VariantKind::GPU, OP_CODE> {
  TaskContext context;
  explicit WindowImplBody(TaskContext context) : context(context) {}

  void operator()(
    const AccessorWO<double, 1>& out, const Rect<1>& rect, bool dense, int64_t M, double beta) const
  {
    WindowOp<OP_CODE> gen(M, beta);
    auto stream = context.get_task_stream();

    auto volume         = static_cast<int64_t>(rect.volume());
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      dense_kernel<OP_CODE>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(gen, volume, outptr, rect.lo[0]);
    } else {
      generic_kernel<OP_CODE>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(gen, volume, out, rect.lo[0]);
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void WindowTask::gpu_variant(TaskContext context)
{
  window_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
