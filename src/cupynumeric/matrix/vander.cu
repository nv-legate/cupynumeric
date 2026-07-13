/* Copyright 2026 NVIDIA Corporation
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

#include "cupynumeric/matrix/vander.h"
#include "cupynumeric/matrix/vander_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename VAL>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  vander_kernel(AccessorWO<VAL, 2> out,
                AccessorRO<VAL, 2> in,
                Pitches<1> pitches,
                Point<2> lo,
                size_t volume,
                int64_t N,
                bool increasing)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto p              = pitches.unflatten(idx, lo);
  const int64_t power = increasing ? static_cast<int64_t>(p[1]) : (N - 1 - p[1]);
  out[p]              = vander_pow<VAL>(in[p], power);
}

template <Type::Code CODE>
struct VanderImplBody<VariantKind::GPU, CODE> {
  using VAL = type_of<CODE>;

  TaskContext context;
  explicit VanderImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in,
                  const Pitches<1>& pitches,
                  const Point<2>& lo,
                  size_t volume,
                  int64_t N,
                  bool increasing) const
  {
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();
    vander_kernel<VAL>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, in, pitches, lo, volume, N, increasing);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void VanderTask::gpu_variant(TaskContext context)
{
  vander_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
