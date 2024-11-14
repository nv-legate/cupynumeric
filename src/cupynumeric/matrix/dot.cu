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

#include "cupynumeric/matrix/dot.h"
#include "cupynumeric/matrix/dot_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename Output, typename ReadAcc, typename Point, typename ACC>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM) reduction_kernel(
  size_t volume, Output out, ReadAcc rhs1, ReadAcc rhs2, Point origin, size_t iters, ACC identity)
{
  auto value = identity;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point = origin + offset;
      SumReduction<ACC>::fold<true>(value,
                                    static_cast<ACC>(rhs1[point]) * static_cast<ACC>(rhs2[point]));
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

template <typename Buffer, typename RedAcc>
static __global__ void __launch_bounds__(1, 1) copy_kernel(Buffer result, RedAcc out)
{
  out.reduce(0, result.read());
}

template <Type::Code CODE>
struct DotImplBody<VariantKind::GPU, CODE> {
  using VAL = type_of<CODE>;
  using ACC = acc_type_of<VAL>;

  template <typename AccessorRD>
  void operator()(AccessorRD out,
                  const AccessorRO<VAL, 1>& rhs1,
                  const AccessorRO<VAL, 1>& rhs2,
                  const Rect<1>& rect,
                  bool dense)
  {
    auto stream = get_cached_stream();

    const auto volume   = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    DeviceScalarReductionBuffer<SumReduction<ACC>> result(stream);
    size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(ACC);

    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      reduction_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, result, rhs1, rhs2, rect.lo, iters, SumReduction<ACC>::identity);
    } else {
      reduction_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        volume, result, rhs1, rhs2, rect.lo, 1, SumReduction<ACC>::identity);
    }

    copy_kernel<<<1, 1, 0, stream>>>(result, out);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void DotTask::gpu_variant(TaskContext context)
{
  dot_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
