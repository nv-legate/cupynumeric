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

#include <thrust/scan.h>

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/utilities/thrust_util.h"

namespace cupynumeric {

using namespace legate;

template <typename Output, typename Pitches, typename Point, typename VAL, int32_t DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  count_nonzero_kernel(size_t volume,
                       Output out,
                       AccessorRO<VAL, DIM> in,
                       Pitches pitches,
                       Point origin,
                       size_t iters,
                       Buffer<int64_t> offsets)
{
  uint64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < volume) {
      auto point      = pitches.unflatten(offset, origin);
      auto val        = static_cast<uint64_t>(in[point] != VAL(0));
      offsets[offset] = val;
      SumReduction<uint64_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(out, value);
}

static void exclusive_sum(int64_t* offsets, size_t volume, cudaStream_t stream)
{
  thrust::exclusive_scan(DEFAULT_POLICY.on(stream), offsets, offsets + volume, offsets);
}

template <typename VAL, int32_t DIM>
int64_t compute_offsets(const AccessorRO<VAL, DIM>& in,
                        const Pitches<DIM - 1>& pitches,
                        const Rect<DIM>& rect,
                        const size_t volume,
                        Buffer<int64_t>& offsets,
                        cudaStream_t stream)
{
  DeviceScalarReductionBuffer<SumReduction<uint64_t>> size(stream);

  const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(uint64_t);

  if (blocks >= MAX_REDUCTION_CTAS) {
    const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
    count_nonzero_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
      volume, size, in, pitches, rect.lo, iters, offsets);
  } else {
    count_nonzero_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
      volume, size, in, pitches, rect.lo, 1, offsets);
  }

  auto p_offsets = offsets.ptr(0);

  exclusive_sum(p_offsets, volume, stream);

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  return size.read(stream);
}

}  // namespace cupynumeric
