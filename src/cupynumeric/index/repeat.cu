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

#include "cupynumeric/index/repeat.h"
#include "cupynumeric/index/repeat_template.inl"
#include "cupynumeric/utilities/thrust_util.h"
#include "cupynumeric/cuda_help.h"

#include <thrust/scan.h>

namespace cupynumeric {

template <typename Output, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  count_repeat_kernel(const int64_t extent,
                      Output sum,
                      const AccessorRO<int64_t, DIM> repeats,
                      const Point<DIM> origin,
                      const int32_t axis,
                      const size_t iters,
                      Buffer<int64_t> offsets)
{
  uint64_t value = 0;
  for (size_t idx = 0; idx < iters; idx++) {
    const int64_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < extent) {
      auto p = origin;
      p[axis] += offset;
      auto val        = repeats[p];
      offsets[offset] = val;
      SumReduction<uint64_t>::fold<true>(value, val);
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  reduce_output(sum, value);
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(AccessorWO<VAL, DIM> out,
                const AccessorRO<VAL, DIM> in,
                int64_t repeats,
                const int32_t axis,
                const Point<DIM> out_lo,
                const Pitches<DIM - 1> pitches,
                const size_t volume)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto out_p = pitches.unflatten(idx, out_lo);
  auto in_p  = out_p;
  in_p[axis] /= repeats;
  out[out_p] = in[in_p];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(Buffer<VAL, DIM> out,
                const AccessorRO<VAL, DIM> in,
                const AccessorRO<int64_t, DIM> repeats,
                Buffer<int64_t> offsets,
                const int32_t axis,
                const Point<DIM> in_lo,
                const Pitches<DIM - 1> pitches,
                const int volume)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto in_p  = pitches.unflatten(idx, in_lo);
  auto out_p = in_p - in_lo;

  int64_t off_start = offsets[in_p[axis] - in_lo[axis]];
  int64_t off_end   = off_start + repeats[in_p];

  auto in_v = in[in_p];
  for (int64_t out_idx = off_start; out_idx < off_end; ++out_idx) {
    out_p[axis] = out_idx;
    out[out_p]  = in_v;
  }
}

template <Type::Code CODE, int DIM>
struct RepeatImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const int64_t repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    auto out_rect = out_array.shape<DIM>();
    auto out      = out_array.write_accessor<VAL, DIM>(out_rect);
    Pitches<DIM - 1> pitches{};

    auto out_volume   = pitches.flatten(out_rect);
    const auto blocks = (out_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto stream = get_cached_stream();
    repeat_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out, in, repeats, axis, out_rect.lo, pitches, out_volume);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }

  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const AccessorRO<int64_t, DIM>& repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    auto stream = get_cached_stream();

    Pitches<DIM - 1> pitches{};
    const auto volume = pitches.flatten(in_rect);

    // Compute offsets
    int64_t extent = in_rect.hi[axis] - in_rect.lo[axis] + 1;
    auto offsets   = create_buffer<int64_t>(Point<1>(extent), legate::Memory::Kind::Z_COPY_MEM);

    DeviceScalarReductionBuffer<SumReduction<uint64_t>> sum(stream);
    const size_t blocks_count = (extent + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(uint64_t);

    if (blocks_count > MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks_count + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      count_repeat_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        extent, sum, repeats, in_rect.lo, axis, iters, offsets);
    } else {
      count_repeat_kernel<<<blocks_count, THREADS_PER_BLOCK, shmem_size, stream>>>(
        extent, sum, repeats, in_rect.lo, axis, 1, offsets);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

    Point<DIM> out_extents = in_rect.hi - in_rect.lo + Point<DIM>::ONES();
    out_extents[axis]      = static_cast<coord_t>(sum.read(stream));

    auto out = out_array.create_output_buffer<VAL, DIM>(out_extents, true);

    auto p_offsets = offsets.ptr(0);
    thrust::exclusive_scan(DEFAULT_POLICY.on(stream), p_offsets, p_offsets + extent, p_offsets);

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    repeat_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out, in, repeats, offsets, axis, in_rect.lo, pitches, volume);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void RepeatTask::gpu_variant(TaskContext context)
{
  repeat_template<VariantKind::GPU>(context);
}
}  // namespace cupynumeric
