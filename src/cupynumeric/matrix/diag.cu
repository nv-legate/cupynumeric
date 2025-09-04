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

#include "cupynumeric/matrix/diag.h"
#include "cupynumeric/matrix/diag_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  diag_populate(const AccessorRW<VAL, 2> out,
                const AccessorRO<VAL, 2> in,
                const coord_t distance,
                const Point<2> start)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= distance) {
    return;
  }
  Point<2> p(start[0] + idx, start[1] + idx);
  out[p] = in[p];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  diag_extract(const AccessorRD<SumReduction<VAL>, true, DIM> out,
               const AccessorRO<VAL, DIM> in,
               const coord_t distance,
               const size_t volume,
               const size_t skip_size,
               const coord_t start,
               const size_t naxes,
               const Pitches<DIM - 1> m_pitches,
               const Rect<DIM> m_shape)
{
  const int idx = skip_size * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= volume) {
    return;
  }

  auto in_p  = m_pitches.unflatten(idx, m_shape.lo);
  auto out_p = in_p;
  for (coord_t d = 0; d < distance; ++d) {
    for (int i = DIM - naxes; i < DIM; i++) {
      in_p[i]  = start + d;
      out_p[i] = start + d;
    }
    auto v = in[in_p];
    out.reduce(out_p, v);
  }
}

template <Type::Code CODE, int DIM>
struct DiagImplBody<VariantKind::GPU, CODE, DIM, true> {
  TaskContext context;
  explicit DiagImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorRD<SumReduction<VAL>, true, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const coord_t start,
                  const Pitches<DIM - 1>& m_pitches,
                  const Rect<DIM>& m_shape,
                  const int naxes,
                  const coord_t distance) const
  {
    size_t skip_size = 1;

    for (int i = 0; i < naxes; i++) {
      auto diff = 1 + m_shape.hi[DIM - i - 1] - m_shape.lo[DIM - i - 1];
      if (diff != 0) {
        skip_size *= diff;
      }
    }

    const size_t volume    = m_shape.volume();
    const size_t loop_size = volume / skip_size + 1;

    const size_t blocks = (loop_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto stream = context.get_task_stream();
    diag_extract<VAL><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out, in, distance, volume, skip_size, start, naxes, m_pitches, m_shape);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

// not extract (create a new 2D matrix with diagonal from vector)
template <Type::Code CODE>
struct DiagImplBody<VariantKind::GPU, CODE, 2, false> {
  TaskContext context;
  explicit DiagImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorRO<VAL, 2>& in,
                  const AccessorRW<VAL, 2>& out,
                  const Point<2>& start,
                  const size_t distance)
  {
    const size_t blocks = (distance + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto stream         = context.get_task_stream();
    diag_populate<VAL><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out, in, distance, start);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void DiagTask::gpu_variant(TaskContext context)
{
  diag_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
