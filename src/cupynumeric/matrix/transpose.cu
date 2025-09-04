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

#include "cupynumeric/matrix/transpose.h"
#include "cupynumeric/matrix/transpose_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

#define TILE_DIM 32
#define BLOCK_ROWS 8

template <typename VAL>
__global__ static void __launch_bounds__((TILE_DIM * BLOCK_ROWS), MIN_CTAS_PER_SM)
  transpose_2d_physical(const AccessorWO<VAL, 2> out,
                        const AccessorRO<VAL, 2> in,
                        const Point<2> lo,
                        const Point<2> hi)
{
  __shared__ VAL tile[TILE_DIM][TILE_DIM + 1 /*avoid bank conflicts*/];

  // These are reversed here for coalescing
  coord_t x = blockIdx.y * TILE_DIM + threadIdx.y;
  coord_t y = blockIdx.x * TILE_DIM + threadIdx.x;

  // Check to see if we hit our y-bounds, if so we can just mask off those threads
  if ((lo[1] + y) <= hi[1]) {
    // Check to see if we're going to hit our x-bounds while striding
    if ((lo[0] + (blockIdx.y + 1) * TILE_DIM - 1) <= hi[0]) {
// No overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = in[lo + Point<2>(x + i, y)];
      }
    } else {
// Overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((lo[0] + x + i) <= hi[0]) {
          tile[threadIdx.y + i][threadIdx.x] = in[lo + Point<2>(x + i, y)];
        }
      }
    }
  }

  // Make sure all the data is in shared memory
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  // Check to see if we hit our x-bounds, if so we can just mask off those threads
  if ((lo[0] + x) <= hi[0]) {
    // Check to see if we're going to hit our y-bounds while striding
    if ((lo[1] + (blockIdx.x + 1) * TILE_DIM - 1) <= hi[1]) {
// No overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        out[lo + Point<2>(x, y + i)] = tile[threadIdx.x][threadIdx.y + i];
      }
    } else {
// Overflow case
#pragma unroll
      for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((lo[1] + y + i) <= hi[1]) {
          out[lo + Point<2>(x, y + i)] = tile[threadIdx.x][threadIdx.y + i];
        }
      }
    }
  }
}

template <Type::Code CODE>
struct TransposeImplBody<VariantKind::GPU, CODE> {
  TaskContext context;
  explicit TransposeImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const Rect<2>& rect,
                  const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in) const
  {
    const coord_t m = (rect.hi[0] - rect.lo[0]) + 1;
    const coord_t n = (rect.hi[1] - rect.lo[1]) + 1;
    const dim3 blocks((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM, 1);
    const dim3 threads(TILE_DIM, BLOCK_ROWS, 1);

    auto stream = context.get_task_stream();
    transpose_2d_physical<VAL><<<blocks, threads, 0, stream>>>(out, in, rect.lo, rect.hi);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void TransposeTask::gpu_variant(TaskContext context)
{
  transpose_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
