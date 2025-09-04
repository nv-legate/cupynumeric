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

#include "cupynumeric/matrix/batched_cholesky.h"
#include "cupynumeric/matrix/potrf.h"
#include "cupynumeric/matrix/batched_cholesky_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

#define TILE_DIM 32
#define BLOCK_ROWS 8

template <>
void CopyBlockImpl<VariantKind::GPU>::operator()(void* dst, const void* src, size_t size)
{
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, context.get_task_stream());
}

template <typename VAL>
__global__ static void __launch_bounds__((TILE_DIM * BLOCK_ROWS), MIN_CTAS_PER_SM)
  transpose_2d_lower(VAL* out, int n)
{
  __shared__ VAL tile[TILE_DIM][TILE_DIM + 1 /*avoid bank conflicts*/];

  // The y dim is fast-moving index for coalescing
  auto r_block = blockIdx.x * TILE_DIM;
  auto c_block = blockIdx.y * TILE_DIM;
  auto r       = blockIdx.x * TILE_DIM + threadIdx.x;
  auto c       = blockIdx.y * TILE_DIM + threadIdx.y;
  auto stride  = BLOCK_ROWS;
  // The tile coordinates
  auto tr     = threadIdx.x;
  auto tc     = threadIdx.y;
  auto offset = r * n + c;

  // only execute across the upper diagonal
  // a single thread block will store the upper diagonal block into
  // a temp shared memory then set the block to zeros
  if (c_block >= r_block) {
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS, offset += stride) {
      if (r < n && (c + i) < n) {
        if (r <= (c + i)) {
          tile[tr][tc + i] = out[offset];
          // clear the upper diagonal entry
          out[offset] = 0;
        } else {
          tile[tr][tc + i] = 0;
        }
      }
    }

    // Make sure all the data is in shared memory
    __syncthreads();

    // Transpose the global coordinates, keep y the fast-moving index
    r      = blockIdx.y * TILE_DIM + threadIdx.x;
    c      = blockIdx.x * TILE_DIM + threadIdx.y;
    offset = r * n + c;

#pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS, offset += stride) {
      if (r < n && (c + i) < n) {
        if (r >= (c + i)) {
          out[offset] = tile[tc + i][tr];
        }
      }
    }
  }
}

template <Type::Code CODE>
struct BatchedTransposeImplBody<VariantKind::GPU, CODE> {
  TaskContext context;
  explicit BatchedTransposeImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(VAL* out, int n) const
  {
    const dim3 blocks((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM, 1);
    const dim3 threads(TILE_DIM, BLOCK_ROWS, 1);

    auto stream = context.get_task_stream();

    // CUDA Potrf produces the full matrix, we only want
    // the lower diagonal
    transpose_2d_lower<VAL><<<blocks, threads, 0, stream>>>(out, n);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void BatchedCholeskyTask::gpu_variant(TaskContext context)
{
  batched_cholesky_task_context_dispatch<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
