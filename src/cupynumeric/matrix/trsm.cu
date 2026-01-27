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

#include "cupynumeric/matrix/trsm.h"
#include "cupynumeric/matrix/trsm_template.inl"

#include "cupynumeric/cuda_help.h"

#include <vector>

namespace cupynumeric {

using namespace legate;

template <typename Trsm, typename TrsmBatched, typename VAL>
static inline void trsm_template(Trsm trsm,
                                 TrsmBatched trsm_batched,
                                 const VAL* a,
                                 const VAL* b,
                                 VAL* x,
                                 int32_t m,
                                 int32_t n,
                                 VAL alpha,
                                 cudaStream_t stream,
                                 int32_t num_blocks,
                                 int64_t a_block_stride,
                                 int64_t b_block_stride,
                                 bool side_left,
                                 bool lower,
                                 int32_t transa_op,
                                 bool unit_diag)
{
  auto cu_context = get_cublas();
  CHECK_CUBLAS(cublasSetStream(cu_context, stream));

  auto side   = side_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  auto uplo   = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  auto transa = (transa_op == 0) ? CUBLAS_OP_N : (transa_op == 1) ? CUBLAS_OP_T : CUBLAS_OP_C;
  auto diag   = unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  if (b != x) {
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemcpyAsync(x, b, sizeof(VAL) * m * n * num_blocks, cudaMemcpyDeviceToDevice, stream));
  }

  auto lda = side_left ? m : n;

  if (num_blocks == 1) {
    // Single solve - use regular (non-batched) API
    CHECK_CUBLAS(trsm(cu_context, side, uplo, transa, diag, m, n, &alpha, a, lda, x, m));
  } else {
    // Batched solve - use batched API
    // Create pointer arrays in zero-copy memory
    Buffer<VAL*> x_array       = create_buffer<VAL*>(num_blocks, legate::Memory::Z_COPY_MEM);
    Buffer<const VAL*> a_array = create_buffer<const VAL*>(num_blocks, legate::Memory::Z_COPY_MEM);

    for (int32_t i = 0; i < num_blocks; ++i) {
      x_array[i] = x + i * b_block_stride;
      a_array[i] = a + i * a_block_stride;
    }

    // Call batched TRSM
    CHECK_CUBLAS(trsm_batched(cu_context,
                              side,
                              uplo,
                              transa,
                              diag,
                              m,
                              n,
                              &alpha,
                              a_array.ptr(0),
                              lda,
                              x_array.ptr(0),
                              m,
                              num_blocks));
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(const float* a,
                  const float* b,
                  float* x,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t b_block_stride,
                  bool side,
                  bool lower,
                  int32_t transa,
                  bool unit_diagonal)
  {
    auto stream = context.get_task_stream();
    trsm_template(cublasStrsm,
                  cublasStrsmBatched,
                  a,
                  b,
                  x,
                  m,
                  n,
                  1.0F,
                  stream,
                  num_blocks,
                  a_block_stride,
                  b_block_stride,
                  side,
                  lower,
                  transa,
                  unit_diagonal);
  }
};

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(const double* a,
                  const double* b,
                  double* x,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t b_block_stride,
                  bool side,
                  bool lower,
                  int32_t transa,
                  bool unit_diagonal)
  {
    auto stream = context.get_task_stream();
    trsm_template(cublasDtrsm,
                  cublasDtrsmBatched,
                  a,
                  b,
                  x,
                  m,
                  n,
                  1.0,
                  stream,
                  num_blocks,
                  a_block_stride,
                  b_block_stride,
                  side,
                  lower,
                  transa,
                  unit_diagonal);
  }
};

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(const legate::Complex<float>* a_,
                  const legate::Complex<float>* b_,
                  legate::Complex<float>* x_,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t b_block_stride,
                  bool side,
                  bool lower,
                  int32_t transa,
                  bool unit_diagonal)
  {
    auto stream = context.get_task_stream();
    auto a      = reinterpret_cast<const cuComplex*>(a_);
    auto b      = reinterpret_cast<const cuComplex*>(b_);
    auto x      = reinterpret_cast<cuComplex*>(x_);

    trsm_template(cublasCtrsm,
                  cublasCtrsmBatched,
                  a,
                  b,
                  x,
                  m,
                  n,
                  make_float2(1.0, 0.0),
                  stream,
                  num_blocks,
                  a_block_stride,
                  b_block_stride,
                  side,
                  lower,
                  transa,
                  unit_diagonal);
  }
};

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(const legate::Complex<double>* a_,
                  const legate::Complex<double>* b_,
                  legate::Complex<double>* x_,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t b_block_stride,
                  bool side,
                  bool lower,
                  int32_t transa,
                  bool unit_diagonal)
  {
    auto stream = context.get_task_stream();
    auto a      = reinterpret_cast<const cuDoubleComplex*>(a_);
    auto b      = reinterpret_cast<const cuDoubleComplex*>(b_);
    auto x      = reinterpret_cast<cuDoubleComplex*>(x_);

    trsm_template(cublasZtrsm,
                  cublasZtrsmBatched,
                  a,
                  b,
                  x,
                  m,
                  n,
                  make_double2(1.0, 0.0),
                  stream,
                  num_blocks,
                  a_block_stride,
                  b_block_stride,
                  side,
                  lower,
                  transa,
                  unit_diagonal);
  }
};

/*static*/ void TrsmTask::gpu_variant(TaskContext context)
{
  trsm_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
