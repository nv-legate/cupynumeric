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

#include "cupynumeric/matrix/matmul.h"
#include "cupynumeric/matrix/matmul_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

// NOTE:
// cuBLAS doesn't support row-major, so reverse the matrix order so it thinks things are
// column-major. Effectively we get NxM = NxK * KxM.
// Use the extended (*gemmEx) interface where possible, so we use tensor cores if they are available
// for this matrix shape and GPU.

template <>
struct MatMulImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit MatMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  float* lhs,
                  const float* rhs1,
                  const float* rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.0;
    // lhs_overwritable being true means that the matmul tasks can overwrite the lhs
    const float beta = lhs_overwritable ? 0.0 : 1.0;

    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               n,
                               m,
                               k,
                               &alpha,
                               rhs2,
                               CUDA_R_32F,
                               rhs2_stride,
                               rhs1,
                               CUDA_R_32F,
                               rhs1_stride,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               lhs_stride));

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatMulImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit MatMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  double* lhs,
                  const double* rhs1,
                  const double* rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const double alpha = 1.0;
    const double beta  = lhs_overwritable ? 0.0 : 1.0;

    CHECK_CUBLAS(cublasDgemm(cublas_handle,
                             rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                             rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             rhs2,
                             rhs2_stride,
                             rhs1,
                             rhs1_stride,
                             &beta,
                             lhs,
                             lhs_stride));

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatMulImplBody<VariantKind::GPU, Type::Code::FLOAT16> {
  TaskContext context;
  explicit MatMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  float* lhs,
                  const legate::Half* rhs1,
                  const legate::Half* rhs2,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.0;
    const float beta  = lhs_overwritable ? 0.0 : 1.0;

    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               n,
                               m,
                               k,
                               &alpha,
                               rhs2,
                               CUDA_R_16F,
                               rhs2_stride,
                               rhs1,
                               CUDA_R_16F,
                               rhs1_stride,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               lhs_stride));

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatMulImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit MatMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  legate::Complex<float>* lhs_,
                  const legate::Complex<float>* rhs1_,
                  const legate::Complex<float>* rhs2_,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    cuComplex* lhs        = reinterpret_cast<cuComplex*>(lhs_);
    const cuComplex* rhs1 = reinterpret_cast<const cuComplex*>(rhs1_);
    const cuComplex* rhs2 = reinterpret_cast<const cuComplex*>(rhs2_);

    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const cuComplex alpha = make_float2(1.0, 0.0);
    const cuComplex beta  = make_float2(lhs_overwritable ? 0.0 : 1.0, 0.0);

    CHECK_CUBLAS(cublasCgemmEx(cublas_handle,
                               rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                               n,
                               m,
                               k,
                               &alpha,
                               rhs2,
                               CUDA_C_32F,
                               rhs2_stride,
                               rhs1,
                               CUDA_C_32F,
                               rhs1_stride,
                               &beta,
                               lhs,
                               CUDA_C_32F,
                               lhs_stride));

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatMulImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit MatMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  size_t k,
                  legate::Complex<double>* lhs_,
                  const legate::Complex<double>* rhs1_,
                  const legate::Complex<double>* rhs2_,
                  size_t lhs_stride,
                  size_t rhs1_stride,
                  size_t rhs2_stride,
                  bool rhs1_transposed,
                  bool rhs2_transposed,
                  bool lhs_overwritable)
  {
    cuDoubleComplex* lhs        = reinterpret_cast<cuDoubleComplex*>(lhs_);
    const cuDoubleComplex* rhs1 = reinterpret_cast<const cuDoubleComplex*>(rhs1_);
    const cuDoubleComplex* rhs2 = reinterpret_cast<const cuDoubleComplex*>(rhs2_);

    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const cuDoubleComplex alpha = make_double2(1.0, 0.0);
    const cuDoubleComplex beta  = make_double2(lhs_overwritable ? 0.0 : 1.0, 0.0);

    CHECK_CUBLAS(cublasZgemm(cublas_handle,
                             rhs2_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                             rhs1_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             rhs2,
                             rhs2_stride,
                             rhs1,
                             rhs1_stride,
                             &beta,
                             lhs,
                             lhs_stride));

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

/*static*/ void MatMulTask::gpu_variant(TaskContext context)
{
  matmul_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
