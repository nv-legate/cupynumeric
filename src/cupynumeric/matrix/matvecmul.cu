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

#include "cupynumeric/matrix/matvecmul.h"
#include "cupynumeric/matrix/matvecmul_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <>
struct MatVecMulImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit MatVecMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const float* mat,
                  const float* vec,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_overwritable)
  {
    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.0;
    // lhs_overwritable being true means that the matvecmul tasks can overwrite the lhs
    const float beta = lhs_overwritable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;

    // XXX: There is a bug in older versions of cuBLAS that are triggered
    //      by some degenerate matrix-vector multiplications. We simply use
    //      matrix-matrix multiplication all the time unless we're on a recent
    //      cuBLAS version
    int32_t version;
    CHECK_CUBLAS(cublasGetVersion(cublas_handle, &version));
    if (version >= 11700) {
      CHECK_CUBLAS(
        cublasSgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
    } else {
      CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                                 trans,
                                 CUBLAS_OP_N,
                                 transpose_mat ? n : m,
                                 1,
                                 transpose_mat ? m : n,
                                 &alpha,
                                 mat,
                                 CUDA_R_32F,
                                 mat_stride,
                                 vec,
                                 CUDA_R_32F,
                                 transpose_mat ? m : n,
                                 &beta,
                                 lhs,
                                 CUDA_R_32F,
                                 transpose_mat ? n : m));
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit MatVecMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  double* lhs,
                  const double* mat,
                  const double* vec,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_overwritable)
  {
    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const double alpha = 1.0;
    const double beta  = lhs_overwritable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;

    // FIXME: It's actually unknown that the cuBLAS bug for 32-bit floats reproduces for
    //        64-bit flots as well. We're simply being conservative here.
    int32_t version;
    CHECK_CUBLAS(cublasGetVersion(cublas_handle, &version));
    if (version >= 11700) {
      CHECK_CUBLAS(
        cublasDgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
    } else {
      CHECK_CUBLAS(cublasDgemm(cublas_handle,
                               trans,
                               CUBLAS_OP_N,
                               transpose_mat ? n : m,
                               1,
                               transpose_mat ? m : n,
                               &alpha,
                               mat,
                               mat_stride,
                               vec,
                               transpose_mat ? m : n,
                               &beta,
                               lhs,
                               transpose_mat ? n : m));
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, Type::Code::FLOAT16> {
  TaskContext context;
  explicit MatVecMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  float* lhs,
                  const legate::Half* mat,
                  const legate::Half* vec,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_overwritable)
  {
    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const float alpha = 1.0;
    const float beta  = lhs_overwritable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;
    // Use SgemmEx here since there is no half precision gemv yet
    CHECK_CUBLAS(cublasSgemmEx(cublas_handle,
                               trans,
                               CUBLAS_OP_N,
                               transpose_mat ? n : m,
                               1,
                               transpose_mat ? m : n,
                               &alpha,
                               mat,
                               CUDA_R_16F,
                               mat_stride,
                               vec,
                               CUDA_R_16F,
                               transpose_mat ? m : n,
                               &beta,
                               lhs,
                               CUDA_R_32F,
                               transpose_mat ? n : m));

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit MatVecMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  legate::Complex<float>* lhs_,
                  const legate::Complex<float>* mat_,
                  const legate::Complex<float>* vec_,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_overwritable)
  {
    cuComplex* lhs       = reinterpret_cast<cuComplex*>(lhs_);
    const cuComplex* mat = reinterpret_cast<const cuComplex*>(mat_);
    const cuComplex* vec = reinterpret_cast<const cuComplex*>(vec_);

    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const cuComplex alpha = make_float2(1.0, 0.0);
    const cuComplex beta  = make_float2(lhs_overwritable ? 0.0 : 1.0, 0.0);

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;

    // FIXME: It's actually unknown that the cuBLAS bug for 32-bit floats reproduces for
    //        complex64 as well. We're simply being conservative here.
    int32_t version;
    CHECK_CUBLAS(cublasGetVersion(cublas_handle, &version));
    if (version >= 11700) {
      CHECK_CUBLAS(
        cublasCgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
    } else {
      CHECK_CUBLAS(cublasCgemmEx(cublas_handle,
                                 trans,
                                 CUBLAS_OP_N,
                                 transpose_mat ? n : m,
                                 1,
                                 transpose_mat ? m : n,
                                 &alpha,
                                 mat,
                                 CUDA_C_32F,
                                 mat_stride,
                                 vec,
                                 CUDA_C_32F,
                                 transpose_mat ? m : n,
                                 &beta,
                                 lhs,
                                 CUDA_C_32F,
                                 transpose_mat ? n : m));
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

template <>
struct MatVecMulImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit MatVecMulImplBody(TaskContext context) : context(context) {}

  void operator()(size_t m,
                  size_t n,
                  legate::Complex<double>* lhs_,
                  const legate::Complex<double>* mat_,
                  const legate::Complex<double>* vec_,
                  size_t mat_stride,
                  bool transpose_mat,
                  bool lhs_overwritable)
  {
    cuDoubleComplex* lhs       = reinterpret_cast<cuDoubleComplex*>(lhs_);
    const cuDoubleComplex* mat = reinterpret_cast<const cuDoubleComplex*>(mat_);
    const cuDoubleComplex* vec = reinterpret_cast<const cuDoubleComplex*>(vec_);

    auto cublas_handle = get_cublas();
    auto task_stream   = context.get_task_stream();
    CHECK_CUBLAS(cublasSetStream(cublas_handle, task_stream));

    const cuDoubleComplex alpha = make_double2(1.0, 0.0);
    const cuDoubleComplex beta  = make_double2(lhs_overwritable ? 0.0 : 1.0, 0.0);

    auto trans = transpose_mat ? CUBLAS_OP_N : CUBLAS_OP_T;

    // FIXME: It's actually unknown that the cuBLAS bug for 32-bit floats reproduces for
    //        complex128 as well. We're simply being conservative here.
    int32_t version;
    CHECK_CUBLAS(cublasGetVersion(cublas_handle, &version));
    if (version >= 11700) {
      CHECK_CUBLAS(
        cublasZgemv(cublas_handle, trans, n, m, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1));
    } else {
      CHECK_CUBLAS(cublasZgemm(cublas_handle,
                               trans,
                               CUBLAS_OP_N,
                               transpose_mat ? n : m,
                               1,
                               transpose_mat ? m : n,
                               &alpha,
                               mat,
                               mat_stride,
                               vec,
                               transpose_mat ? m : n,
                               &beta,
                               lhs,
                               transpose_mat ? n : m));
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);
  }
};

/*static*/ void MatVecMulTask::gpu_variant(TaskContext context)
{
  matvecmul_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
