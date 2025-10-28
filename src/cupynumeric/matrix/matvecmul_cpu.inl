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

#pragma once

#include "legate/type/half.h"
#include "cupynumeric/matrix/matvecmul.h"
#include "cupynumeric/matrix/matvecmul_template.inl"
#include "cupynumeric/matrix/util.h"
#include "cupynumeric/utilities/blas_lapack.h"

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, Type::Code::FLOAT32> {
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
    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    // lhs_overwritable being true means that the matvecmul tasks can overwrite the lhs
    float beta = lhs_overwritable ? 0.0 : 1.0;
    cblas_sgemv(CblasRowMajor, trans, m, n, 1, mat, mat_stride, vec, 1, beta, lhs, 1);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, Type::Code::FLOAT64> {
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
    auto trans  = transpose_mat ? CblasTrans : CblasNoTrans;
    double beta = lhs_overwritable ? 0.0 : 1.0;
    cblas_dgemv(CblasRowMajor, trans, m, n, 1, mat, mat_stride, vec, 1, beta, lhs, 1);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, Type::Code::FLOAT16> {
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
    auto vec_size = transpose_mat ? m : n;

    auto mat_copy = allocate_buffer(m * n);
    auto vec_copy = allocate_buffer(vec_size);

    half_matrix_to_float(mat_copy, mat, m, n, mat_stride);
    half_vector_to_float(vec_copy, vec, vec_size);

    MatVecMulImplBody<KIND, Type::Code::FLOAT32>{context}(
      m, n, lhs, mat_copy, vec_copy, n, transpose_mat, lhs_overwritable);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, Type::Code::COMPLEX64> {
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
    __complex__ float* lhs       = reinterpret_cast<__complex__ float*>(lhs_);
    const __complex__ float* mat = reinterpret_cast<const __complex__ float*>(mat_);
    const __complex__ float* vec = reinterpret_cast<const __complex__ float*>(vec_);
    __complex__ float alpha      = 1.0;
    __complex__ float beta       = lhs_overwritable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_cgemv(CblasRowMajor, trans, m, n, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1);
  }
};

template <VariantKind KIND>
struct MatVecMulImplBody<KIND, Type::Code::COMPLEX128> {
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
    __complex__ double* lhs       = reinterpret_cast<__complex__ double*>(lhs_);
    const __complex__ double* mat = reinterpret_cast<const __complex__ double*>(mat_);
    const __complex__ double* vec = reinterpret_cast<const __complex__ double*>(vec_);
    __complex__ double alpha      = 1.0;
    __complex__ double beta       = lhs_overwritable ? 0.0 : 1.0;

    auto trans = transpose_mat ? CblasTrans : CblasNoTrans;
    cblas_zgemv(CblasRowMajor, trans, m, n, &alpha, mat, mat_stride, vec, 1, &beta, lhs, 1);
  }
};

}  // namespace cupynumeric
