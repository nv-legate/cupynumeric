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

#include "cupynumeric/matrix/gemm.h"
#include "cupynumeric/matrix/gemm_template.inl"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <typename Gemm, typename VAL>
static inline void gemm_template(
  Gemm gemm, VAL* lhs, const VAL* rhs1, const VAL* rhs2, int32_t m, int32_t n, int32_t k)
{
  auto transa = CblasNoTrans;
  auto transb = CblasTrans;

  gemm(CblasColMajor, transa, transb, m, n, k, -1.0, rhs1, m, rhs2, n, 1.0, lhs, m);
}

template <typename Gemm, typename VAL>
static inline void complex_gemm_template(
  Gemm gemm, VAL* lhs, const VAL* rhs1, const VAL* rhs2, int32_t m, int32_t n, int32_t k)
{
  auto transa = CblasNoTrans;
  auto transb = CblasConjTrans;

  VAL alpha = -1.0;
  VAL beta  = 1.0;

  gemm(CblasColMajor, transa, transb, m, n, k, &alpha, rhs1, m, rhs2, n, &beta, lhs, m);
}

template <>
struct GemmImplBody<VariantKind::CPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit GemmImplBody(TaskContext context) : context(context) {}

  void operator()(float* lhs, const float* rhs1, const float* rhs2, int32_t m, int32_t n, int32_t k)
  {
    gemm_template(cblas_sgemm, lhs, rhs1, rhs2, m, n, k);
  }
};

template <>
struct GemmImplBody<VariantKind::CPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit GemmImplBody(TaskContext context) : context(context) {}

  void operator()(
    double* lhs, const double* rhs1, const double* rhs2, int32_t m, int32_t n, int32_t k)
  {
    gemm_template(cblas_dgemm, lhs, rhs1, rhs2, m, n, k);
  }
};

template <>
struct GemmImplBody<VariantKind::CPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit GemmImplBody(TaskContext context) : context(context) {}

  void operator()(legate::Complex<float>* lhs_,
                  const legate::Complex<float>* rhs1_,
                  const legate::Complex<float>* rhs2_,
                  int32_t m,
                  int32_t n,
                  int32_t k)
  {
    auto lhs  = reinterpret_cast<__complex__ float*>(lhs_);
    auto rhs1 = reinterpret_cast<const __complex__ float*>(rhs1_);
    auto rhs2 = reinterpret_cast<const __complex__ float*>(rhs2_);

    complex_gemm_template(cblas_cgemm, lhs, rhs1, rhs2, m, n, k);
  }
};

template <>
struct GemmImplBody<VariantKind::CPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit GemmImplBody(TaskContext context) : context(context) {}

  void operator()(legate::Complex<double>* lhs_,
                  const legate::Complex<double>* rhs1_,
                  const legate::Complex<double>* rhs2_,
                  int32_t m,
                  int32_t n,
                  int32_t k)
  {
    auto lhs  = reinterpret_cast<__complex__ double*>(lhs_);
    auto rhs1 = reinterpret_cast<const __complex__ double*>(rhs1_);
    auto rhs2 = reinterpret_cast<const __complex__ double*>(rhs2_);

    complex_gemm_template(cblas_zgemm, lhs, rhs1, rhs2, m, n, k);
  }
};

/*static*/ void GemmTask::omp_variant(TaskContext context)
{
  blas_set_num_threads(omp_get_max_threads());
  gemm_template<VariantKind::CPU>(context);
}

}  // namespace cupynumeric
