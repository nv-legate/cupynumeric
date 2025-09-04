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

#include "cupynumeric/utilities/blas_lapack.h"

namespace cupynumeric {

using namespace legate;

template <typename Trsm, typename VAL>
static inline void trsm_template(Trsm trsm, VAL* lhs, const VAL* rhs, int32_t m, int32_t n)
{
  auto side   = CblasRight;
  auto uplo   = CblasLower;
  auto transa = CblasTrans;
  auto diag   = CblasNonUnit;

  trsm(CblasColMajor, side, uplo, transa, diag, m, n, 1.0, rhs, n, lhs, m);
}

template <typename Trsm, typename VAL>
static inline void complex_trsm_template(Trsm trsm, VAL* lhs, const VAL* rhs, int32_t m, int32_t n)
{
  auto side   = CblasRight;
  auto uplo   = CblasLower;
  auto transa = CblasConjTrans;
  auto diag   = CblasNonUnit;

  VAL alpha = 1.0;

  trsm(CblasColMajor, side, uplo, transa, diag, m, n, &alpha, rhs, n, lhs, m);
}

template <>
struct TrsmImplBody<VariantKind::CPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(float* lhs, const float* rhs, int32_t m, int32_t n)
  {
    trsm_template(cblas_strsm, lhs, rhs, m, n);
  }
};

template <>
struct TrsmImplBody<VariantKind::CPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(double* lhs, const double* rhs, int32_t m, int32_t n)
  {
    trsm_template(cblas_dtrsm, lhs, rhs, m, n);
  }
};

template <>
struct TrsmImplBody<VariantKind::CPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(complex<float>* lhs_, const complex<float>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<__complex__ float*>(lhs_);
    auto rhs = reinterpret_cast<const __complex__ float*>(rhs_);

    complex_trsm_template(cblas_ctrsm, lhs, rhs, m, n);
  }
};

template <>
struct TrsmImplBody<VariantKind::CPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  void operator()(complex<double>* lhs_, const complex<double>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<__complex__ double*>(lhs_);
    auto rhs = reinterpret_cast<const __complex__ double*>(rhs_);

    complex_trsm_template(cblas_ztrsm, lhs, rhs, m, n);
  }
};

/*static*/ void TrsmTask::cpu_variant(TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  blas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  trsm_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  TrsmTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
