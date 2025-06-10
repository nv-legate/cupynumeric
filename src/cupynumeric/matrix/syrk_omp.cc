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

#include "cupynumeric/matrix/syrk.h"
#include "cupynumeric/matrix/syrk_template.inl"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <typename Syrk, typename VAL>
static inline void syrk_template(Syrk syrk, VAL* lhs, const VAL* rhs, int32_t m, int32_t n)
{
  auto uplo  = CblasLower;
  auto trans = CblasNoTrans;

  syrk(CblasColMajor, uplo, trans, m, n, -1.0, rhs, m, 1.0, lhs, m);
}

template <>
struct SyrkImplBody<VariantKind::CPU, Type::Code::FLOAT32> {
  void operator()(float* lhs, const float* rhs, int32_t m, int32_t n)
  {
    syrk_template(cblas_ssyrk, lhs, rhs, m, n);
  }
};

template <>
struct SyrkImplBody<VariantKind::CPU, Type::Code::FLOAT64> {
  void operator()(double* lhs, const double* rhs, int32_t m, int32_t n)
  {
    syrk_template(cblas_dsyrk, lhs, rhs, m, n);
  }
};

template <>
struct SyrkImplBody<VariantKind::CPU, Type::Code::COMPLEX64> {
  void operator()(complex<float>* lhs_, const complex<float>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<__complex__ float*>(lhs_);
    auto rhs = reinterpret_cast<const __complex__ float*>(rhs_);

    // TODO: We're not calling syrk but actually calling herk instead here,
    //       as this task is used only for Cholesky factorization right now.
    //       (the complex64 version of syrk is csyrk)
    //       Will need to fix this once we start porting scipy.linalg
    syrk_template(cblas_cherk, lhs, rhs, m, n);
  }
};

template <>
struct SyrkImplBody<VariantKind::CPU, Type::Code::COMPLEX128> {
  void operator()(complex<double>* lhs_, const complex<double>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<__complex__ double*>(lhs_);
    auto rhs = reinterpret_cast<const __complex__ double*>(rhs_);

    // TODO: the same problem here as in the complex64 case
    syrk_template(cblas_zherk, lhs, rhs, m, n);
  }
};

/*static*/ void SyrkTask::omp_variant(TaskContext context)
{
  blas_set_num_threads(omp_get_max_threads());
  syrk_template<VariantKind::CPU>(context);
}

}  // namespace cupynumeric
