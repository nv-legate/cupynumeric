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

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename Syrk, typename VAL, typename CONS>
static inline void syrk_template(
  Syrk syrk, VAL* lhs, const VAL* rhs, int32_t m, int32_t n, CONS _fake_param_for_type_inference)
{
  auto context = get_cublas();
  auto stream  = get_cached_stream();
  CHECK_CUBLAS(cublasSetStream(context, stream));

  auto uplo  = CUBLAS_FILL_MODE_LOWER;
  auto trans = CUBLAS_OP_N;
  CONS alpha = -1.0;
  CONS beta  = 1.0;

  CHECK_CUBLAS(syrk(context, uplo, trans, m, n, &alpha, rhs, m, &beta, lhs, m));

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <>
struct SyrkImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  void operator()(float* lhs, const float* rhs, int32_t m, int32_t n)
  {
    syrk_template(cublasSsyrk, lhs, rhs, m, n, static_cast<float>(0));
  }
};

template <>
struct SyrkImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  void operator()(double* lhs, const double* rhs, int32_t m, int32_t n)
  {
    syrk_template(cublasDsyrk, lhs, rhs, m, n, static_cast<double>(0));
  }
};

template <>
struct SyrkImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  void operator()(complex<float>* lhs_, const complex<float>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<cuComplex*>(lhs_);
    auto rhs = reinterpret_cast<const cuComplex*>(rhs_);

    syrk_template(cublasCherk, lhs, rhs, m, n, static_cast<float>(0));
  }
};

template <>
struct SyrkImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  void operator()(complex<double>* lhs_, const complex<double>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<cuDoubleComplex*>(lhs_);
    auto rhs = reinterpret_cast<const cuDoubleComplex*>(rhs_);

    // TODO: We're not actually calling syrk but calling hekr instead here,
    //       as this task is used only for Cholesky factorization.
    syrk_template(cublasZherk, lhs, rhs, m, n, static_cast<double>(0));
  }
};

/*static*/ void SyrkTask::gpu_variant(TaskContext context)
{
  syrk_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
