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

namespace cupynumeric {

using namespace legate;

template <typename Trsm, typename VAL>
static inline void trsm_template(
  Trsm trsm, VAL* lhs, const VAL* rhs, int32_t m, int32_t n, VAL alpha)
{
  auto context = get_cublas();
  auto stream  = get_cached_stream();
  CHECK_CUBLAS(cublasSetStream(context, stream));

  // TODO: We need to expose these parameters to the API later we port scipy.linalg
  auto side   = CUBLAS_SIDE_RIGHT;
  auto uplo   = CUBLAS_FILL_MODE_LOWER;
  auto transa = CUBLAS_OP_C;
  auto diag   = CUBLAS_DIAG_NON_UNIT;

  CHECK_CUBLAS(trsm(context, side, uplo, transa, diag, m, n, &alpha, rhs, n, lhs, m));

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  void operator()(float* lhs, const float* rhs, int32_t m, int32_t n)
  {
    trsm_template(cublasStrsm, lhs, rhs, m, n, 1.0F);
  }
};

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  void operator()(double* lhs, const double* rhs, int32_t m, int32_t n)
  {
    trsm_template(cublasDtrsm, lhs, rhs, m, n, 1.0);
  }
};

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  void operator()(complex<float>* lhs_, const complex<float>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<cuComplex*>(lhs_);
    auto rhs = reinterpret_cast<const cuComplex*>(rhs_);

    trsm_template(cublasCtrsm, lhs, rhs, m, n, make_float2(1.0, 0.0));
  }
};

template <>
struct TrsmImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  void operator()(complex<double>* lhs_, const complex<double>* rhs_, int32_t m, int32_t n)
  {
    auto lhs = reinterpret_cast<cuDoubleComplex*>(lhs_);
    auto rhs = reinterpret_cast<const cuDoubleComplex*>(rhs_);

    trsm_template(cublasZtrsm, lhs, rhs, m, n, make_double2(1.0, 0.0));
  }
};

/*static*/ void TrsmTask::gpu_variant(TaskContext context)
{
  trsm_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
