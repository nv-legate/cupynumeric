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
static inline void trsm_template(Trsm trsm,
                                 const VAL* a,
                                 const VAL* b,
                                 VAL* x,
                                 int32_t m,
                                 int32_t n,
                                 int32_t num_blocks,
                                 int64_t a_block_stride,
                                 int64_t b_block_stride,
                                 bool side_left,
                                 bool lower,
                                 int32_t transa_op,
                                 bool unit_diag)
{
  auto side   = side_left ? CblasLeft : CblasRight;
  auto uplo   = lower ? CblasLower : CblasUpper;
  auto transa = (transa_op == 0) ? CblasNoTrans
                                 : CblasTrans;  // For real types, Trans and ConjTrans are the same
  auto diag   = unit_diag ? CblasUnit : CblasNonUnit;

  // Copy b to x if they're different
  if (b != x) {
    std::memcpy(x, b, sizeof(VAL) * m * n * num_blocks);
  }

  auto lda = side_left ? m : n;

  if (num_blocks == 1) {
    // Single solve
    trsm(CblasColMajor, side, uplo, transa, diag, m, n, 1.0, a, lda, x, m);
  } else {
    // Batched solve - loop over batches
    for (int32_t i = 0; i < num_blocks; ++i) {
      trsm(CblasColMajor,
           side,
           uplo,
           transa,
           diag,
           m,
           n,
           1.0,
           a + i * a_block_stride,
           lda,
           x + i * b_block_stride,
           m);
    }
  }
}

template <typename Trsm, typename VAL>
static inline void complex_trsm_template(Trsm trsm,
                                         const VAL* a,
                                         const VAL* b,
                                         VAL* x,
                                         int32_t m,
                                         int32_t n,
                                         int32_t num_blocks,
                                         int64_t a_block_stride,
                                         int64_t b_block_stride,
                                         bool side_left,
                                         bool lower,
                                         int32_t transa_op,
                                         bool unit_diag)
{
  auto side   = side_left ? CblasLeft : CblasRight;
  auto uplo   = lower ? CblasLower : CblasUpper;
  auto transa = (transa_op == 0) ? CblasNoTrans : (transa_op == 1) ? CblasTrans : CblasConjTrans;
  auto diag   = unit_diag ? CblasUnit : CblasNonUnit;

  VAL alpha = 1.0;

  // Copy b to x if they're different
  if (b != x) {
    std::memcpy(x, b, sizeof(VAL) * m * n * num_blocks);
  }

  auto lda = side_left ? m : n;

  if (num_blocks == 1) {
    // Single solve
    trsm(CblasColMajor, side, uplo, transa, diag, m, n, &alpha, a, lda, x, m);
  } else {
    // Batched solve - loop over batches
    for (int32_t i = 0; i < num_blocks; ++i) {
      trsm(CblasColMajor,
           side,
           uplo,
           transa,
           diag,
           m,
           n,
           &alpha,
           a + i * a_block_stride,
           lda,
           x + i * b_block_stride,
           m);
    }
  }
}

template <>
void TrsmImplBody<VariantKind::CPU, Type::Code::FLOAT32>::operator()(const float* a,
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
  trsm_template(cblas_strsm,
                a,
                b,
                x,
                m,
                n,
                num_blocks,
                a_block_stride,
                b_block_stride,
                side,
                lower,
                transa,
                unit_diagonal);
}

template <>
void TrsmImplBody<VariantKind::CPU, Type::Code::FLOAT64>::operator()(const double* a,
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
  trsm_template(cblas_dtrsm,
                a,
                b,
                x,
                m,
                n,
                num_blocks,
                a_block_stride,
                b_block_stride,
                side,
                lower,
                transa,
                unit_diagonal);
}

template <>
void TrsmImplBody<VariantKind::CPU, Type::Code::COMPLEX64>::operator()(
  const legate::Complex<float>* a_,
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
  auto a = reinterpret_cast<const __complex__ float*>(a_);
  auto b = reinterpret_cast<const __complex__ float*>(b_);
  auto x = reinterpret_cast<__complex__ float*>(x_);

  complex_trsm_template(cblas_ctrsm,
                        a,
                        b,
                        x,
                        m,
                        n,
                        num_blocks,
                        a_block_stride,
                        b_block_stride,
                        side,
                        lower,
                        transa,
                        unit_diagonal);
}

template <>
void TrsmImplBody<VariantKind::CPU, Type::Code::COMPLEX128>::operator()(
  const legate::Complex<double>* a_,
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
  auto a = reinterpret_cast<const __complex__ double*>(a_);
  auto b = reinterpret_cast<const __complex__ double*>(b_);
  auto x = reinterpret_cast<__complex__ double*>(x_);

  complex_trsm_template(cblas_ztrsm,
                        a,
                        b,
                        x,
                        m,
                        n,
                        num_blocks,
                        a_block_stride,
                        b_block_stride,
                        side,
                        lower,
                        transa,
                        unit_diagonal);
}

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
