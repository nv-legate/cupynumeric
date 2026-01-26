/* Copyright 2025 NVIDIA Corporation
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

#include "cupynumeric/matrix/potrs.h"
#include "cupynumeric/matrix/potrs_template.inl"

#include "cupynumeric/utilities/blas_lapack.h"

namespace cupynumeric {

using namespace legate;

template <typename Potrs, typename VAL>
static inline void potrs_template(Potrs potrs,
                                  const VAL* a,
                                  VAL* x,
                                  int32_t m,
                                  int32_t n,
                                  int32_t num_blocks,
                                  int64_t a_block_stride,
                                  int64_t x_block_stride,
                                  bool lower)
{
  char uplo    = lower ? 'L' : 'U';
  int32_t info = 0;

  for (int32_t i = 0; i < num_blocks; ++i) {
    potrs(&uplo,
          &m,
          &n,
          const_cast<VAL*>(a) + i * a_block_stride,
          &m,
          x + i * x_block_stride,
          &m,
          &info);
  }
  if (info != 0) {
    throw legate::TaskException("Singular matrix");
  }
}

template <>
void PotrsImplBody<VariantKind::CPU, Type::Code::FLOAT32>::operator()(const float* a,
                                                                      float* x,
                                                                      int32_t m,
                                                                      int32_t n,
                                                                      int32_t num_blocks,
                                                                      int64_t a_block_stride,
                                                                      int64_t x_block_stride,
                                                                      bool lower)
{
  potrs_template(spotrs_, a, x, m, n, num_blocks, a_block_stride, x_block_stride, lower);
}

template <>
void PotrsImplBody<VariantKind::CPU, Type::Code::FLOAT64>::operator()(const double* a,
                                                                      double* x,
                                                                      int32_t m,
                                                                      int32_t n,
                                                                      int32_t num_blocks,
                                                                      int64_t a_block_stride,
                                                                      int64_t x_block_stride,
                                                                      bool lower)
{
  potrs_template(dpotrs_, a, x, m, n, num_blocks, a_block_stride, x_block_stride, lower);
}

template <>
void PotrsImplBody<VariantKind::CPU, Type::Code::COMPLEX64>::operator()(
  const legate::Complex<float>* a_,
  legate::Complex<float>* x_,
  int32_t m,
  int32_t n,
  int32_t num_blocks,
  int64_t a_block_stride,
  int64_t x_block_stride,
  bool lower)
{
  auto a = reinterpret_cast<const __complex__ float*>(a_);
  auto x = reinterpret_cast<__complex__ float*>(x_);

  potrs_template(cpotrs_, a, x, m, n, num_blocks, a_block_stride, x_block_stride, lower);
}

template <>
void PotrsImplBody<VariantKind::CPU, Type::Code::COMPLEX128>::operator()(
  const legate::Complex<double>* a_,
  legate::Complex<double>* x_,
  int32_t m,
  int32_t n,
  int32_t num_blocks,
  int64_t a_block_stride,
  int64_t x_block_stride,
  bool lower)
{
  auto a = reinterpret_cast<const __complex__ double*>(a_);
  auto x = reinterpret_cast<__complex__ double*>(x_);

  potrs_template(zpotrs_, a, x, m, n, num_blocks, a_block_stride, x_block_stride, lower);
}

/*static*/ void PotrsTask::cpu_variant(TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  blas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  potrs_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  PotrsTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
