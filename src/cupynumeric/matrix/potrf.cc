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

#include "cupynumeric/matrix/potrf.h"
#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/matrix/potrf_template.inl"

#include <legate/type/types.h>
#include "cupynumeric/utilities/blas_lapack.h"

namespace cupynumeric {

using namespace legate;

template <>
void CopyBlockImpl<VariantKind::CPU>::operator()(void* dst, const void* src, size_t size)
{
  ::memcpy(dst, src, size);
}

// PotrfImplBody for CPU - FLOAT32
template <>
void PotrfImplBody<VariantKind::CPU, Type::Code::FLOAT32>::operator()(
  float* array, int32_t n, int32_t lda, bool lower, int32_t num_blocks, int64_t block_stride)
{
  char uplo = lower ? 'L' : 'U';

  for (int32_t i = 0; i < num_blocks; ++i) {
    float* block_ptr = array + i * block_stride;
    int32_t info     = 0;
    spotrf_(&uplo, &n, block_ptr, &lda, &info);
    if (info != 0) {
      throw legate::TaskException("Matrix is not positive definite");
    }
  }
}

// PotrfImplBody for CPU - FLOAT64
template <>
void PotrfImplBody<VariantKind::CPU, Type::Code::FLOAT64>::operator()(
  double* array, int32_t n, int32_t lda, bool lower, int32_t num_blocks, int64_t block_stride)
{
  char uplo = lower ? 'L' : 'U';

  for (int32_t i = 0; i < num_blocks; ++i) {
    double* block_ptr = array + i * block_stride;
    int32_t info      = 0;
    dpotrf_(&uplo, &n, block_ptr, &lda, &info);
    if (info != 0) {
      throw legate::TaskException("Matrix is not positive definite");
    }
  }
}

// PotrfImplBody for CPU - COMPLEX64
template <>
void PotrfImplBody<VariantKind::CPU, Type::Code::COMPLEX64>::operator()(
  legate::Complex<float>* array,
  int32_t n,
  int32_t lda,
  bool lower,
  int32_t num_blocks,
  int64_t block_stride)
{
  char uplo = lower ? 'L' : 'U';

  for (int32_t i = 0; i < num_blocks; ++i) {
    auto* block_ptr = reinterpret_cast<__complex__ float*>(array + i * block_stride);
    int32_t info    = 0;
    cpotrf_(&uplo, &n, block_ptr, &lda, &info);
    if (info != 0) {
      throw legate::TaskException("Matrix is not positive definite");
    }
  }
}

// PotrfImplBody for CPU - COMPLEX128
template <>
void PotrfImplBody<VariantKind::CPU, Type::Code::COMPLEX128>::operator()(
  legate::Complex<double>* array,
  int32_t n,
  int32_t lda,
  bool lower,
  int32_t num_blocks,
  int64_t block_stride)
{
  char uplo = lower ? 'L' : 'U';

  for (int32_t i = 0; i < num_blocks; ++i) {
    auto* block_ptr = reinterpret_cast<__complex__ double*>(array + i * block_stride);
    int32_t info    = 0;
    zpotrf_(&uplo, &n, block_ptr, &lda, &info);
    if (info != 0) {
      throw legate::TaskException("Matrix is not positive definite");
    }
  }
}

// BatchedTriluImplBody for CPU - zeros out upper or lower triangle
template <Type::Code CODE>
struct BatchedTriluImplBody<VariantKind::CPU, CODE> {
  TaskContext context;
  explicit BatchedTriluImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(VAL* array, int32_t n, bool lower, int32_t num_blocks, int64_t block_stride) const
  {
    for (int32_t batch = 0; batch < num_blocks; ++batch) {
      VAL* block_ptr = array + batch * block_stride;

      // Column-major layout: element at (row, col) is at block_ptr[col * n + row]
      for (int32_t col = 0; col < n; ++col) {
        for (int32_t row = 0; row < n; ++row) {
          if (lower) {
            // Zero out upper triangle (elements where row < col)
            if (row < col) {
              block_ptr[col * n + row] = VAL(0);
            }
          } else {
            // Zero out lower triangle (elements where row > col)
            if (row > col) {
              block_ptr[col * n + row] = VAL(0);
            }
          }
        }
      }
    }
  }
};

/*static*/ void PotrfTask::cpu_variant(TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  blas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  potrf_task_context_dispatch<VariantKind::CPU>(context);
}

namespace  // unnamed
{
const auto cupynumeric_reg_task_ = []() -> char {
  PotrfTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
