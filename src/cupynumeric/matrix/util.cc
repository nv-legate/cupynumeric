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

#include "legate_defines.h"
#include "legate/type/half.h"
#include "legate/data/buffer.h"
#include "cupynumeric/matrix/util.h"
#include "legate/utilities/macros.h"

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
#include <omp.h>
#endif

namespace cupynumeric {

size_t stride_for_blas(size_t m, size_t n, size_t x_stride, size_t y_stride, bool& transpose)
{
  size_t blas_stride;
  if (n == 1) {
    // Column matrix: Every row has exactly 1 element, therefore it is trivially contiguous. Any
    // stride between rows is acceptable.
#ifdef DEBUG_CUPYNUMERIC
    assert(x_stride >= 1);
#endif
    blas_stride = x_stride;
    transpose   = false;
  } else if (m == 1) {
    // Row matrix
#ifdef DEBUG_CUPYNUMERIC
    assert(y_stride >= 1);
#endif
    if (y_stride == 1) {
      // Row elements are contiguous, so there is nothing to do. There is only one row, so the row
      // stride is irrelevant, but we have to pass a value >= n to appease the BLAS library.
      blas_stride = n;
      transpose   = false;
    } else {
      // If the elements in the row are s>1 slots apart, present the input to the BLAS library as a
      // row-major nx1 matrix with row stride equal to s, and ask for the matrix to be transposed.
      blas_stride = y_stride;
      transpose   = true;
    }
  } else {
    // General case: One dimension needs to be contiguous. If that's not the last dimension, then
    // the matrix represents the transpose of a row-major nxm matrix. We then tell the BLAS library
    // that we are passing a row-major nxm matrix, and ask for the matrix to be transposed.
#ifdef DEBUG_CUPYNUMERIC
    assert((x_stride == 1 && y_stride > 1) || (y_stride == 1 && x_stride > 1));
#endif
    blas_stride = std::max(x_stride, y_stride);
    transpose   = x_stride == 1;
  }
  return blas_stride;
}

int64_t calculate_volume(size_t ndim, const int64_t* shape, int64_t* strides)
{
  int64_t volume = 1;
  for (int d = ndim - 1; d >= 0; --d) {
    if (strides != nullptr) {
      strides[d] = volume;
    }
    volume *= shape[d];
  }
  return volume;
}

float* allocate_buffer(size_t size)
{
  auto buffer = legate::create_buffer<float, 1>(size);
  return buffer.ptr(0);
}

void half_vector_to_float(float* out, const legate::Half* ptr, size_t n)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  if (legate::Processor::get_executing_processor().kind() == legate::Processor::OMP_PROC) {
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
      out[idx] = ptr[idx];
    }
    return;
  }
#endif
  for (size_t idx = 0; idx < n; idx++) {
    out[idx] = ptr[idx];
  }
}

void half_matrix_to_float(float* out, const legate::Half* ptr, size_t m, size_t n, size_t pitch)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  if (legate::Processor::get_executing_processor().kind() == legate::Processor::OMP_PROC) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        out[i * n + j] = ptr[i * pitch + j];
      }
    }
    return;
  }
#endif
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      out[i * n + j] = ptr[i * pitch + j];
    }
  }
}

void half_tensor_to_float(
  float* out, const legate::Half* in, size_t ndim, const int64_t* shape, const int64_t* in_strides)
{
  int64_t volume = calculate_volume(ndim, shape);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  if (legate::Processor::get_executing_processor().kind() == legate::Processor::OMP_PROC) {
#pragma omp parallel for schedule(static)
    for (int64_t out_idx = 0; out_idx < volume; ++out_idx) {
      int64_t in_idx = unflatten_with_strides(out_idx, ndim, shape, in_strides);
      out[out_idx]   = in[in_idx];
    }
    return;
  }
#endif
  for (int64_t out_idx = 0; out_idx < volume; ++out_idx) {
    int64_t in_idx = unflatten_with_strides(out_idx, ndim, shape, in_strides);
    out[out_idx]   = in[in_idx];
  }
}

void float_tensor_to_half(
  legate::Half* out, const float* in, size_t ndim, const int64_t* shape, const int64_t* out_strides)
{
  int64_t volume = calculate_volume(ndim, shape);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  if (legate::Processor::get_executing_processor().kind() == legate::Processor::OMP_PROC) {
#pragma omp parallel for schedule(static)
    for (int64_t in_idx = 0; in_idx < volume; ++in_idx) {
      int64_t out_idx = unflatten_with_strides(in_idx, ndim, shape, out_strides);
      out[out_idx]    = in[in_idx];
    }
    return;
  }
#endif
  for (int64_t in_idx = 0; in_idx < volume; ++in_idx) {
    int64_t out_idx = unflatten_with_strides(in_idx, ndim, shape, out_strides);
    out[out_idx]    = in[in_idx];
  }
}

}  // namespace cupynumeric
