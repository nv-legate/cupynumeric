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

#include "mathtypes/half.h"

namespace cupynumeric {

size_t stride_for_blas(size_t m, size_t n, size_t x_stride, size_t y_stride, bool& transpose);

inline int64_t unflatten_with_strides(int64_t flat_idx,
                                      size_t ndim,
                                      const int64_t* shape,
                                      const int64_t* strides)
{
  std::vector<int64_t> coords(ndim);
  for (int d = ndim - 1; d >= 0; --d) {
    coords[d] = flat_idx % shape[d];
    flat_idx /= shape[d];
  }
  int64_t idx = 0;
  for (size_t d = 0; d < ndim; ++d) {
    idx += coords[d] * strides[d];
  }
  return idx;
}

int64_t calculate_volume(size_t ndim, const int64_t* shape, int64_t* strides = nullptr);

float* allocate_buffer(size_t size);

// The following assume that the float array was created using allocate_buffer

void half_vector_to_float(float* out, const __half* ptr, size_t n);

void half_matrix_to_float(float* out, const __half* ptr, size_t m, size_t n, size_t pitch);

void half_tensor_to_float(
  float* out, const __half* in, size_t ndim, const int64_t* shape, const int64_t* in_strides);

void float_tensor_to_half(
  __half* out, const float* in, size_t ndim, const int64_t* shape, const int64_t* out_strides);

}  // namespace cupynumeric
