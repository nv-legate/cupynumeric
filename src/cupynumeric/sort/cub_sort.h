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

namespace cupynumeric {

void cub_local_sort(const bool* values_in,
                    bool* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const int8_t* values_in,
                    int8_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const int16_t* values_in,
                    int16_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const int32_t* values_in,
                    int32_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const int64_t* values_in,
                    int64_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const uint8_t* values_in,
                    uint8_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const uint16_t* values_in,
                    uint16_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const uint32_t* values_in,
                    uint32_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const uint64_t* values_in,
                    uint64_t* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const __half* values_in,
                    __half* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const float* values_in,
                    float* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

void cub_local_sort(const double* values_in,
                    double* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

}  // namespace cupynumeric
