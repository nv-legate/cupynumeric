/* Copyright 2026 NVIDIA Corporation
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

#include "cupynumeric/mapper_gpu_util.h"

#include <cub/device/device_histogram.cuh>
#include <cstddef>
#include <cuda_runtime_api.h>

namespace cupynumeric {

// Returns the temp storage bytes required by cub::DeviceHistogram::HistogramEven
// for a histogram of `num_samples` int samples into `num_bins` bins.
// Pointer types and parameter order exactly match compute_histogram() in
// all2all.cuh so CUB selects the same algorithm and temp-buffer layout.
// Returns 0 on query failure; the caller must treat 0 as "CUB temp budget unknown,
// pool estimate may be short" rather than "CUB needs no temp storage."
std::size_t query_cub_histogram_even_temp_bytes(int num_bins, std::size_t num_samples) noexcept
{
  std::size_t temp_bytes = 0;
  cudaGetLastError();  // clear any pre-existing stale CUDA error
  const cudaError_t err = cub::DeviceHistogram::HistogramEven(
    nullptr,  // d_temp_storage = nullptr → query mode
    temp_bytes,
    static_cast<const int*>(nullptr),           // d_samples (int, matches target_ranks)
    static_cast<unsigned long long*>(nullptr),  // d_histogram (ull, matches send_counts_per_rank)
    num_bins + 1,                               // num_levels
    0,                                          // lower_level
    num_bins,                                   // upper_level
    num_samples,                                // num_samples (size_t)
    static_cast<cudaStream_t>(nullptr));        // stream
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    return 0;
  }
  return temp_bytes;
}

}  // namespace cupynumeric
