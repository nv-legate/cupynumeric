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

#pragma once

#include "legate.h"

#include <cstddef>

#if LEGATE_DEFINED(LEGATE_USE_CUDA)

namespace cupynumeric {

// Returns the temp storage bytes required by cub::DeviceHistogram::HistogramEven
// for a histogram of `num_samples` int samples into `num_bins` bins.
// Pointer types and parameter order exactly match compute_histogram() in
// all2all.cuh so CUB selects the same algorithm and temp-buffer layout.
// Returns 0 on query failure; the caller must treat 0 as "CUB temp budget unknown,
// pool estimate may be short" rather than "CUB needs no temp storage."
std::size_t query_cub_histogram_even_temp_bytes(int num_bins, std::size_t num_samples) noexcept;

}  // namespace cupynumeric

#endif  // LEGATE_DEFINED(LEGATE_USE_CUDA)
