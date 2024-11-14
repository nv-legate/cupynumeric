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

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#if LEGATE_DEFINED(LEGATE_USE_CUDA) and LEGATE_DEFINED(LEGATE_NVCC)
#include "cupynumeric/cuda_help.h"
#else
#define cudaStream_t std::int32_t
#endif
#include "cupynumeric/stat/histogram_gen.h"

namespace cupynumeric {
namespace detail {

// host specialization:
//
template <typename exe_policy_t, typename weight_t, typename offset_t>
struct segmented_sum_t<exe_policy_t,
                       weight_t,
                       offset_t,
                       std::enable_if_t<is_host_policy_v<exe_policy_t>>> {
  segmented_sum_t(exe_policy_t exe_pol,
                  weight_t const* p_weights,
                  size_t n_samples,
                  weight_t* p_hist,
                  size_t n_intervals,
                  offset_t* p_offsets,
                  cudaStream_t stream)
    : ptr_weights_(p_weights),
      n_samples_(n_samples),
      ptr_hist_(p_hist),
      n_intervals_(n_intervals),
      ptr_offsets_(p_offsets)
  {
  }
  void operator()(void)
  {
    for (size_t interval_index = 0; interval_index < n_intervals_; ++interval_index) {
      auto next_index = interval_index + 1;

      ptr_hist_[interval_index] = thrust::reduce(thrust::seq,
                                                 ptr_weights_ + ptr_offsets_[interval_index],
                                                 ptr_weights_ + ptr_offsets_[next_index],
                                                 0.0);
    }
  }

  weight_t* get_histogram(void) { return ptr_hist_; }

 private:
  weight_t const* ptr_weights_{nullptr};
  size_t n_samples_{0};
  weight_t* ptr_hist_{nullptr};
  size_t n_intervals_{0};
  offset_t* ptr_offsets_{nullptr};
};

template <typename exe_policy_t>
struct sync_policy_t<exe_policy_t, std::enable_if_t<is_host_policy_v<exe_policy_t>>> {
  sync_policy_t(void) {}

  void operator()(cudaStream_t stream)
  {
    // purposely empty: there's nothing to sync on host
  }
};

}  // namespace detail
}  // namespace cupynumeric
