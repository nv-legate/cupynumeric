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

#include "cupynumeric/stat/histogram_gen.h"

#include <legate/redop/base.h>

namespace cupynumeric {
namespace detail {

template <typename T>
__CUDA_HD__ size_t find_bucket(const T elem, const T* bins_ptr, size_t bins_size)
{
  std::size_t high = bins_size - 1;
  std::size_t low  = 0;

  while ((high - low) > 1) {
    auto mid = (low + high) / 2;
    assert(mid < bins_size);
    if (elem < bins_ptr[mid]) {
      high = mid;
    } else {
      low = mid;
    }
  }

  return low;
}

enum class BucketUpdate : std::int8_t { NO_WEIGHT, WEIGHTED };

template <BucketUpdate UPD_KIND, typename elem_t, typename bin_t, typename weight_t>
struct PerElemFunc {
  const elem_t* src_ptr;
  const size_t src_size;
  const bin_t* bins_ptr;
  const size_t bins_size;
  const weight_t* weights_ptr;
  weight_t* result_ptr;

  __CUDA_HD__ void operator()(const size_t idx)
  {
    auto xc = static_cast<bin_t>(src_ptr[idx]);

    assert(bins_size > 1);

    if (isnan(xc)) {
      return;
    }

    if (xc < bins_ptr[0] || xc > bins_ptr[bins_size - 1]) {
      return;
    }

    size_t bucket = find_bucket(xc, bins_ptr, bins_size);

    if constexpr (UPD_KIND == BucketUpdate::NO_WEIGHT) {
      legate::SumReduction<weight_t>::template apply</*EXCLUSIVE=*/false>(result_ptr[bucket],
                                                                          weight_t{1});
    } else {
      legate::SumReduction<weight_t>::template apply</*EXCLUSIVE=*/false>(result_ptr[bucket],
                                                                          weights_ptr[idx]);
    }
  }
};

template <BucketUpdate UPD_KIND, typename elem_t, typename bin_t, typename weight_t>
auto make_per_elem_functor(const elem_t* src_ptr,
                           const size_t src_size,
                           const bin_t* bins_ptr,
                           const size_t bins_size,
                           const weight_t* weights_ptr,
                           weight_t* result_ptr)
{
  return PerElemFunc<UPD_KIND, elem_t, bin_t, weight_t>{
    src_ptr, src_size, bins_ptr, bins_size, weights_ptr, result_ptr};
}

// for cpu/omp or when number of bins >= 2^32
template <typename exe_policy_t, typename elem_t, typename bin_t, typename weight_t>
void histogram_no_weight_thrust(exe_policy_t exe_pol,
                                const AccessorRO<elem_t, 1>& src,
                                const Rect<1>& src_rect,
                                const AccessorRO<bin_t, 1>& bins,
                                const Rect<1>& bins_rect,
                                const AccessorRD<SumReduction<weight_t>, true, 1>& result,
                                const Rect<1>& result_rect)
{
  auto [src_size, src_ptr]       = accessors::get_accessor_ptr(src, src_rect);
  auto [bins_size, bins_ptr]     = accessors::get_accessor_ptr(bins, bins_rect);
  auto [result_size, result_ptr] = accessors::get_accessor_ptr(result, result_rect);

  thrust::for_each(
    exe_pol,
    thrust::counting_iterator<size_t>{0},
    thrust::counting_iterator<size_t>{src_size},
    make_per_elem_functor<BucketUpdate::NO_WEIGHT>(
      src_ptr, src_size, bins_ptr, bins_size, static_cast<const weight_t*>(nullptr), result_ptr));
}

template <typename exe_policy_t, typename elem_t, typename bin_t, typename weight_t>
void histogram_weighted_thrust(exe_policy_t exe_pol,
                               const AccessorRO<elem_t, 1>& src,
                               const Rect<1>& src_rect,
                               const AccessorRO<bin_t, 1>& bins,
                               const Rect<1>& bins_rect,
                               const AccessorRO<weight_t, 1>& weights,
                               const Rect<1>& weights_rect,
                               const AccessorRD<SumReduction<weight_t>, true, 1>& result,
                               const Rect<1>& result_rect,
                               cudaStream_t stream = {})
{
  auto [src_size, src_ptr]         = accessors::get_accessor_ptr(src, src_rect);
  auto [bins_size, bins_ptr]       = accessors::get_accessor_ptr(bins, bins_rect);
  auto [weights_size, weights_ptr] = accessors::get_accessor_ptr(weights, weights_rect);
  auto [result_size, result_ptr]   = accessors::get_accessor_ptr(result, result_rect);

  assert(weights_size == src_size);
  assert(bins_size == result_size + 1);

  // TODO(amberhassaan): replace with custom kernel for GPU case
  thrust::for_each(exe_pol,
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(src_size),
                   make_per_elem_functor<BucketUpdate::WEIGHTED>(
                     src_ptr, src_size, bins_ptr, bins_size, weights_ptr, result_ptr));
}

}  // namespace detail
}  // namespace cupynumeric
