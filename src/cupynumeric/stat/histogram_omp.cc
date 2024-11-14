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

#include "cupynumeric/stat/histogram.h"
#include "cupynumeric/stat/histogram_template.inl"

#define _USE_THRUST_

#ifdef _USE_THRUST_
#include "cupynumeric/stat/histogram_cpu.h"
#include "cupynumeric/stat/histogram_impl.h"
#endif

#include <omp.h>

#include <algorithm>
#include <numeric>
#include <tuple>

namespace cupynumeric {
using namespace legate;

template <Type::Code CODE>
struct HistogramImplBody<VariantKind::OMP, CODE> {
  using VAL = type_of<CODE>;

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = double;

  // in the future we might relax relax that requirement,
  // but complicate dispatching:
  //
  // template <typename BinType = VAL, typename WeightType = VAL>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRO<WeightType, 1>& weights,
                  const Rect<1>& weights_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto exe_pol = thrust::omp::par;
#ifdef _USE_THRUST_
    detail::histogram_wrapper(
      exe_pol, src, src_rect, bins, bins_rect, weights, weights_rect, result, result_rect);
#else
    namespace det_acc                              = detail::accessors;
    auto&& [global_result_size, global_result_ptr] = det_acc::get_accessor_ptr(result, result_rect);
    auto&& [src_size, src_ptr]                     = det_acc::get_accessor_ptr(src, src_rect);

    auto&& [weights_size, weights_ptr] = det_acc::get_accessor_ptr(weights, weights_rect);

    auto&& [bins_size, bins_ptr] = det_acc::get_accessor_ptr(bins, bins_rect);

    auto num_intervals              = bins_size - 1;
    Buffer<WeightType> local_result = create_buffer<WeightType>(num_intervals);

    WeightType* local_result_ptr = local_result.ptr(0);

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (auto bin_index = 0; bin_index < num_intervals; ++bin_index) {
        local_result_ptr[bin_index] = 0;

        if (bin_index == num_intervals - 1) {  // interior[l, r]
          for (auto src_index = 0; src_index < src_size; ++src_index) {
            if (src_ptr[src_index] >= bins_ptr[bin_index] &&
                src_ptr[src_index] <= bins_ptr[bin_index + 1]) {
              local_result_ptr[bin_index] += weights_ptr[src_index];
            }
          }
        } else {  // interior[l, r)
          for (auto src_index = 0; src_index < src_size; ++src_index) {
            if (src_ptr[src_index] >= bins_ptr[bin_index] &&
                src_ptr[src_index] < bins_ptr[bin_index + 1]) {
              local_result_ptr[bin_index] += weights_ptr[src_index];
            }
          }
        }
      }

      if (true) {
#pragma omp barrier
      }

      // fold into RD result:
      //
      assert(num_intervals == global_result_size);

#pragma omp for schedule(static)
      for (auto bin_index = 0; bin_index < num_intervals; ++bin_index) {
        global_result_ptr[bin_index] += local_result_ptr[bin_index];
      }
    }
#endif
  }
};

/*static*/ void HistogramTask::omp_variant(TaskContext context)
{
  histogram_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
