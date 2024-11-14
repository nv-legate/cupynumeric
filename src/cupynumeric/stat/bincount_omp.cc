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

#include "cupynumeric/stat/bincount.h"
#include "cupynumeric/stat/bincount_template.inl"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct BincountImplBody<VariantKind::OMP, CODE> {
  using VAL = type_of<CODE>;

  std::vector<std::vector<int64_t>> _bincount(const AccessorRO<VAL, 1>& rhs,
                                              const Rect<1>& rect,
                                              const Rect<1>& lhs_rect) const
  {
    const int max_threads   = omp_get_max_threads();
    const size_t lhs_volume = lhs_rect.volume();
    std::vector<std::vector<int64_t>> all_local_bins(max_threads);
    for (auto& local_bins : all_local_bins) {
      auto init  = SumReduction<int64_t>::identity;
      local_bins = std::vector<int64_t>(lhs_volume, init);
    }
#pragma omp parallel
    {
      auto tid                         = omp_get_thread_num();
      std::vector<int64_t>& local_bins = all_local_bins[tid];
#pragma omp for schedule(static)
      for (int64_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
        auto value = rhs[idx];
        assert(lhs_rect.contains(value));
        SumReduction<int64_t>::fold<true>(local_bins[value], 1);
      }
    }
    return all_local_bins;
  }

  std::vector<std::vector<double>> _bincount(const AccessorRO<VAL, 1>& rhs,
                                             const AccessorRO<double, 1>& weights,
                                             const Rect<1>& rect,
                                             const Rect<1>& lhs_rect) const
  {
    const int max_threads   = omp_get_max_threads();
    const size_t lhs_volume = lhs_rect.volume();
    std::vector<std::vector<double>> all_local_bins(max_threads);
    for (auto& local_bins : all_local_bins) {
      auto init  = SumReduction<double>::identity;
      local_bins = std::vector<double>(lhs_volume, init);
    }
#pragma omp parallel
    {
      auto tid                        = omp_get_thread_num();
      std::vector<double>& local_bins = all_local_bins[tid];
#pragma omp for schedule(static)
      for (int64_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
        auto value = rhs[idx];
        assert(lhs_rect.contains(value));
        SumReduction<double>::fold<true>(local_bins[value], weights[idx]);
      }
    }
    return all_local_bins;
  }

  void operator()(AccessorRD<SumReduction<int64_t>, true, 1> lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    auto all_local_bins = _bincount(rhs, rect, lhs_rect);
    for (auto& local_bins : all_local_bins) {
      for (size_t bin_num = 0; bin_num < local_bins.size(); ++bin_num) {
        lhs.reduce(bin_num, local_bins[bin_num]);
      }
    }
  }

  void operator()(AccessorRD<SumReduction<double>, true, 1> lhs,
                  const AccessorRO<VAL, 1>& rhs,
                  const AccessorRO<double, 1>& weights,
                  const Rect<1>& rect,
                  const Rect<1>& lhs_rect) const
  {
    auto all_local_bins = _bincount(rhs, weights, rect, lhs_rect);
    for (auto& local_bins : all_local_bins) {
      for (size_t bin_num = 0; bin_num < local_bins.size(); ++bin_num) {
        lhs.reduce(bin_num, local_bins[bin_num]);
      }
    }
  }
};

/*static*/ void BincountTask::omp_variant(TaskContext context)
{
  bincount_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
