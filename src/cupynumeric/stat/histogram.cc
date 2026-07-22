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

#include "cupynumeric/stat/histogram_cpu.h"
#include "cupynumeric/stat/histogram_impl.h"

#include <algorithm>
#include <numeric>
#include <tuple>

// #define _DEBUG
#ifdef _DEBUG
#include <iostream>
#include <iterator>
#endif

namespace cupynumeric {
using namespace legate;

template <Type::Code CODE>
struct HistogramNoWeightImplBody<VariantKind::CPU, CODE> {
  TaskContext context;
  explicit HistogramNoWeightImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  template <typename BinType, typename WeightType>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto exe_pol = thrust::host;

    detail::histogram_no_weight_thrust(
      exe_pol, src, src_rect, bins, bins_rect, result, result_rect);
  }
};

/*static*/ void HistogramNoWeightTask::cpu_variant(TaskContext context)
{
  histogram_impl_type_dispatch(context, HistogramNoWeightImpl<VariantKind::CPU>{context});
}

template <Type::Code CODE>
struct HistogramWeightedImplBody<VariantKind::CPU, CODE> {
  TaskContext context;
  explicit HistogramWeightedImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  template <typename BinType, typename WeightType>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRO<WeightType, 1>& weights,
                  const Rect<1>& weights_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto exe_pol = thrust::host;

    detail::histogram_weighted_thrust(
      exe_pol, src, src_rect, bins, bins_rect, weights, weights_rect, result, result_rect);
  }
};

/*static*/ void HistogramWeightedTask::cpu_variant(TaskContext context)
{
  histogram_impl_type_dispatch(context, HistogramWeightedImpl<VariantKind::CPU>{context});
}

namespace  // unnamed
{
const auto cupynumeric_reg_task_ = []() -> char {
  HistogramNoWeightTask::register_variants();
  HistogramWeightedTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
