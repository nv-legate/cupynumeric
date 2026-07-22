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

// Useful for IDEs
#include "cupynumeric/stat/histogram.h"
#include "legate/redop/redop.h"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
inline constexpr bool is_candidate = is_floating_point<CODE>::value || is_integral<CODE>::value;

template <typename HistogramImplTy>
static void histogram_impl_type_dispatch(const TaskContext& context, const HistogramImplTy& impl)
{
  // assumes the data array is the first input
  type_dispatch(context.input(0).code(), impl);
}

template <VariantKind KIND, Type::Code CODE>
struct HistogramNoWeightImplBody;

template <VariantKind KIND>
struct HistogramNoWeightImpl {
  TaskContext context;
  explicit HistogramNoWeightImpl(TaskContext context) : context(context) {}

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = std::int64_t;

  template <Type::Code CODE, std::enable_if_t<is_candidate<CODE>>* = nullptr>
  void operator()() const
  {
    using VAL = type_of<CODE>;

    auto src    = context.input(0);
    auto bins   = context.input(1);
    auto result = context.reduction(0);

    auto src_rect    = src.template shape<1>();
    auto bins_rect   = bins.template shape<1>();
    auto result_rect = result.template shape<1>();

    if (src_rect.empty()) {
      return;
    }

    auto src_acc  = src.template read_accessor<VAL, 1>(src_rect);
    auto bins_acc = bins.template read_accessor<BinType, 1>(bins_rect);
    auto result_acc =
      result.template reduce_accessor<SumReduction<WeightType>, true, 1>(result_rect);

    HistogramNoWeightImplBody<KIND, CODE>{context}(
      src_acc, src_rect, bins_acc, bins_rect, result_acc, result_rect);
  }

  template <Type::Code CODE, std::enable_if_t<!is_candidate<CODE>>* = nullptr>
  void operator()() const
  {
    assert(false);
  }
};

template <VariantKind KIND, Type::Code CODE>
struct HistogramWeightedImplBody;

template <VariantKind KIND>
struct HistogramWeightedImpl {
  TaskContext context;
  explicit HistogramWeightedImpl(TaskContext context) : context(context) {}

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = double;

  template <Type::Code CODE, std::enable_if_t<is_candidate<CODE>>* = nullptr>
  void operator()() const
  {
    using VAL = type_of<CODE>;

    auto src     = context.input(0);
    auto bins    = context.input(1);
    auto weights = context.input(2);
    auto result  = context.reduction(0);

    auto src_rect     = src.template shape<1>();
    auto bins_rect    = bins.template shape<1>();
    auto weights_rect = weights.template shape<1>();
    auto result_rect  = result.template shape<1>();

    if (src_rect.empty()) {
      return;
    }

    auto src_acc     = src.template read_accessor<VAL, 1>(src_rect);
    auto bins_acc    = bins.template read_accessor<BinType, 1>(bins_rect);
    auto weights_acc = weights.template read_accessor<WeightType, 1>(weights_rect);
    auto result_acc =
      result.template reduce_accessor<SumReduction<WeightType>, true, 1>(result_rect);

    HistogramWeightedImplBody<KIND, CODE>{context}(
      src_acc, src_rect, bins_acc, bins_rect, weights_acc, weights_rect, result_acc, result_rect);
  }

  template <Type::Code CODE, std::enable_if_t<!is_candidate<CODE>>* = nullptr>
  void operator()() const
  {
    assert(false);
  }
};

}  // namespace cupynumeric
