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

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct HistogramImplBody;

template <Type::Code CODE>
inline constexpr bool is_candidate = (is_floating_point<CODE>::value || is_integral<CODE>::value);

template <VariantKind KIND>
struct HistogramImpl {
  TaskContext context;
  explicit HistogramImpl(TaskContext context) : context(context) {}

  // for now, it has been decided to hardcode these types:
  //
  using BinType    = double;
  using WeightType = double;

  template <Type::Code CODE, std::enable_if_t<is_candidate<CODE>>* = nullptr>
  void operator()(HistogramArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto result_rect  = args.result.shape<1>();
    auto src_rect     = args.src.shape<1>();
    auto bins_rect    = args.bins.shape<1>();
    auto weights_rect = args.weights.shape<1>();

    if (src_rect.empty()) {
      return;
    }

    auto result  = args.result.reduce_accessor<SumReduction<WeightType>, true, 1>(result_rect);
    auto src     = args.src.read_accessor<VAL, 1>(src_rect);
    auto bins    = args.bins.read_accessor<BinType, 1>(bins_rect);
    auto weights = args.weights.read_accessor<WeightType, 1>(weights_rect);

    HistogramImplBody<KIND, CODE>{context}(
      src, src_rect, bins, bins_rect, weights, weights_rect, result, result_rect);
  }

  template <Type::Code CODE, std::enable_if_t<!is_candidate<CODE>>* = nullptr>
  void operator()(HistogramArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void histogram_template(TaskContext& context)
{
  auto inputs     = context.inputs();
  auto reductions = context.reductions();
  HistogramArgs args{reductions[0], inputs[0], inputs[1], inputs[2]};
  type_dispatch(args.src.code(), HistogramImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
