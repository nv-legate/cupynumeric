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
#include "cupynumeric/stat/bincount.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct BincountImplBody;

template <VariantKind KIND>
struct BincountImpl {
  TaskContext context;
  explicit BincountImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, std::enable_if_t<is_integral<CODE>::value>* = nullptr>
  void operator()(BincountArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto rect     = args.rhs.shape<1>();
    auto lhs_rect = args.lhs.shape<1>();
    if (rect.empty()) {
      return;
    }

    auto rhs = args.rhs.read_accessor<VAL, 1>(rect);
    if (args.has_weights) {
      auto weights = args.weights.read_accessor<double, 1>(rect);
      auto lhs =
        args.lhs.reduce_accessor<SumReduction<double>, KIND != VariantKind::GPU, 1>(lhs_rect);
      BincountImplBody<KIND, CODE>{context}(lhs, rhs, weights, rect, lhs_rect);
    } else {
      auto lhs =
        args.lhs.reduce_accessor<SumReduction<int64_t>, KIND != VariantKind::GPU, 1>(lhs_rect);
      BincountImplBody<KIND, CODE>{context}(lhs, rhs, rect, lhs_rect);
    }
  }

  template <Type::Code CODE, std::enable_if_t<!is_integral<CODE>::value>* = nullptr>
  void operator()(BincountArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void bincount_template(TaskContext& context)
{
  auto inputs     = context.inputs();
  auto reductions = context.reductions();

  BincountArgs args;
  args.lhs = std::move(reductions[0]);
  args.rhs = std::move(inputs[0]);
  if (inputs.size() >= 2) {
    args.has_weights = true;
    args.weights     = std::move(inputs[1]);
  } else {
    args.has_weights = false;
  }

  type_dispatch(args.rhs.code(), BincountImpl<KIND>{context}, args);
}

}  // namespace cupynumeric
