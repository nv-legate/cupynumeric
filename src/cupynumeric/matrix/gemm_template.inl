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
#include "cupynumeric/matrix/gemm.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct GemmImplBody;

template <Type::Code CODE>
struct support_gemm : std::false_type {};
template <>
struct support_gemm<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_gemm<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_gemm<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_gemm<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct GemmImpl {
  TaskContext context;
  explicit GemmImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, std::enable_if_t<support_gemm<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore lhs_array,
                  legate::PhysicalStore rhs1_array,
                  legate::PhysicalStore rhs2_array) const
  {
    using VAL = type_of<CODE>;

    auto lhs_shape  = lhs_array.shape<2>();
    auto rhs1_shape = rhs1_array.shape<2>();
    auto rhs2_shape = rhs2_array.shape<2>();

    if (lhs_shape.empty()) {
      return;
    }

    size_t lhs_strides[2];
    size_t rhs1_strides[2];
    size_t rhs2_strides[2];

    auto lhs  = lhs_array.write_accessor<VAL, 2>(lhs_shape).ptr(lhs_shape, lhs_strides);
    auto rhs1 = rhs1_array.read_accessor<VAL, 2>(rhs1_shape).ptr(rhs1_shape, rhs1_strides);
    auto rhs2 = rhs2_array.read_accessor<VAL, 2>(rhs2_shape).ptr(rhs2_shape, rhs2_strides);

    auto m = static_cast<int32_t>(lhs_shape.hi[0] - lhs_shape.lo[0] + 1);
    auto n = static_cast<int32_t>(lhs_shape.hi[1] - lhs_shape.lo[1] + 1);
    auto k = static_cast<int32_t>(rhs1_shape.hi[1] - rhs1_shape.lo[1] + 1);
    assert(rhs2_shape.hi[0] - rhs2_shape.lo[0] + 1 == n);
    assert(rhs2_shape.hi[1] - rhs2_shape.lo[1] + 1 == k);

    GemmImplBody<KIND, CODE>{context}(lhs, rhs1, rhs2, m, n, k);
  }

  template <Type::Code CODE, std::enable_if_t<!support_gemm<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore lhs_array,
                  legate::PhysicalStore rhs1_array,
                  legate::PhysicalStore rhs2_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void gemm_template(TaskContext& context)
{
  auto inputs  = context.inputs();
  auto outputs = context.outputs();

  auto& lhs  = outputs[0];
  auto& rhs1 = inputs[0];
  auto& rhs2 = inputs[1];

  type_dispatch(lhs.type().code(), GemmImpl<KIND>{context}, lhs, rhs1, rhs2);
}

}  // namespace cupynumeric
