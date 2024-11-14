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
#include "cupynumeric/matrix/trsm.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct TrsmImplBody;

template <Type::Code CODE>
struct support_trsm : std::false_type {};
template <>
struct support_trsm<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_trsm<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_trsm<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_trsm<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct TrsmImpl {
  template <Type::Code CODE, std::enable_if_t<support_trsm<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore lhs_array, legate::PhysicalStore rhs_array) const
  {
    using VAL = type_of<CODE>;

    auto lhs_shape = lhs_array.shape<2>();
    auto rhs_shape = rhs_array.shape<2>();

    if (lhs_shape.empty()) {
      return;
    }

    size_t lhs_strides[2];
    size_t rhs_strides[2];

    auto lhs = lhs_array.write_accessor<VAL, 2>(lhs_shape).ptr(lhs_shape, lhs_strides);
    auto rhs = rhs_array.read_accessor<VAL, 2>(rhs_shape).ptr(rhs_shape, rhs_strides);

    auto m = static_cast<int32_t>(lhs_shape.hi[0] - lhs_shape.lo[0] + 1);
    auto n = static_cast<int32_t>(lhs_shape.hi[1] - lhs_shape.lo[1] + 1);
    assert(m > 0 && n > 0);

    TrsmImplBody<KIND, CODE>()(lhs, rhs, m, n);
  }

  template <Type::Code CODE, std::enable_if_t<!support_trsm<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore lhs_array, legate::PhysicalStore rhs_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void trsm_template(TaskContext& context)
{
  auto lhs = context.output(0);
  auto rhs = context.input(0);

  type_dispatch(lhs.type().code(), TrsmImpl<KIND>{}, lhs, rhs);
}

}  // namespace cupynumeric
