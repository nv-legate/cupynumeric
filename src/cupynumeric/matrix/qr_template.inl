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

#include <vector>

// Useful for IDEs
#include "cupynumeric/matrix/qr.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct QrImplBody;

template <Type::Code CODE>
struct support_qr : std::false_type {};
template <>
struct support_qr<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_qr<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_qr<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_qr<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct QrImpl {
  TaskContext context;
  explicit QrImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, std::enable_if_t<support_qr<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array,
                  legate::PhysicalStore q_array,
                  legate::PhysicalStore r_array) const
  {
    using VAL = type_of<CODE>;

#ifdef DEBUG_CUPYNUMERIC
    assert(a_array.dim() == 2);
    assert(q_array.dim() == 2);
    assert(r_array.dim() == 2);
#endif
    const auto a_shape = a_array.shape<2>();
    const auto q_shape = q_array.shape<2>();
    const auto r_shape = r_array.shape<2>();

    const int64_t m = a_shape.hi[0] - a_shape.lo[0] + 1;
    const int64_t n = a_shape.hi[1] - a_shape.lo[1] + 1;
    const int64_t k = std::min(m, n);

#ifdef DEBUG_CUPYNUMERIC
    assert(q_shape.hi[0] - q_shape.lo[0] + 1 == m);
    assert(q_shape.hi[1] - q_shape.lo[1] + 1 == k);
    assert(r_shape.hi[0] - r_shape.lo[0] + 1 == k);
    assert(r_shape.hi[1] - r_shape.lo[1] + 1 == n);
#endif

    auto a_acc = a_array.read_accessor<VAL, 2>(a_shape);
    auto q_acc = q_array.write_accessor<VAL, 2>(q_shape);
    auto r_acc = r_array.write_accessor<VAL, 2>(r_shape);
#ifdef DEBUG_CUPYNUMERIC
    assert(a_acc.accessor.is_dense_col_major(a_shape));
    assert(q_acc.accessor.is_dense_col_major(q_shape));
    assert(r_acc.accessor.is_dense_col_major(r_shape));
    assert(m > 0 && n > 0 && k > 0);
#endif

    QrImplBody<KIND, CODE>{context}(
      m, n, k, a_acc.ptr(a_shape), q_acc.ptr(q_shape), r_acc.ptr(r_shape));
  }

  template <Type::Code CODE, std::enable_if_t<!support_qr<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array,
                  legate::PhysicalStore q_array,
                  legate::PhysicalStore r_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void qr_template(TaskContext& context)
{
  auto a_array = context.input(0);
  auto q_array = context.output(0);
  auto r_array = context.output(1);
  type_dispatch(a_array.type().code(), QrImpl<KIND>{context}, a_array, q_array, r_array);
}

}  // namespace cupynumeric
