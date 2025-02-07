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
#include "cupynumeric/matrix/svd.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct SvdImplBody;

template <Type::Code CODE>
struct support_svd : std::false_type {};
template <>
struct support_svd<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_svd<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_svd<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_svd<Type::Code::COMPLEX128> : std::true_type {};

template <Type::Code CODE>
struct real_type {
  using TYPE = float;
};
template <>
struct real_type<Type::Code::FLOAT64> {
  using TYPE = double;
};
template <>
struct real_type<Type::Code::COMPLEX128> {
  using TYPE = double;
};

template <VariantKind KIND>
struct SvdImpl {
  template <Type::Code CODE, std::enable_if_t<support_svd<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array,
                  legate::PhysicalStore u_array,
                  legate::PhysicalStore s_array,
                  legate::PhysicalStore vh_array) const
  {
    using VAL      = type_of<CODE>;
    using VAL_REAL = typename real_type<CODE>::TYPE;

#ifdef DEBUG_CUPYNUMERIC
    assert(a_array.dim() == 2);
    assert(u_array.dim() == 2);
    assert(s_array.dim() == 1);
    assert(vh_array.dim() == 2);
#endif
    const auto a_shape  = a_array.shape<2>();
    const auto u_shape  = u_array.shape<2>();
    const auto s_shape  = s_array.shape<1>();
    const auto vh_shape = vh_array.shape<2>();

    const int64_t m = a_shape.hi[0] - a_shape.lo[0] + 1;
    const int64_t n = a_shape.hi[1] - a_shape.lo[1] + 1;
    const int64_t k = std::min(m, n);

    assert(m >= n);
    bool full_matrices = (u_shape.hi[1] - u_shape.lo[1] + 1 == m);

#ifdef DEBUG_CUPYNUMERIC
    assert(u_shape.hi[0] - u_shape.lo[0] + 1 == m);
    if (full_matrices) {
      assert(u_shape.hi[1] - u_shape.lo[1] + 1 == m);
    } else {
      assert(u_shape.hi[1] - u_shape.lo[1] + 1 == k);
    }
    assert(s_shape.hi[0] - s_shape.lo[0] + 1 == k);
    assert(vh_shape.hi[0] - vh_shape.lo[0] + 1 == n);
    assert(vh_shape.hi[1] - vh_shape.lo[1] + 1 == n);
#endif

    auto a_acc  = a_array.read_accessor<VAL, 2>(a_shape);
    auto u_acc  = u_array.write_accessor<VAL, 2>(u_shape);
    auto s_acc  = s_array.write_accessor<VAL_REAL, 1>(s_shape);
    auto vh_acc = vh_array.write_accessor<VAL, 2>(vh_shape);
#ifdef DEBUG_CUPYNUMERIC
    assert(a_acc.accessor.is_dense_col_major(a_shape));
    assert(u_acc.accessor.is_dense_col_major(u_shape));
    assert(vh_acc.accessor.is_dense_col_major(vh_shape));
    assert(m > 0 && n > 0 && k > 0);
#endif

    SvdImplBody<KIND, CODE>()(m,
                              n,
                              k,
                              full_matrices,
                              a_acc.ptr(a_shape),
                              u_acc.ptr(u_shape),
                              s_acc.ptr(s_shape),
                              vh_acc.ptr(vh_shape));
  }

  template <Type::Code CODE, std::enable_if_t<!support_svd<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array,
                  legate::PhysicalStore u_array,
                  legate::PhysicalStore s_array,
                  legate::PhysicalStore vh_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void svd_template(TaskContext& context)
{
  auto a_array  = context.input(0);
  auto u_array  = context.output(0);
  auto s_array  = context.output(1);
  auto vh_array = context.output(2);
  type_dispatch(a_array.type().code(), SvdImpl<KIND>{}, a_array, u_array, s_array, vh_array);
}

}  // namespace cupynumeric
