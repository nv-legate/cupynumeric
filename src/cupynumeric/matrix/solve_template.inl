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
#include "cupynumeric/matrix/solve.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct SolveImplBody;

template <Type::Code CODE>
struct support_solve : std::false_type {};
template <>
struct support_solve<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_solve<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_solve<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_solve<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct SolveImpl {
  template <Type::Code CODE, std::enable_if_t<support_solve<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array, legate::PhysicalStore b_array) const
  {
    using VAL = type_of<CODE>;

#ifdef DEBUG_CUPYNUMERIC
    assert(a_array.dim() == 2);
    assert(b_array.dim() == 1 || b_array.dim() == 2);
#endif
    const auto a_shape = a_array.shape<2>();

    const int64_t m = a_shape.hi[0] - a_shape.lo[0] + 1;
    const int64_t n = a_shape.hi[1] - a_shape.lo[1] + 1;

#ifdef DEBUG_CUPYNUMERIC
    // The Python code guarantees this property
    assert(m == n);
#endif

    size_t a_strides[2];
    VAL* a = a_array.read_write_accessor<VAL, 2>(a_shape).ptr(a_shape, a_strides);
#ifdef DEBUG_CUPYNUMERIC
    assert(a_array.is_future() || (a_strides[0] == 1 && static_cast<int64_t>(a_strides[1]) == m));
#endif
    VAL* b = nullptr;

    int64_t nrhs = 1;
    if (b_array.dim() == 1) {
      const auto b_shape = b_array.shape<1>();
#ifdef DEBUG_CUPYNUMERIC
      assert(m == b_shape.hi[0] - b_shape.lo[0] + 1);
#endif
      size_t b_strides;
      b = b_array.read_write_accessor<VAL, 1>(b_shape).ptr(b_shape, &b_strides);
    } else {
      const auto b_shape = b_array.shape<2>();
#ifdef DEBUG_CUPYNUMERIC
      assert(m == b_shape.hi[0] - b_shape.lo[0] + 1);
#endif
      nrhs = b_shape.hi[1] - b_shape.lo[1] + 1;
      size_t b_strides[2];
      b = b_array.read_write_accessor<VAL, 2>(b_shape).ptr(b_shape, b_strides);
#ifdef DEBUG_CUPYNUMERIC
      assert(b_array.is_future() ||
             (b_strides[0] == 1 && (nrhs == 1 || static_cast<int64_t>(b_strides[1]) == m)));
#endif
    }

#ifdef DEBUG_CUPYNUMERIC
    assert(m > 0 && n > 0 && nrhs > 0);
#endif

    SolveImplBody<KIND, CODE>()(m, n, nrhs, a, b);
  }

  template <Type::Code CODE, std::enable_if_t<!support_solve<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array, legate::PhysicalStore b_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void solve_template(TaskContext& context)
{
  auto a_array = context.output(0);
  auto b_array = context.output(1);
  type_dispatch(a_array.type().code(), SolveImpl<KIND>{}, a_array, b_array);
}

}  // namespace cupynumeric
