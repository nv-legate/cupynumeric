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
#include "cupynumeric/matrix/potrf.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct PotrfImplBody;

template <VariantKind KIND>
struct PotrfImplBody<KIND, Type::Code::FLOAT32> {
  void operator()(float* array, int32_t m, int32_t n);
};

template <VariantKind KIND>
struct PotrfImplBody<KIND, Type::Code::FLOAT64> {
  void operator()(double* array, int32_t m, int32_t n);
};

template <VariantKind KIND>
struct PotrfImplBody<KIND, Type::Code::COMPLEX64> {
  void operator()(complex<float>* array, int32_t m, int32_t n);
};

template <VariantKind KIND>
struct PotrfImplBody<KIND, Type::Code::COMPLEX128> {
  void operator()(complex<double>* array, int32_t m, int32_t n);
};

template <Type::Code CODE>
struct support_potrf : std::false_type {};
template <>
struct support_potrf<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_potrf<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_potrf<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_potrf<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct PotrfImpl {
  template <Type::Code CODE, std::enable_if_t<support_potrf<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore array) const
  {
    using VAL = type_of<CODE>;

    auto shape = array.shape<2>();

    if (shape.empty()) {
      return;
    }

    size_t strides[2];

    auto arr = array.write_accessor<VAL, 2>(shape).ptr(shape, strides);
    auto m   = static_cast<int32_t>(shape.hi[0] - shape.lo[0] + 1);
    auto n   = static_cast<int32_t>(shape.hi[1] - shape.lo[1] + 1);
    assert(m > 0 && n > 0);

    PotrfImplBody<KIND, CODE>()(arr, m, n);
  }

  template <Type::Code CODE, std::enable_if_t<!support_potrf<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void potrf_template(TaskContext& context)
{
  auto array = context.output(0);
  type_dispatch(array.type().code(), PotrfImpl<KIND>{}, array);
}

}  // namespace cupynumeric
