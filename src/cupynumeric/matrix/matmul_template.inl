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
#include "cupynumeric/matrix/matmul.h"
#include "cupynumeric/matrix/util.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct MatMulImplBody;

template <Type::Code CODE>
struct support_matmul : std::false_type {};
template <>
struct support_matmul<Type::Code::FLOAT64> : std::true_type {
  using ACC_TYPE = double;
};
template <>
struct support_matmul<Type::Code::FLOAT32> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matmul<Type::Code::FLOAT16> : std::true_type {
  using ACC_TYPE = float;
};
template <>
struct support_matmul<Type::Code::COMPLEX64> : std::true_type {
  using ACC_TYPE = complex<float>;
};
template <>
struct support_matmul<Type::Code::COMPLEX128> : std::true_type {
  using ACC_TYPE = complex<double>;
};

template <VariantKind KIND>
struct MatMulImpl {
  template <Type::Code CODE, std::enable_if_t<support_matmul<CODE>::value>* = nullptr>
  void operator()(MatMulArgs& args) const
  {
    using VAL = type_of<CODE>;
    using ACC = typename support_matmul<CODE>::ACC_TYPE;

    auto shape_lhs  = args.lhs.shape<2>();
    auto shape_rhs1 = args.rhs1.shape<2>();
    auto shape_rhs2 = args.rhs2.shape<2>();

    if (shape_lhs.empty() || shape_rhs1.empty() || shape_rhs2.empty()) {
      return;
    }

    const auto m = shape_lhs.hi[0] - shape_lhs.lo[0] + 1;
    const auto n = shape_lhs.hi[1] - shape_lhs.lo[1] + 1;
    const auto k = shape_rhs1.hi[1] - shape_rhs1.lo[1] + 1;

#ifdef DEBUG_CUPYNUMERIC
    assert(m == shape_rhs1.hi[0] - shape_rhs1.lo[0] + 1);
    assert(k == shape_rhs2.hi[0] - shape_rhs2.lo[0] + 1);
    assert(n == shape_rhs2.hi[1] - shape_rhs2.lo[1] + 1);
#endif

    size_t strides_lhs[2];
    size_t strides_rhs1[2];
    size_t strides_rhs2[2];

    auto rhs1 = args.rhs1.read_accessor<VAL, 2>(shape_rhs1).ptr(shape_rhs1, strides_rhs1);
    auto rhs2 = args.rhs2.read_accessor<VAL, 2>(shape_rhs2).ptr(shape_rhs2, strides_rhs2);
    auto lhs  = args.lhs.read_write_accessor<ACC, 2>(shape_lhs).ptr(shape_lhs, strides_lhs);

#ifdef DEBUG_CUPYNUMERIC
    assert(strides_rhs1[0] == 1 || strides_rhs1[1] == 1);
    assert(strides_rhs2[0] == 1 || strides_rhs2[1] == 1);
    assert(strides_lhs[1] == 1);
#endif

    bool transposed_rhs1;
    bool transposed_rhs2;
    size_t stride_rhs1 = stride_for_blas(m, k, strides_rhs1[0], strides_rhs1[1], transposed_rhs1);
    size_t stride_rhs2 = stride_for_blas(k, n, strides_rhs2[0], strides_rhs2[1], transposed_rhs2);

    MatMulImplBody<KIND, CODE>()(m,
                                 n,
                                 k,
                                 lhs,
                                 rhs1,
                                 rhs2,
                                 strides_lhs[0],
                                 stride_rhs1,
                                 stride_rhs2,
                                 transposed_rhs1,
                                 transposed_rhs2,
                                 /*args.lhs.is_readable()*/ false);
  }

  template <Type::Code CODE, std::enable_if_t<!support_matmul<CODE>::value>* = nullptr>
  void operator()(MatMulArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void matmul_template(TaskContext& context)
{
  auto outputs = context.outputs();
  auto inputs  = context.inputs();

  MatMulArgs args{outputs[0], inputs[1], inputs[2]};
  // Note that we can't dispatch on the lhs's type,
  // as the lhs can have a different type than the rhs'
  type_dispatch(args.rhs1.code(), MatMulImpl<KIND>{}, args);
}

}  // namespace cupynumeric
