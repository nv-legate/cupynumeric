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
  TaskContext context;
  explicit MatMulImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, std::enable_if_t<support_matmul<CODE>::value>* = nullptr>
  void operator()(MatMulArgs& args) const
  {
    if (args.partition_type == PartitionStrategy::Batched) {
      batched_partitioning<CODE>(args);
    } else {
      unbatched_partitioning<CODE>(args);
    }
  }

  template <Type::Code CODE, std::enable_if_t<!support_matmul<CODE>::value>* = nullptr>
  void operator()(MatMulArgs& args) const
  {
    assert(false);
  }

 private:
  template <Type::Code CODE>
  void batched_partitioning(MatMulArgs& args) const
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

    MatMulImplBody<KIND, CODE>{context}(m,
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

  template <Type::Code CODE>
  void unbatched_partitioning(MatMulArgs& args) const
  {
    using VAL = type_of<CODE>;
    using ACC = typename support_matmul<CODE>::ACC_TYPE;

    // Note that rhs1 and rhs2 may have different shapes. Here's why: rhs1 and rhs2 are promoted
    // on one of their dimensions, and in case that the promoted dimension is partitioned,
    // the store cannot see that partitioning, because that dimension doesn't map to the store's
    // original domain whose partitioning is only what the store can observe. Therefore, we must
    // take an intersection of the rhs1's and rhs2's shapes to get a correct "active" area
    // in their bloated domains.
    auto shape = args.rhs1.shape<3>().intersection(args.rhs2.shape<3>());

    if (shape.empty()) {
      return;
    }

    const auto m = shape.hi[0] - shape.lo[0] + 1;
    const auto k = shape.hi[1] - shape.lo[1] + 1;
    const auto n = shape.hi[2] - shape.lo[2] + 1;

    size_t lhs_strides[3];
    size_t rhs1_strides[3];
    size_t rhs2_strides[3];

    auto rhs1 = args.rhs1.read_accessor<VAL, 3>(shape).ptr(shape, rhs1_strides);
    auto rhs2 = args.rhs2.read_accessor<VAL, 3>(shape).ptr(shape, rhs2_strides);
    auto lhs  = args.lhs.reduce_accessor<SumReduction<ACC>, true, 3>(shape).ptr(shape, lhs_strides);

#ifdef DEBUG_CUPYNUMERIC
    assert(rhs1_strides[2] == 0);
    assert(rhs2_strides[0] == 0);
    assert(lhs_strides[2] == 1 && lhs_strides[1] == 0);
#endif

    bool rhs1_transposed;
    bool rhs2_transposed;
    size_t rhs1_stride = stride_for_blas(m, k, rhs1_strides[0], rhs1_strides[1], rhs1_transposed);
    size_t rhs2_stride = stride_for_blas(k, n, rhs2_strides[1], rhs2_strides[2], rhs2_transposed);

    MatMulImplBody<KIND, CODE>{context}(m,
                                        n,
                                        k,
                                        lhs,
                                        rhs1,
                                        rhs2,
                                        lhs_strides[0],
                                        rhs1_stride,
                                        rhs2_stride,
                                        rhs1_transposed,
                                        rhs2_transposed,
                                        args.lhs.is_readable());
  }
};

template <VariantKind KIND>
static void matmul_template(TaskContext& context)
{
  auto inputs  = context.inputs();
  auto scalars = context.scalars();

  PartitionStrategy partition_type{PartitionStrategy::Batched};
  if (scalars.size() > 0) {
    int partition_selector = scalars[0].value<int>();
    partition_type         = static_cast<PartitionStrategy>(partition_selector);
  }

  if (partition_type == PartitionStrategy::Batched) {
    auto outputs = context.outputs();
    MatMulArgs args{outputs[0], inputs[1], inputs[2], partition_type};

    // Note that we can't dispatch on the lhs's type,
    // as the lhs can have a different type than the rhs'
    type_dispatch(args.rhs1.code(), MatMulImpl<KIND>{context}, args);
  } else {
    auto reductions = context.reductions();
#ifdef DEBUG_CUPYNUMERIC
    assert(inputs.size() == 2);
    assert(reductions.size() == 1);
#endif
    MatMulArgs args{reductions[0], inputs[0], inputs[1], partition_type};

    // Note that we can't dispatch on the lhs's type,
    // as the lhs can have a different type than the rhs'
    type_dispatch(args.rhs1.code(), MatMulImpl<KIND>{context}, args);
  }
}

}  // namespace cupynumeric
