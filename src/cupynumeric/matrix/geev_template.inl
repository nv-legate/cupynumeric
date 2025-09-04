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
#include "cupynumeric/matrix/geev.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct GeevImplBody;

template <Type::Code CODE>
struct support_geev : std::false_type {};
template <>
struct support_geev<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_geev<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_geev<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_geev<Type::Code::COMPLEX128> : std::true_type {};

template <Type::Code CODE>
struct complex_type {
  using TYPE = complex<float>;
};
template <>
struct complex_type<Type::Code::FLOAT64> {
  using TYPE = complex<double>;
};
template <>
struct complex_type<Type::Code::COMPLEX128> {
  using TYPE = complex<double>;
};

template <VariantKind KIND>
struct GeevImpl {
  TaskContext context;
  explicit GeevImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<support_geev<CODE>::value && DIM >= 2>* = nullptr>
  void operator()(TaskContext& context) const
  {
    using VAL         = type_of<CODE>;
    using VAL_COMPLEX = typename complex_type<CODE>::TYPE;

    legate::PhysicalStore a_array  = context.input(0);
    legate::PhysicalStore ew_array = context.output(0);

#ifdef DEBUG_CUPYNUMERIC
    assert(a_array.dim() >= 2);
    assert(a_array.dim() == DIM);
    assert(ew_array.dim() == DIM - 1);
#endif
    const auto a_shape  = a_array.shape<DIM>();
    const auto ew_shape = ew_array.shape<DIM - 1>();

    if (a_shape.empty()) {
      return;
    }

    int64_t batchsize_total = 1;
    std::vector<int64_t> batchdims;
    for (auto i = 0; i < DIM - 2; ++i) {
      batchdims.push_back(a_shape.hi[i] - a_shape.lo[i] + 1);
      batchsize_total *= batchdims.back();
    }

    const int64_t m = a_shape.hi[DIM - 1] - a_shape.lo[DIM - 1] + 1;

#ifdef DEBUG_CUPYNUMERIC
    assert(m > 0);
    assert(batchsize_total > 0);
    assert(a_shape.hi[DIM - 2] - a_shape.lo[DIM - 2] + 1 == m);
    assert(ew_shape.hi[DIM - 2] - ew_shape.lo[DIM - 2] + 1 == m);
    for (auto i = 0; i < batchdims.size(); ++i) {
      assert(ew_shape.hi[i] - ew_shape.lo[i] + 1 == batchdims[i]);
    }
#endif
    size_t a_strides[DIM];
    size_t ew_strides[DIM - 1];
    size_t ev_strides[DIM];

    auto* a_acc = a_array.read_accessor<VAL, DIM>(a_shape).ptr(a_shape, a_strides);
    auto* ew_acc =
      ew_array.write_accessor<VAL_COMPLEX, DIM - 1>(ew_shape).ptr(ew_shape, ew_strides);
    VAL_COMPLEX* ev_acc = nullptr;

    // optional computation of eigenvectors
    bool compute_evs = context.outputs().size() > 1;
    if (compute_evs) {
      legate::PhysicalStore ev_array = context.output(1);
#ifdef DEBUG_CUPYNUMERIC
      assert(ev_array.dim() == DIM);
#endif
      const auto ev_shape = ev_array.shape<DIM>();
#ifdef DEBUG_CUPYNUMERIC
      assert(ev_shape.hi[DIM - 2] - ev_shape.lo[DIM - 2] + 1 == m);
      assert(ev_shape.hi[DIM - 1] - ev_shape.lo[DIM - 1] + 1 == m);
      for (auto i = 0; i < batchdims.size(); ++i) {
        assert(ev_shape.hi[i] - ev_shape.lo[i] + 1 == batchdims[i]);
      }
#endif
      ev_acc = ev_array.write_accessor<VAL_COMPLEX, DIM>(ev_shape).ptr(ev_shape, ev_strides);
    }

#ifdef DEBUG_CUPYNUMERIC
    int64_t full_stride = m * m;
    for (int i = batchdims.size() - 1; i >= 0; --i) {
      if (batchdims[i] > 1) {
        assert(a_strides[i] == full_stride);
        full_stride *= batchdims[i];
      }
    }
#endif

    GeevImplBody<KIND, CODE>{context}(m, batchsize_total, m, m * m, a_acc, ew_acc, ev_acc);
  }

  template <Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!support_geev<CODE>::value || DIM<2>* = nullptr> void
            operator()(TaskContext& context) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void geev_template(TaskContext& context)
{
  auto a_array = context.input(0);
  double_dispatch(a_array.dim(), a_array.type().code(), GeevImpl<KIND>{context}, context);
}

}  // namespace cupynumeric
