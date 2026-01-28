/* Copyright 2025 NVIDIA Corporation
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
#include "cupynumeric/matrix/potrs.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct PotrsImplBody {
  TaskContext context;
  explicit PotrsImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const VAL* a,
                  VAL* x,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t x_block_stride,
                  bool lower);
};

template <Type::Code CODE>
struct support_potrs : std::false_type {};
template <>
struct support_potrs<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_potrs<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_potrs<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_potrs<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct PotrsImpl {
  TaskContext context;
  bool lower;

  explicit PotrsImpl(TaskContext context, bool lower) : context(context), lower(lower) {}

  template <Type::Code CODE, std::enable_if_t<support_potrs<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a_array, legate::PhysicalStore x_array) const
  {
    using VAL         = type_of<CODE>;
    constexpr int DIM = 3;
    auto a_shape      = a_array.shape<DIM>();
    auto x_shape      = x_array.shape<DIM>();

    if (a_shape.empty() || x_shape.empty()) {
      return;
    }

    size_t a_strides[DIM];
    size_t x_strides[DIM];

    auto a = a_array.read_accessor<VAL, DIM>(a_shape).ptr(a_shape, a_strides);
    auto x = x_array.write_accessor<VAL, DIM>(x_shape).ptr(x_shape, x_strides);

    // Calculate number of blocks (1 for DIM==2, product of batch dimensions for DIM>2)
    int32_t num_blocks = 1;
    for (int32_t i = 0; i < (DIM - 2); ++i) {
      num_blocks *= (a_shape.hi[i] - a_shape.lo[i] + 1);
    }

    auto m = static_cast<int32_t>(x_shape.hi[DIM - 2] - x_shape.lo[DIM - 2] + 1);
    auto n = static_cast<int32_t>(x_shape.hi[DIM - 1] - x_shape.lo[DIM - 1] + 1);
    assert(m > 0 && n > 0);

    auto nrows_a = static_cast<int32_t>(a_shape.hi[DIM - 2] - a_shape.lo[DIM - 2] + 1);
    auto ncols_a = static_cast<int32_t>(a_shape.hi[DIM - 1] - a_shape.lo[DIM - 1] + 1);
    assert(nrows_a == m);
    assert(ncols_a == m);

    if (num_blocks > 1) {
      // Check that last two dimensions are contiguous
      if ((ncols_a > 1 && (a_strides[DIM - 1] != nrows_a)) || a_strides[DIM - 2] != 1) {
        throw legate::TaskException(
          "Bad a accessor in potrs, last two dimensions must be contiguous");
      }
      if ((n > 1 && (x_strides[DIM - 1] != m)) || x_strides[DIM - 2] != 1) {
        throw legate::TaskException(
          "Bad x accessor in potrs, last two dimensions must be contiguous");
      }
    }

    auto x_block_stride = m * n;
    auto a_block_stride = m * m;

    // OMP variants use CPU implementation (with multiple threads)
    constexpr VariantKind CPU_OR_GPU =
      (KIND == VariantKind::GPU) ? VariantKind::GPU : VariantKind::CPU;
    PotrsImplBody<CPU_OR_GPU, CODE>{context}(
      a, x, m, n, num_blocks, a_block_stride, x_block_stride, lower);
  }

  template <Type::Code CODE, std::enable_if_t<!support_potrs<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore a, legate::PhysicalStore x) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void potrs_template(TaskContext& context)
{
  auto x_array = context.output(0);
  auto a_array = context.input(0);

  // Fetch scalar argument: lower (bool)
  auto scalars = context.scalars();
  assert(scalars.size() == 1);
  bool lower = scalars[0].value<bool>();

  type_dispatch(a_array.type().code(), PotrsImpl<KIND>{context, lower}, a_array, x_array);
}

}  // namespace cupynumeric
