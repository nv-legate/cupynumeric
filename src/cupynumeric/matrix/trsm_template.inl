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
struct TrsmImplBody {
  TaskContext context;
  explicit TrsmImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const VAL* a,
                  const VAL* b,
                  VAL* x,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t b_block_stride,
                  bool side,
                  bool lower,
                  int32_t transa,
                  bool unit_diagonal);
};

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
  TaskContext context;
  bool side_left;
  bool lower;
  int32_t transa;
  bool unit_diagonal;

  explicit TrsmImpl(
    TaskContext context, bool side_left, bool lower, int32_t transa, bool unit_diagonal)
    : context(context),
      side_left(side_left),
      lower(lower),
      transa(transa),
      unit_diagonal(unit_diagonal)
  {
  }

  template <Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<support_trsm<CODE>::value && (DIM >= 2) && (DIM < 4)>* = nullptr>
  void operator()(legate::PhysicalStore a_array,
                  legate::PhysicalStore b_array,
                  legate::PhysicalStore x_array) const
  {
    using VAL = type_of<CODE>;

    auto a_shape = a_array.shape<DIM>();
    auto b_shape = b_array.shape<DIM>();
    auto x_shape = x_array.shape<DIM>();

    assert(x_shape == b_shape);

    if (a_shape.empty() || b_shape.empty() || x_shape.empty()) {
      return;
    }

    size_t a_strides[DIM];
    size_t b_strides[DIM];
    size_t x_strides[DIM];

    auto a = a_array.read_accessor<VAL, DIM>(a_shape).ptr(a_shape, a_strides);
    auto b = b_array.read_accessor<VAL, DIM>(b_shape).ptr(b_shape, b_strides);
    auto x = x_array.write_accessor<VAL, DIM>(x_shape).ptr(x_shape, x_strides);

    // Calculate number of blocks (1 for DIM==2, product of batch dimensions for DIM>2)
    int32_t num_blocks = 1;
    for (int32_t i = 0; i < (DIM - 2); ++i) {
      num_blocks *= (a_shape.hi[i] - a_shape.lo[i] + 1);
    }

    auto m = static_cast<int32_t>(b_shape.hi[DIM - 2] - b_shape.lo[DIM - 2] + 1);
    auto n = static_cast<int32_t>(b_shape.hi[DIM - 1] - b_shape.lo[DIM - 1] + 1);
    assert(m > 0 && n > 0);

    auto nrows_a = static_cast<int32_t>(a_shape.hi[DIM - 2] - a_shape.lo[DIM - 2] + 1);
    auto ncols_a = static_cast<int32_t>(a_shape.hi[DIM - 1] - a_shape.lo[DIM - 1] + 1);
    assert(nrows_a == side_left ? m : n);
    assert(ncols_a == nrows_a);

    if (num_blocks > 1) {
      // Check that last two dimensions are contiguous
      if ((ncols_a > 1 && (a_strides[DIM - 1] != nrows_a)) || a_strides[DIM - 2] != 1) {
        throw legate::TaskException(
          "Bad a accessor in trsm, last two dimensions must be contiguous");
      }
      if ((n > 1 && (b_strides[DIM - 1] != m) || b_strides[DIM - 2] != 1)) {
        throw legate::TaskException(
          "Bad b accessor in trsm, last two dimensions must be contiguous");
      }
      if ((n > 1 && (x_strides[DIM - 1] != m || x_strides[DIM - 2] != 1))) {
        throw legate::TaskException(
          "Bad x accessor in trsm, last two dimensions must be contiguous");
      }
    }

    auto b_block_stride = m * n;
    auto a_block_stride = side_left ? m * m : n * n;

    // OMP variants use CPU implementation (with multiple threads)
    constexpr VariantKind CPU_OR_GPU =
      (KIND == VariantKind::GPU) ? VariantKind::GPU : VariantKind::CPU;
    TrsmImplBody<CPU_OR_GPU, CODE>{context}(a,
                                            b,
                                            x,
                                            m,
                                            n,
                                            num_blocks,
                                            a_block_stride,
                                            b_block_stride,
                                            side_left,
                                            lower,
                                            transa,
                                            unit_diagonal);
  }

  template <Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!support_trsm<CODE>::value || (DIM < 2) || (DIM >= 4)>* = nullptr>
  void operator()(legate::PhysicalStore a, legate::PhysicalStore b, legate::PhysicalStore x) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void trsm_template(TaskContext& context)
{
  auto x_array = context.output(0);
  auto a_array = context.input(0);
  auto b_array = context.input(1);

  // Fetch scalar arguments: side_left (bool), lower (bool), transa (int32), unit_diagonal (bool)
  auto scalars = context.scalars();
  assert(scalars.size() == 4);
  bool side_left     = scalars[0].value<bool>();
  bool lower         = scalars[1].value<bool>();
  int32_t transa     = scalars[2].value<int32_t>();
  bool unit_diagonal = scalars[3].value<bool>();

  double_dispatch(a_array.dim(),
                  a_array.type().code(),
                  TrsmImpl<KIND>{context, side_left, lower, transa, unit_diagonal},
                  a_array,
                  b_array,
                  x_array);
}

}  // namespace cupynumeric
