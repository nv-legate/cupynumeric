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
#include <legate/task/exception.h>
#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/matrix/potrf.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND>
struct CopyBlockImpl {
  TaskContext context;
  explicit CopyBlockImpl(TaskContext context) : context(context) {}

  void operator()(void* dst, const void* src, size_t n);
};

template <VariantKind KIND, Type::Code CODE>
struct BatchedTriluImplBody {
  TaskContext context;
  explicit BatchedTriluImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(
    VAL* array, int32_t m, int32_t n, bool lower, int32_t num_blocks, int64_t block_stride);
};

template <VariantKind KIND, Type::Code CODE>
struct PotrfImplBody {
  TaskContext context;
  explicit PotrfImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(
    VAL* array, int32_t m, int32_t n, bool lower, int32_t num_blocks, int64_t block_stride);
};

template <Type::Code CODE>
struct _cholesky_supported {
  static constexpr bool value = CODE == Type::Code::FLOAT64 || CODE == Type::Code::FLOAT32 ||
                                CODE == Type::Code::COMPLEX64 || CODE == Type::Code::COMPLEX128;
};

template <VariantKind KIND>
struct PotrfImpl {
  TaskContext context;
  bool lower;
  bool zeroout;
  explicit PotrfImpl(TaskContext context, bool lower, bool zeroout)
    : context(context), lower(lower), zeroout(zeroout)
  {
  }

  template <
    Type::Code CODE,
    int32_t DIM,
    std::enable_if_t<_cholesky_supported<CODE>::value && (DIM >= 2) && (DIM < 4)>* = nullptr>
  void operator()(Array& input_array, Array& output_array) const
  {
    using VAL = type_of<CODE>;

    auto shape = input_array.shape<DIM>();
    if (shape != output_array.shape<DIM>()) {
      throw legate::TaskException("Potrf is not supported when input/output shapes differ");
    }

    if (shape.empty()) {
      return;
    }

    auto num_rows = shape.hi[DIM - 2] - shape.lo[DIM - 2] + 1;
    auto num_cols = shape.hi[DIM - 1] - shape.lo[DIM - 1] + 1;

    assert(num_rows == num_cols);

    // Calculate number of blocks (1 for DIM==2, product of batch dimensions for DIM>2)
    int32_t num_blocks = 1;
    for (int32_t i = 0; i < (DIM - 2); ++i) {
      num_blocks *= (shape.hi[i] - shape.lo[i] + 1);
    }

    size_t in_strides[DIM];
    size_t out_strides[DIM];

    auto input  = input_array.read_accessor<VAL, DIM>(shape).ptr(shape, in_strides);
    auto output = output_array.write_accessor<VAL, DIM>(shape).ptr(shape, out_strides);

    if (num_blocks > 1) {
      // Check that last two dimensions are contiguous
      if (in_strides[DIM - 1] != num_rows || in_strides[DIM - 2] != 1) {
        throw legate::TaskException(
          "Bad input accessor in potrf, last two dimensions must be contiguous");
      }
      if (out_strides[DIM - 1] != num_rows || out_strides[DIM - 2] != 1) {
        throw legate::TaskException(
          "Bad output accessor in potrf, last two dimensions must be contiguous");
      }
    }

    auto lda          = num_rows;
    auto block_stride = num_rows * num_cols;  // == num_rows*num_rows

    // some OMP variants use CPU implementation
    constexpr VariantKind CPU_OR_GPU =
      (KIND == VariantKind::GPU) ? VariantKind::GPU : VariantKind::CPU;
    if (input != output) {
      CopyBlockImpl<CPU_OR_GPU>{context}(output, input, sizeof(VAL) * block_stride * num_blocks);
    }
    PotrfImplBody<CPU_OR_GPU, CODE>{context}(
      output, num_rows, lda, lower, num_blocks, block_stride);
    if (zeroout) {
      BatchedTriluImplBody<KIND, CODE>{context}(output, num_rows, lower, num_blocks, block_stride);
    }
  }

  template <
    Type::Code CODE,
    int32_t DIM,
    std::enable_if_t<!_cholesky_supported<CODE>::value || (DIM < 2) || (DIM >= 4)>* = nullptr>
  void operator()(Array& input_array, Array& output_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void potrf_task_context_dispatch(TaskContext& context)
{
  legate::PhysicalStore input  = context.input(0);
  legate::PhysicalStore output = context.output(0);
  if (input.type() != output.type()) {
    throw legate::TaskException("potrf is not yet supported when input/output types differ");
  }
  if (input.dim() != output.dim()) {
    throw legate::TaskException("input/output have different dims in potrf");
  }
  auto scalars = context.scalars();
  assert(scalars.size() == 2);
  bool lower   = scalars[0].value<bool>();
  bool zeroout = scalars[1].value<bool>();
  double_dispatch(
    input.dim(), input.type().code(), PotrfImpl<KIND>{context, lower, zeroout}, input, output);
}

}  // namespace cupynumeric
