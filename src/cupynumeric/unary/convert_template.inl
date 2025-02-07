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
#include "cupynumeric/unary/convert.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/unary/convert_util.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, ConvertCode NAN_OP, Type::Code DST_TYPE, Type::Code SRC_TYPE, int DIM>
struct ConvertImplBody;

template <VariantKind KIND, ConvertCode NAN_OP, Type::Code SRC_TYPE>
struct ConvertImpl {
  template <Type::Code DST_TYPE, int DIM, std::enable_if_t<SRC_TYPE != DST_TYPE>* = nullptr>
  void operator()(ConvertArgs& args) const
  {
    using OP  = ConvertOp<NAN_OP, DST_TYPE, SRC_TYPE>;
    using SRC = type_of<SRC_TYPE>;
    using DST = type_of<DST_TYPE>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) {
      return;
    }

    auto out = args.out.write_accessor<DST, DIM>(rect);
    auto in  = args.in.read_accessor<SRC, DIM>(rect);

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    ConvertImplBody<KIND, NAN_OP, DST_TYPE, SRC_TYPE, DIM>()(func, out, in, pitches, rect, dense);
  }

  template <Type::Code DST_TYPE, int DIM, std::enable_if_t<SRC_TYPE == DST_TYPE>* = nullptr>
  void operator()(ConvertArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND, Type::Code SRC_TYPE>
struct ConvertDispatch {
  template <ConvertCode NAN_OP,
            std::enable_if_t<(legate::is_floating_point<SRC_TYPE>::value ||
                              legate::is_complex<SRC_TYPE>::value) ||
                             NAN_OP == ConvertCode::NOOP>* = nullptr>
  void operator()(ConvertArgs& args) const
  {
    auto dim = std::max(1, args.out.dim());
    double_dispatch(dim, args.out.code(), ConvertImpl<KIND, NAN_OP, SRC_TYPE>{}, args);
  }

  template <ConvertCode NAN_OP,
            std::enable_if_t<!((legate::is_floating_point<SRC_TYPE>::value ||
                                legate::is_complex<SRC_TYPE>::value) ||
                               (NAN_OP == ConvertCode::NOOP))>* = nullptr>
  void operator()(ConvertArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct SourceTypeDispatch {
  template <Type::Code SRC_TYPE>
  void operator()(ConvertArgs& args) const
  {
    op_dispatch(args.nan_op, ConvertDispatch<KIND, SRC_TYPE>{}, args);
  }
};

template <VariantKind KIND>
static void convert_template(TaskContext& context)
{
  ConvertArgs args{context.output(0), context.input(0), context.scalar(0).value<ConvertCode>()};
  type_dispatch(args.in.code(), SourceTypeDispatch<KIND>{}, args);
}

}  // namespace cupynumeric
