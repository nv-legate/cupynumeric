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
#include "cupynumeric/index/wrap.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, int DIM>
struct WrapImplBody;

template <VariantKind KIND>
struct WrapImpl {
  template <int DIM>
  void operator()(WrapArgs& args) const
  {
    auto rect_out = args.out.shape<1>();  // output array is always 1D
    auto out      = args.out.write_accessor<Point<DIM>, 1>(rect_out);

    Pitches<0> pitches_out;
    size_t volume_out = pitches_out.flatten(rect_out);
    if (volume_out == 0) {
      return;
    }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    bool dense = out.accessor.is_dense_row_major(rect_out);
#else
    bool dense = false;
#endif

    Point<DIM> point_lo, point_hi;
    for (int dim = 0; dim < DIM; ++dim) {
      point_lo[dim] = 0;
      point_hi[dim] = args.shape[dim] - 1;
    }
    Rect<DIM> rect_base(point_lo, point_hi);

    Pitches<DIM - 1> pitches_base{};
    size_t volume_base = pitches_base.flatten(rect_base);
#ifdef DEBUG_CUPYNUMERIC
    assert(volume_base != 0);
#else
    static_cast<void>(volume_base);
#endif

    if (args.has_input) {
      auto rect_in = args.in.shape<1>();
      auto in = args.in.read_accessor<int64_t, 1>(rect_in);  // input should be always integer type
#ifdef DEBUG_CUPYNUMERIC
      assert(rect_in == rect_out);
#endif
      WrapImplBody<KIND, DIM>()(
        out, pitches_out, rect_out, pitches_base, rect_base, dense, args.check_bounds, in);

    } else {
      bool tmp = false;
      WrapImplBody<KIND, DIM>()(
        out, pitches_out, rect_out, pitches_base, rect_base, dense, args.check_bounds, tmp);
    }  // else
  }
};

template <VariantKind KIND>
static void wrap_template(TaskContext& context)
{
  auto shape        = context.scalar(0).value<DomainPoint>();
  int dim           = shape.dim;
  bool has_input    = context.scalar(1).value<bool>();
  bool check_bounds = context.scalar(2).value<bool>();
  legate::PhysicalStore tmp_array{nullptr};
  WrapArgs args{
    context.output(0), shape, has_input, check_bounds, has_input ? context.input(0) : tmp_array};
  dim_dispatch(dim, WrapImpl<KIND>{}, args);
}

}  // namespace cupynumeric
