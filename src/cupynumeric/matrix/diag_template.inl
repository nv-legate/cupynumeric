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
#include "cupynumeric/matrix/diag.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM, bool extract>
struct DiagImplBody;

template <VariantKind KIND>
struct DiagImpl {
  template <Type::Code CODE, int DIM>
  void operator()(DiagArgs& args) const
  {
    using VAL = type_of<CODE>;

    if (args.extract) {
      auto shape_in = args.matrix.shape<DIM>();
      Pitches<DIM - 1> pitches_in;
      size_t volume_in = pitches_in.flatten(shape_in);
      if (volume_in == 0) {
        return;
      }
      auto shape_out        = args.diag.shape<DIM>();
      size_t diag_start_dim = DIM - args.naxes;
      coord_t start         = shape_in.lo[diag_start_dim];
      coord_t end           = shape_in.hi[diag_start_dim];

      for (int i = diag_start_dim + 1; i < DIM; i++) {
        start = std::max(start, shape_in.lo[i]);
        end   = std::min(end, shape_in.hi[i]);
      }
      coord_t distance = end - start + 1;
      if (distance < 0) {
        return;
      }

      auto in  = args.matrix.read_accessor<VAL, DIM>(shape_in);
      auto out = args.diag.reduce_accessor<SumReduction<VAL>, true, DIM>(shape_out);

      DiagImplBody<KIND, CODE, DIM, true>()(
        out, in, start, pitches_in, shape_in, args.naxes, distance);

    } else {  // extract=False version: returning diagonal matrix from 1d array
      auto shape = args.matrix.shape<2>();

      // Solve for the start
      // y = x
      // x >= shape.lo[0]
      const Point<2> start1(shape.lo[0], shape.lo[0]);
      // y >= shape.lo[1]
      const Point<2> start2(shape.lo[1], shape.lo[1]);
      // See if our shape intersects with the diagonal
      if (!shape.contains(start1) && !shape.contains(start2)) {
        return;
      }

      // Pick whichever one fits in our rect
      const Point<2> start = shape.contains(start1) ? start1 : start2;
      // Now do the same thing for the end
      // x <= shape.hi[0]
      const Point<2> stop1(shape.hi[0], shape.hi[0]);
      // y <= shape.hi[1]
      const Point<2> stop2(shape.hi[1], shape.hi[1]);
#ifdef DEBUG_CUPYNUMERIC
      assert(shape.contains(stop1) || shape.contains(stop2));
#endif
      const Point<2> stop = shape.contains(stop1) ? stop1 : stop2;
      // Walk the path from the stop to the start
      const coord_t distance = (stop[0] - start[0]) + 1;
#ifdef DEBUG_CUPYNUMERIC
      // Should be the same along both dimensions
      assert(distance == ((stop[1] - start[1]) + 1));
      // no extract is supported only for 1d input array (2d output)
      assert(DIM == 2);
#endif
      auto shape_out = args.matrix.shape<2>();
      auto shape_in  = args.diag.shape<2>();

      auto in  = args.diag.read_accessor<VAL, 2>(shape_in);
      auto out = args.matrix.read_write_accessor<VAL, 2>(shape_out);
      DiagImplBody<KIND, CODE, 2, false>()(in, out, start, distance);
    }
  }
};

template <VariantKind KIND>
static void diag_template(TaskContext& context)
{
  int naxes                    = context.scalar(0).value<int>();
  bool extract                 = context.scalar(1).value<bool>();
  legate::PhysicalStore matrix = extract ? context.input(0) : context.output(0);
  legate::PhysicalStore diag   = extract ? context.reduction(0) : context.input(0);
  DiagArgs args{naxes, extract, matrix, diag};
  double_dispatch(matrix.dim(), matrix.code(), DiagImpl<KIND>{}, args);
}

}  // namespace cupynumeric
