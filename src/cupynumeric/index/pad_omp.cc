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

#include "cupynumeric/index/pad.h"
#include "cupynumeric/index/pad_template.inl"

#include <algorithm>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/omp/execution_policy.h>
#include "cupynumeric/omp_help.h"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct PadImplBody<VariantKind::OMP, CODE, DIM> {
  TaskContext context;
  explicit PadImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  // Overload for CONSTANT mode
  void operator()(AccessorRW<VAL, DIM>& output,
                  PadMode mode,
                  Span<const std::pair<int64_t, int64_t>> pad_width,
                  Span<const int64_t> inner_shape,
                  int64_t constant_rows,
                  int64_t constant_cols,
                  const AccessorRO<VAL, 1>& const_acc,
                  const Rect<DIM>& output_rect) const
  {
    (void)mode;
    Point<DIM> center_lo_global, center_hi_global;
    detail::compute_center_bounds<DIM>(pad_width, inner_shape, center_lo_global, center_hi_global);

    const int64_t rows = constant_rows;
    const int64_t cols = constant_cols;
    Pitches<DIM - 1> pitches;
    auto volume = pitches.flatten(output_rect);

    auto begin         = thrust::make_counting_iterator<size_t>(0);
    auto end           = begin + volume;
    const auto functor = detail::ConstantPadPointFunctor<DIM, VAL>(
      output, const_acc, center_lo_global, center_hi_global, rows, cols, pitches, output_rect.lo);
    thrust::for_each(thrust::omp::par, begin, end, functor);
  }

  // Overload for EDGE mode
  void operator()(AccessorRW<VAL, DIM>& output,
                  PadMode mode,
                  Span<const std::pair<int64_t, int64_t>> pad_width,
                  Span<const int64_t> inner_shape,
                  const Rect<DIM>& output_rect) const
  {
    // Calculate center region in GLOBAL coordinates
    Point<DIM> center_lo_global, center_hi_global;
    detail::compute_center_bounds<DIM>(pad_width, inner_shape, center_lo_global, center_hi_global);
    Point<DIM> center_lo_tile, center_hi_tile;
    const bool has_center = detail::intersect_center_with_tile<DIM>(
      center_lo_global, center_hi_global, output_rect, center_lo_tile, center_hi_tile);

    // EDGE mode - replicate edges from center
    Pitches<DIM - 1> pitches;
    auto volume = pitches.flatten(output_rect);

    auto begin         = thrust::make_counting_iterator<size_t>(0);
    auto end           = begin + volume;
    const auto functor = detail::EdgePadPointFunctor<DIM, VAL>(
      output, has_center, center_lo_tile, center_hi_tile, pitches, output_rect.lo);
    thrust::for_each(thrust::omp::par, begin, end, functor);
  }
};

/*static*/ void PadTask::omp_variant(TaskContext context)
{
  pad_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
