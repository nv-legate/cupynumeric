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

#include "cupynumeric/index/repeat.h"
#include "cupynumeric/index/repeat_template.inl"
#include "cupynumeric/omp_help.h"

#include <omp.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct RepeatImplBody<VariantKind::OMP, CODE, DIM> {
  TaskContext context;
  explicit RepeatImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const int64_t repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    auto out_rect = out_array.shape<DIM>();
    auto out      = out_array.write_accessor<VAL, DIM>(out_rect);
    Pitches<DIM - 1> pitches;

    auto out_volume = pitches.flatten(out_rect);
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < out_volume; ++idx) {
      auto out_p = pitches.unflatten(idx, out_rect.lo);
      auto in_p  = out_p;
      in_p[axis] /= repeats;
      out[out_p] = in[in_p];
    }
  }

  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const AccessorRO<int64_t, DIM>& repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    int64_t axis_extent = in_rect.hi[axis] - in_rect.lo[axis] + 1;
    auto offsets        = create_buffer<int64_t>(axis_extent);

    const auto max_threads = omp_get_max_threads();
    ThreadLocalStorage<int64_t> local_sums(max_threads);
    for (auto idx = 0; idx < max_threads; ++idx) {
      local_sums[idx] = 0;
    }

#pragma omp parallel
    {
      const auto tid  = omp_get_thread_num();
      auto p          = in_rect.lo;
      int64_t axis_lo = p[axis];
#pragma omp for schedule(static) private(p)
      for (int64_t idx = 0; idx < axis_extent; ++idx) {
        p[axis]      = axis_lo + idx;
        auto val     = repeats[p];
        offsets[idx] = val;
        local_sums[tid] += val;
      }
    }

    auto p_offsets = offsets.ptr(0);
    thrust::exclusive_scan(thrust::omp::par, p_offsets, p_offsets + axis_extent, p_offsets);

    int64_t sum = 0;
    for (auto idx = 0; idx < max_threads; ++idx) {
      sum += local_sums[idx];
    }

    Point<DIM> extents = in_rect.hi - in_rect.lo + Point<DIM>::ONES();
    extents[axis]      = sum;

    auto out = out_array.create_output_buffer<VAL, DIM>(extents, true);

    Pitches<DIM - 1> in_pitches;
    auto in_volume = in_pitches.flatten(in_rect);

#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < in_volume; ++idx) {
      auto in_p  = in_pitches.unflatten(idx, in_rect.lo);
      auto out_p = in_p - in_rect.lo;

      int64_t off_start = offsets[in_p[axis] - in_rect.lo[axis]];
      int64_t off_end   = off_start + repeats[in_p];

      auto in_v = in[in_p];
      for (int64_t out_idx = off_start; out_idx < off_end; ++out_idx) {
        out_p[axis] = out_idx;
        out[out_p]  = in_v;
      }
    }
  }
};

/*static*/ void RepeatTask::omp_variant(TaskContext context)
{
  repeat_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
