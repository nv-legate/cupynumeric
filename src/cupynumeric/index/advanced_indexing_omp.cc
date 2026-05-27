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

#include "cupynumeric/index/advanced_indexing.h"
#include "cupynumeric/index/advanced_indexing_template.inl"

#include "cupynumeric/omp_help.h"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int IN_DIM, int OUT_DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::OMP, CODE, IN_DIM, OUT_DIM, OUT_TYPE> {
  TaskContext context;
  explicit AdvancedIndexingImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  static constexpr auto KEY_DIM = IN_DIM - OUT_DIM + 1;

  size_t compute_output_offsets(ThreadLocalStorage<int64_t>& offsets,
                                const AccessorRO<bool, IN_DIM>& index,
                                const Pitches<IN_DIM - 1>& pitches,
                                const Rect<IN_DIM>& rect,
                                const size_t volume,
                                const size_t skip_size,
                                const size_t max_threads) const
  {
    ThreadLocalStorage<int64_t> sizes(max_threads);

    for (size_t idx = 0; idx < max_threads; ++idx) {
      sizes[idx] = 0;
    }

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);

        sizes[tid] += static_cast<int64_t>(index[p] && ((idx + 1) % skip_size == 0));
      }
    }  // end of parallel
    size_t size = 0;

    for (size_t idx = 0; idx < max_threads; ++idx) {
      offsets[idx] = size;
      size += sizes[idx];
    }

    return size;
  }

  void operator()(PhysicalStore& out_arr,
                  const AccessorRO<VAL, IN_DIM>& input,
                  const AccessorRO<bool, IN_DIM>& index,
                  const Pitches<IN_DIM - 1>& input_pitches,
                  const Rect<IN_DIM>& input_rect,
                  const size_t input_volume,
                  const size_t skip_size) const
  {
    const auto max_threads = omp_get_max_threads();
    ThreadLocalStorage<int64_t> offsets(max_threads);
    size_t size = compute_output_offsets(
      offsets, index, input_pitches, input_rect, input_volume, skip_size, max_threads);

    // calculating the shape of the output region for this sub-task
    Point<OUT_DIM> extents;

    extents[0] = size;
    for (int32_t i = 1; i < OUT_DIM; i++) {
      size_t j = KEY_DIM + i - 1;

      extents[i] = (input_rect.hi[j] - input_rect.lo[j]) + 1;
    }

    auto out = out_arr.create_output_buffer<OUT_TYPE, OUT_DIM>(extents, true);

    if (size > 0)
#pragma omp parallel
    {
      const int tid   = omp_get_thread_num();
      int64_t out_idx = offsets[tid];
#pragma omp for schedule(static)

      for (size_t idx = 0; idx < input_volume; ++idx) {
        auto p = input_pitches.unflatten(idx, input_rect.lo);

        if (!index[p]) {
          continue;
        }

        Point<OUT_DIM> out_p;

        out_p[0] = out_idx;
        for (int32_t i = 1; i < OUT_DIM; i++) {
          size_t j = KEY_DIM + i - 1;

          out_p[i] = p[j];
        }
        fill_out(out[out_p], p, input[p]);

        out_idx += (idx + 1) % skip_size == 0;
      }
    }  // end parallel region
  }
};

/*static*/ void AdvancedIndexingTask::omp_variant(TaskContext context)
{
  advanced_indexing_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
