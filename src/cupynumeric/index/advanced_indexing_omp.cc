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
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::OMP, CODE, DIM, OUT_TYPE> {
  using VAL = type_of<CODE>;

  size_t compute_output_offsets(ThreadLocalStorage<int64_t>& offsets,
                                const AccessorRO<bool, DIM>& index,
                                const Pitches<DIM - 1>& pitches,
                                const Rect<DIM>& rect,
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
                  const AccessorRO<VAL, DIM>& input,
                  const AccessorRO<bool, DIM>& index,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t key_dim) const
  {
    size_t skip_size = 1;
    for (size_t i = key_dim; i < DIM; i++) {
      auto diff = 1 + rect.hi[i] - rect.lo[i];
      if (diff != 0) {
        skip_size *= diff;
      }
    }

    const auto max_threads = omp_get_max_threads();
    const size_t volume    = rect.volume();
    ThreadLocalStorage<int64_t> offsets(max_threads);
    size_t size =
      compute_output_offsets(offsets, index, pitches, rect, volume, skip_size, max_threads);

    // calculating the shape of the output region for this sub-task
    Point<DIM> extents;
    extents[0] = size;
    for (size_t i = 0; i < DIM - key_dim; i++) {
      size_t j       = key_dim + i;
      extents[i + 1] = 1 + rect.hi[j] - rect.lo[j];
    }
    for (size_t i = DIM - key_dim + 1; i < DIM; i++) {
      extents[i] = 1;
    }

    auto out = out_arr.create_output_buffer<OUT_TYPE, DIM>(extents, true);
    if (size > 0)
#pragma omp parallel
    {
      const int tid   = omp_get_thread_num();
      int64_t out_idx = offsets[tid];
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        if (index[p] == true) {
          Point<DIM> out_p;
          out_p[0] = out_idx;
          for (size_t i = 0; i < DIM - key_dim; i++) {
            size_t j     = key_dim + i;
            out_p[i + 1] = p[j];
          }
          for (size_t i = DIM - key_dim + 1; i < DIM; i++) {
            out_p[i] = 0;
          }
          fill_out(out[out_p], p, input[p]);
          if ((idx + 1) % skip_size == 0) {
            out_idx++;
          }
        }
      }

    }  // end parallel region
  }
};

/*static*/ void AdvancedIndexingTask::omp_variant(TaskContext context)
{
  advanced_indexing_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
