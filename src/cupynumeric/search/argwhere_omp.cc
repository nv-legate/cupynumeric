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

#include "cupynumeric/search/argwhere.h"
#include "cupynumeric/search/argwhere_template.inl"
#include "cupynumeric/omp_help.h"

#include <omp.h>
namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct ArgWhereImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(legate::PhysicalStore& out_array,
                  AccessorRO<VAL, DIM> input,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  size_t volume) const
  {
    const auto max_threads = omp_get_max_threads();

    int64_t size = 0;
    ThreadLocalStorage<int64_t> offsets(max_threads);
    {
      ThreadLocalStorage<int64_t> sizes(max_threads);
      for (auto idx = 0; idx < max_threads; ++idx) {
        sizes[idx] = 0;
      }
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (size_t idx = 0; idx < volume; ++idx) {
          auto in_p = pitches.unflatten(idx, rect.lo);
          sizes[tid] += input[in_p] != VAL(0);
        }
      }
      for (auto idx = 0; idx < max_threads; ++idx) {
        size += sizes[idx];
      }
      offsets[0] = 0;
      for (auto idx = 1; idx < max_threads; ++idx) {
        offsets[idx] = offsets[idx - 1] + sizes[idx - 1];
      }
    }

    auto out = out_array.create_output_buffer<int64_t, 2>(Point<2>(size, DIM), true);

#pragma omp parallel
    {
      const int tid   = omp_get_thread_num();
      int64_t out_idx = offsets[tid];
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto in_p = pitches.unflatten(idx, rect.lo);

        if (input[in_p] != VAL(0)) {
          for (int i = 0; i < DIM; ++i) {
            out[Point<2>(out_idx, i)] = in_p[i];
          }
          out_idx++;
        }
      }
      assert(out_idx == (tid == max_threads - 1 ? size : offsets[tid + 1]));
    }
  }
};

/*static*/ void ArgWhereTask::omp_variant(TaskContext context)
{
  argwhere_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
