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

#include "cupynumeric/set/unique.h"
#include "cupynumeric/set/unique_template.inl"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int32_t DIM>
struct UniqueImplBody<VariantKind::OMP, CODE, DIM> {
  TaskContext context;
  explicit UniqueImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(legate::PhysicalStore& output,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const std::vector<comm::Communicator>& comms,
                  const DomainPoint& point,
                  const Domain& launch_domain)
  {
    const auto max_threads = omp_get_max_threads();
    std::vector<std::set<VAL>> dedup_set(max_threads);

#pragma omp parallel
    {
      const int tid      = omp_get_thread_num();
      auto& my_dedup_set = dedup_set[tid];
#pragma omp for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        my_dedup_set.insert(in[p]);
      }
    }

    size_t remaining = max_threads;
    size_t radix     = (max_threads + 1) / 2;
    while (remaining > 1) {
#pragma omp for schedule(static, 1)
      for (size_t idx = 0; idx < radix; ++idx) {
        if (idx + radix < remaining) {
          auto& my_set    = dedup_set[idx];
          auto& other_set = dedup_set[idx + radix];
          my_set.insert(other_set.begin(), other_set.end());
        }
      }
      remaining = radix;
      radix     = (radix + 1) / 2;
    }

    auto& final_dedup_set = dedup_set[0];
    auto result           = output.create_output_buffer<VAL, 1>(final_dedup_set.size(), true);
    size_t pos            = 0;
    for (auto e : final_dedup_set) {
      result[pos++] = e;
    }
  }
};

/*static*/ void UniqueTask::omp_variant(TaskContext context)
{
  unique_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
