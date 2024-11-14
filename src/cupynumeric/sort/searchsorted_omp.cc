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

#include "cupynumeric/sort/searchsorted.h"
#include "cupynumeric/sort/searchsorted_template.inl"

#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int32_t DIM>
struct SearchSortedImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(const PhysicalStore& input_array,
                  const PhysicalStore& input_values,
                  const PhysicalStore& output_positions,
                  const Rect<1>& rect_base,
                  const Rect<DIM>& rect_values,
                  const Pitches<DIM - 1> pitches,
                  const bool left,
                  const bool is_index_space,
                  const size_t volume,
                  const int64_t global_volume,
                  const size_t num_values)
  {
    auto input   = input_array.read_accessor<VAL, 1>(rect_base);
    auto input_v = input_values.read_accessor<VAL, DIM>(rect_values);
    assert(input.accessor.is_dense_arbitrary(rect_base));

    auto* input_ptr   = input.ptr(rect_base.lo);
    auto* input_v_ptr = input_v.ptr(rect_values.lo);

    int64_t offset = rect_base.lo[0];

    if (left) {
      auto output_reduction =
        output_positions.reduce_accessor<MinReduction<int64_t>, true, DIM>(rect_values);
#pragma omp for
      for (size_t idx = 0; idx < num_values; ++idx) {
        VAL key             = input_v_ptr[idx];
        auto v_point        = pitches.unflatten(idx, rect_values.lo);
        int64_t lower_bound = std::lower_bound(input_ptr, input_ptr + volume, key) - input_ptr;
        if (lower_bound < static_cast<int64_t>(volume)) {
          output_reduction.reduce(v_point, lower_bound + offset);
        }
      }
    } else {
      auto output_reduction =
        output_positions.reduce_accessor<MaxReduction<int64_t>, true, DIM>(rect_values);
#pragma omp for
      for (size_t idx = 0; idx < num_values; ++idx) {
        VAL key             = input_v_ptr[idx];
        auto v_point        = pitches.unflatten(idx, rect_values.lo);
        int64_t upper_bound = std::upper_bound(input_ptr, input_ptr + volume, key) - input_ptr;
        if (upper_bound > 0) {
          output_reduction.reduce(v_point, upper_bound + offset);
        }
      }
    }
  }
};

/*static*/ void SearchSortedTask::omp_variant(TaskContext context)
{
  searchsorted_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
