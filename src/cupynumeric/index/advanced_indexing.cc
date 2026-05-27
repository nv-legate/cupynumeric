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

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int IN_DIM, int OUT_DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::CPU, CODE, IN_DIM, OUT_DIM, OUT_TYPE> {
  TaskContext context;
  explicit AdvancedIndexingImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  static constexpr auto KEY_DIM = IN_DIM - OUT_DIM + 1;

  template <typename OUT_T>
  void compute_output(Buffer<OUT_T, OUT_DIM>& out,
                      const AccessorRO<VAL, IN_DIM>& input,
                      const AccessorRO<bool, IN_DIM>& index,
                      const Pitches<IN_DIM - 1>& input_pitches,
                      const Rect<IN_DIM>& input_rect,
                      const size_t input_volume,
                      const size_t skip_size) const
  {
    size_t out_idx = 0;

    for (size_t idx = 0; idx < input_volume; ++idx) {
      auto p = input_pitches.unflatten(idx, input_rect.lo);

      if (!index[p]) {
        continue;
      }

      Point<OUT_DIM> out_p;

      out_p[0] = out_idx;
      for (int32_t i = 1; i < OUT_DIM; i++) {
        out_p[i] = p[KEY_DIM + i - 1];
      }
      fill_out(out[out_p], p, input[p]);

      // The logic below is based on the assumtion that
      // pitches enumerate points in C-order, but this might
      // change in the future
      // TODO: replace with the order-aware interator when available
      out_idx += (idx + 1) % skip_size == 0;
    }
  }

  void operator()(legate::PhysicalStore& out_arr,
                  const AccessorRO<VAL, IN_DIM>& input,
                  const AccessorRO<bool, IN_DIM>& index,
                  const Pitches<IN_DIM - 1>& input_pitches,
                  const Rect<IN_DIM>& input_rect,
                  const size_t input_volume,
                  const size_t skip_size) const
  {
    // calculate size of the key_dim-1 extend in output region
    size_t size = 0;

    for (size_t idx = 0; idx < input_volume; idx += skip_size) {
      size += (index[input_pitches.unflatten(idx, input_rect.lo)] == true);
    }

    // calculating the shape of the output region for this sub-task
    Point<OUT_DIM> extents;

    extents[0] = size;
    for (int32_t i = 1; i < OUT_DIM; i++) {
      size_t j = KEY_DIM + i - 1;

      extents[i] = (input_rect.hi[j] - input_rect.lo[j]) + 1;
    }

    auto out = out_arr.create_output_buffer<OUT_TYPE, OUT_DIM>(extents, true);

    if (size > 0) {
      compute_output(out, input, index, input_pitches, input_rect, input_volume, skip_size);
    }
  }
};

/*static*/ void AdvancedIndexingTask::cpu_variant(TaskContext context)
{
  advanced_indexing_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
const auto cupynumeric_reg_task_ = []() -> char {
  AdvancedIndexingTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
