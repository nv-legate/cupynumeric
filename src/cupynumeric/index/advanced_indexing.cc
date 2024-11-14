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

template <Type::Code CODE, int DIM, typename OUT_TYPE>
struct AdvancedIndexingImplBody<VariantKind::CPU, CODE, DIM, OUT_TYPE> {
  using VAL = type_of<CODE>;

  template <typename OUT_T>
  void compute_output(Buffer<OUT_T, DIM>& out,
                      const AccessorRO<VAL, DIM>& input,
                      const AccessorRO<bool, DIM>& index,
                      const Pitches<DIM - 1>& pitches,
                      const Rect<DIM>& rect,
                      const size_t volume,
                      const size_t key_dim,
                      const size_t skip_size) const
  {
    size_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p = pitches.unflatten(idx, rect.lo);
      if (index[p] == true) {
        Point<DIM> out_p;
        out_p[0] = out_idx;
        for (int32_t i = 0; i < static_cast<int32_t>(DIM - key_dim); i++) {
          size_t j     = key_dim + i;
          out_p[i + 1] = p[j];
        }
        for (int32_t i = DIM - key_dim + 1; i < DIM; i++) {
          out_p[i] = 0;
        }
        fill_out(out[out_p], p, input[p]);
        // The logic below is based on the assumtion that
        // pitches enumerate points in C-order, but this might
        // change in the future
        // TODO: replace with the order-aware interator when available
        if ((idx + 1) % skip_size == 0) {
          out_idx++;
        }
      }
    }
  }

  void operator()(legate::PhysicalStore& out_arr,
                  const AccessorRO<VAL, DIM>& input,
                  const AccessorRO<bool, DIM>& index,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t key_dim) const
  {
    // skip_size is number of elements per each out[key_dim-1] sub-array
    size_t skip_size = 1;
    for (size_t i = key_dim; i < DIM; i++) {
      auto diff = 1 + rect.hi[i] - rect.lo[i];
      if (diff != 0) {
        skip_size *= diff;
      }
    }

    // calculate size of the key_dim-1 extend in output region
    const size_t volume = rect.volume();
    size_t size         = 0;
    for (size_t idx = 0; idx < volume; idx += skip_size) {
      auto p = pitches.unflatten(idx, rect.lo);
      if (index[p] == true) {
        size++;
      }
    }

    // calculating the shape of the output region for this sub-task
    Point<DIM> extents;
    extents[0] = size;
    for (int32_t i = 0; i < static_cast<int32_t>(DIM - key_dim); i++) {
      size_t j       = key_dim + i;
      extents[i + 1] = 1 + rect.hi[j] - rect.lo[j];
    }
    for (int32_t i = DIM - key_dim + 1; i < DIM; i++) {
      extents[i] = 1;
    }

    auto out = out_arr.create_output_buffer<OUT_TYPE, DIM>(extents, true);
    if (size > 0) {
      compute_output(out, input, index, pitches, rect, volume, key_dim, skip_size);
    }
  }
};

/*static*/ void AdvancedIndexingTask::cpu_variant(TaskContext context)
{
  advanced_indexing_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  AdvancedIndexingTask::register_variants();
}
}  // namespace

}  // namespace cupynumeric
