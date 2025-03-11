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

#include "cupynumeric/search/nonzero.h"
#include "cupynumeric/search/nonzero_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int32_t DIM>
struct NonzeroImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(std::vector<Array>& outputs,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume)
  {
    int64_t size = 0;

    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      size += in[point] != VAL(0);
    }

    std::vector<Buffer<int64_t>> results;
    for (auto& output : outputs) {
      results.push_back(output.create_output_buffer<int64_t, 1>(Point<1>(size), true));
    }

    int64_t out_idx = 0;
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = pitches.unflatten(idx, rect.lo);
      if (in[point] == VAL(0)) {
        continue;
      }
      for (int32_t dim = 0; dim < DIM; ++dim) {
        results[dim][out_idx] = point[dim];
      }
      ++out_idx;
    }
    assert(size == out_idx);
  }
};

/*static*/ void NonzeroTask::cpu_variant(TaskContext context)
{
  nonzero_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  NonzeroTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
