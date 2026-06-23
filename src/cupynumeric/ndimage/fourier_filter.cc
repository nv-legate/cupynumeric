/* Copyright 2026 NVIDIA Corporation
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

#include "cupynumeric/ndimage/fourier_filter.h"
#include "cupynumeric/ndimage/fourier_filter_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int DIM>
struct NdimageFourierFilterImplBody<VariantKind::CPU, VAL, DIM> {
  TaskContext context;

  explicit NdimageFourierFilterImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> output,
                  AccessorRO<VAL, DIM> input,
                  const Rect<DIM>& rect,
                  const NdimageFourierFilterParams params) const
  {
    for (PointInRectIterator<DIM> it(rect); it.valid(); ++it) {
      const Point<DIM> p  = *it;
      const double factor = fourier_filter_factor<DIM>(p, rect, params);
      output[p]           = input[p] * static_cast<VAL>(factor);
    }
  }
};

/*static*/ void NdimageFourierFilterTask::cpu_variant(TaskContext context)
{
  ndimage_fourier_filter_template<VariantKind::CPU>(context);
}

namespace {
static const auto cupynumeric_reg_task_ = []() -> char {
  NdimageFourierFilterTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
