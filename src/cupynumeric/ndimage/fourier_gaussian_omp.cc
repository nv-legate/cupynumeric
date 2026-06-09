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

#include "cupynumeric/ndimage/fourier_gaussian.h"
#include "cupynumeric/ndimage/fourier_gaussian_template.inl"

#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int DIM>
struct NdimageFourierGaussianImplBody<VariantKind::OMP, VAL, DIM> {
  TaskContext context;

  explicit NdimageFourierGaussianImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> output,
                  AccessorRO<VAL, DIM> input,
                  const Rect<DIM>& rect,
                  const NdimageFourierGaussianParams params) const
  {
    Pitches<DIM - 1> pitches;
    const size_t volume = pitches.flatten(rect);

#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < volume; ++idx) {
      const Point<DIM> p  = pitches.unflatten(idx, rect.lo);
      const double factor = fourier_gaussian_factor<DIM>(p, rect, params);
      output[p]           = input[p] * VAL{factor};
    }
  }
};

/*static*/ void NdimageFourierGaussianTask::omp_variant(TaskContext context)
{
  ndimage_fourier_gaussian_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
