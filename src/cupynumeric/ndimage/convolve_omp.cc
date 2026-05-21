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

#include "cupynumeric/ndimage/convolve.h"
#include "cupynumeric/ndimage/convolve_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int DIM, bool BATCHED>
struct NdimageConvolveImplBody<VariantKind::OMP, VAL, DIM, BATCHED> {
  static_assert(!BATCHED || DIM > 2, "ndimage.batched_convolve requires DIM > 2");

  TaskContext context;
  explicit NdimageConvolveImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM>,
                  AccessorRO<VAL, DIM>,
                  AccessorRO<VAL, DIM>,
                  const Rect<DIM>&,
                  const Rect<DIM>&,
                  const Rect<DIM>&,
                  CuPyNumericNdimageConvolveMode,
                  VAL,
                  Point<DIM>) const
  {
    throw legate::TaskException(NdimageConvolveTask::ERROR_MESSAGE);
  }
};

/*static*/ void NdimageConvolveTask::omp_variant(TaskContext context)
{
  ndimage_convolve_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
