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

#include "cupynumeric/matrix/transpose.h"
#include "cupynumeric/matrix/transpose_template.inl"

#include "omp.h"
#include "cupynumeric/utilities/blas_lapack.h"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct TransposeImplBody<VariantKind::OMP, CODE> {
  TaskContext context;
  explicit TransposeImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const Rect<2>& rect,
                  const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in) const
  {
    constexpr coord_t BF = 128 / sizeof(VAL);
#pragma omp parallel for
    for (auto i1 = rect.lo[0]; i1 <= rect.hi[0]; i1 += BF) {
      for (auto j1 = rect.lo[1]; j1 <= rect.hi[1]; j1 += BF) {
        const auto max_i2 = ((i1 + BF) <= rect.hi[0]) ? i1 + BF : rect.hi[0];
        const auto max_j2 = ((j1 + BF) <= rect.hi[1]) ? j1 + BF : rect.hi[1];
        for (auto i2 = i1; i2 <= max_i2; i2++) {
          for (auto j2 = j1; j2 <= max_j2; j2++) {
            out[i2][j2] = in[i2][j2];
          }
        }
      }
    }
  }
};

/*static*/ void TransposeTask::omp_variant(TaskContext context)
{
  blas_set_num_threads(omp_get_max_threads());
  transpose_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
