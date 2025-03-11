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

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
#include "omp.h"
#endif
#include "cblas.h"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct TransposeImplBody<VariantKind::CPU, CODE> {
  using VAL = type_of<CODE>;

  void operator()(const Rect<2>& rect,
                  const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in) const
  {
    constexpr coord_t BF = 128 / sizeof(VAL);
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

/*static*/ void TransposeTask::cpu_variant(TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  transpose_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
const auto reg = []() -> char {
  TransposeTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
