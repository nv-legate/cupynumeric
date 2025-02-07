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

#include "cupynumeric/matrix/batched_cholesky.h"
#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/matrix/batched_cholesky_template.inl"

#include <cblas.h>
#include <legate/type/type_info.h>
#include <lapack.h>

namespace cupynumeric {

using namespace legate;

template <>
void CopyBlockImpl<VariantKind::CPU>::operator()(void* dst, const void* src, size_t size)
{
  ::memcpy(dst, src, size);
}

template <Type::Code CODE>
struct BatchedTransposeImplBody<VariantKind::CPU, CODE> {
  using VAL = type_of<CODE>;

  static constexpr int tile_size = 64;

  void operator()(VAL* out, int n) const
  {
    VAL tile[tile_size][tile_size];
    int nblocks = (n + tile_size - 1) / tile_size;

    for (int rb = 0; rb < nblocks; ++rb) {
      for (int cb = 0; cb < nblocks; ++cb) {
        int r_start = rb * tile_size;
        int r_stop  = std::min(r_start + tile_size, n);
        int c_start = cb * tile_size;
        int c_stop  = std::min(c_start + tile_size, n);
        for (int r = r_start, tr = 0; r < r_stop; ++r, ++tr) {
          for (int c = c_start, tc = 0; c < c_stop; ++c, ++tc) {
            if (r <= c) {
              tile[tr][tc] = out[r * n + c];
            } else {
              tile[tr][tc] = 0;
            }
          }
        }
        for (int r = c_start, tr = 0; r < c_stop; ++r, ++tr) {
          for (int c = r_start, tc = 0; c < r_stop; ++c, ++tc) {
            out[r * n + c] = tile[tc][tr];
          }
        }
      }
    }
  }
};

/*static*/ void BatchedCholeskyTask::cpu_variant(TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  batched_cholesky_task_context_dispatch<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  BatchedCholeskyTask::register_variants();
}
}  // namespace

}  // namespace cupynumeric
