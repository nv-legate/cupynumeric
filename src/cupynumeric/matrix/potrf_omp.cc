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

#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/matrix/potrf.h"
#include "cupynumeric/matrix/potrf_template.inl"

#include "cupynumeric/utilities/blas_lapack.h"
#include <omp.h>

namespace cupynumeric {

using namespace legate;

// BatchedTriluImplBody for OMP - zeros out upper or lower triangle with OpenMP parallelization
template <Type::Code CODE>
struct BatchedTriluImplBody<VariantKind::OMP, CODE> {
  TaskContext context;
  explicit BatchedTriluImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(VAL* array, int32_t n, bool lower, int32_t num_blocks, int64_t block_stride) const
  {
    // Parallelize across batches
#pragma omp parallel for schedule(static)
    for (int32_t batch = 0; batch < num_blocks; ++batch) {
      VAL* block_ptr = array + batch * block_stride;

      // Column-major layout: element at (row, col) is at block_ptr[col * n + row]
      for (int32_t col = 0; col < n; ++col) {
        if (lower) {
          // Zero out upper triangle (elements where row < col)
          for (int32_t row = 0; row < col; ++row) {
            block_ptr[col * n + row] = VAL(0);
          }
        } else {
          // Zero out lower triangle (elements where row > col)
          for (int32_t row = col + 1; row < n; ++row) {
            block_ptr[col * n + row] = VAL(0);
          }
        }
      }
    }
  }
};

/*static*/ void PotrfTask::omp_variant(TaskContext context)
{
  blas_set_num_threads(omp_get_max_threads());
  potrf_task_context_dispatch<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
