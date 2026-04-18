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

#include "cupynumeric/all2all/all2all.h"

#include <cassert>

namespace cupynumeric {

using namespace legate;

/*static*/ void All2AllTask::cpu_variant(TaskContext context)
{
  LEGATE_ABORT("CPU all2all not implemented");
}

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
/*static*/ void All2AllTask::omp_variant(TaskContext context)
{
  LEGATE_ABORT("OMP all2all not implemented");
}
#endif

namespace {
static const auto cupynumeric_reg_task_ = []() -> char {
  All2AllTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
