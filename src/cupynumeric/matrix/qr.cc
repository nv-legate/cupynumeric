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

#include "cupynumeric/matrix/qr.h"
#include "cupynumeric/matrix/qr_template.inl"
#include "cupynumeric/matrix/qr_cpu.inl"

namespace cupynumeric {

using namespace legate;

/*static*/ const char* QrTask::ERROR_MESSAGE = "Factorization failed";

/*static*/ void QrTask::cpu_variant(TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  openblas_set_num_threads(1);  // make sure this isn't overzealous
#endif
  qr_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { QrTask::register_variants(); }
}  // namespace

}  // namespace cupynumeric
