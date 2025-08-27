/* Copyright 2025 NVIDIA Corporation
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

#include "cupynumeric/index/take.h"
#include "cupynumeric/index/take_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

void TakeTask::gpu_variant(TaskContext context)
{
  take_template(context, DEFAULT_POLICY.on(context.get_task_stream()));
}

}  // namespace cupynumeric
