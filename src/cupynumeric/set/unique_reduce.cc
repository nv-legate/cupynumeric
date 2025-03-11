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

#include "cupynumeric/set/unique_reduce.h"
#include "cupynumeric/set/unique_reduce_template.inl"

namespace cupynumeric {

/*static*/ void UniqueReduceTask::cpu_variant(TaskContext context)
{
  unique_reduce_template(context, thrust::host);
}

namespace  // unnamed
{
const auto reg_ = []() -> char {
  UniqueReduceTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
