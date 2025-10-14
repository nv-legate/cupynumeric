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

#include <cupynumeric/stat/detail/histogramdd_using_thrust.h>
#include <cupynumeric/stat/histogramdd.h>
#include <cupynumeric/utilities/thrust_util.h>
#include <cupynumeric/utilities/thrust_allocator.h>

namespace cupynumeric {
using namespace legate;

void HistogramDDTask::gpu_variant(TaskContext context)
{
  auto memkind = Memory::Kind::GPU_FB_MEM;
  auto alloc   = ThrustAllocator(memkind);
  auto policy  = DEFAULT_POLICY(alloc).on(context.get_task_stream());

  detail::histogramdd_using_thrust(context, policy, memkind);
}

}  // namespace cupynumeric
