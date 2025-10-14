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

#include <thrust/execution_policy.h>

#include <cupynumeric/stat/detail/histogramdd_using_thrust.h>
#include <cupynumeric/stat/histogramdd.h>
#include <cupynumeric/utilities/thrust_allocator.h>

namespace cupynumeric {
using namespace legate;

void HistogramDDTask::cpu_variant(TaskContext context)
{
  auto memkind = Memory::Kind::SYSTEM_MEM;
  auto alloc   = ThrustAllocator(memkind);
  auto policy  = thrust::host(alloc);

  detail::histogramdd_using_thrust(context, policy, memkind);
}

/* static */ std::optional<std::size_t> HistogramDDTask::allocation_pool_size(
  const mapping::Task& task, mapping::StoreTarget memory_kind)
{
  return detail::histogramdd_using_thrust_allocation_pool_size(task, memory_kind);
}

/* static */ std::vector<mapping::StoreMapping> HistogramDDTask::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  std::vector<mapping::StoreMapping> mappings;
  auto coords      = task.input(0);
  auto has_weights = task.scalar(0).value<bool>();

  mappings.push_back(mapping::StoreMapping::default_mapping(coords.data(), options.front()));
  mappings.back().policy().ordering.set_fortran_order();
  mappings.back().policy().exact = true;
  if (has_weights) {
    auto weights = task.input(1);
    mappings.push_back(mapping::StoreMapping::default_mapping(weights.data(), options.front()));
    mappings.back().policy().ordering.set_fortran_order();
    mappings.back().policy().exact = true;
  }
  return std::move(mappings);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  HistogramDDTask::register_variants();
}
}  // namespace

}  // namespace cupynumeric
