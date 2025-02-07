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

#include "cupynumeric/cupynumeric_c.h"
#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/mapper.h"
#include "cupynumeric/runtime.h"
#include "cupynumeric/unary/unary_red_util.h"

using namespace legate;

namespace cupynumeric {

static const char* const cupynumeric_library_name = "cupynumeric";

/*static*/ TaskRegistrar& CuPyNumericRegistrar::get_registrar()
{
  static TaskRegistrar registrar;
  return registrar;
}

void unload_cudalibs() noexcept
{
  auto machine = legate::get_machine();

  auto num_gpus = machine.count(legate::mapping::TaskTarget::GPU);
  if (0 == num_gpus) {
    return;
  }

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(cupynumeric_library_name);

  // Issue an execution fence so all outstanding tasks are done before we start destroying handles
  runtime->issue_execution_fence();

  runtime->submit(
    runtime->create_task(library,
                         legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_UNLOAD_CUDALIBS},
                         legate::tuple<uint64_t>{num_gpus}));
}

void registration_callback()
{
  ResourceConfig config;
  config.max_tasks         = CUPYNUMERIC_MAX_TASKS;
  config.max_reduction_ops = CUPYNUMERIC_MAX_REDOPS;

  auto runtime           = legate::Runtime::get_runtime();
  constexpr auto options = legate::VariantOptions{}.with_has_allocations(false);
  auto library           = runtime->create_library(
    cupynumeric_library_name,
    config,
    std::make_unique<CuPyNumericMapper>(),
    {{LEGATE_CPU_VARIANT, options}, {LEGATE_GPU_VARIANT, options}, {LEGATE_OMP_VARIANT, options}});

  CuPyNumericRegistrar::get_registrar().register_all_tasks(library);
  CuPyNumericRuntime::initialize(runtime, library);

  legate::register_shutdown_callback(unload_cudalibs);
}

}  // namespace cupynumeric

extern "C" {

void cupynumeric_perform_registration(void) { cupynumeric::registration_callback(); }

bool cupynumeric_has_cusolvermp()
{
  return LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP);
}
}
