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

#include "env_defaults.h"
#include "cupynumeric/runtime.h"

#include "cupynumeric/ndarray.h"
#include "cupynumeric/unary/unary_red_util.h"

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace cupynumeric {

/*static*/ CuPyNumericRuntime* CuPyNumericRuntime::runtime_;

extern void bootstrapping_callback(Legion::Machine machine,
                                   Legion::Runtime* runtime,
                                   const std::set<Legion::Processor>& local_procs);

void initialize(int32_t argc, char** argv) { cupynumeric_perform_registration(); }

CuPyNumericRuntime::CuPyNumericRuntime(legate::Runtime* legate_runtime, legate::Library library)
  : legate_runtime_(legate_runtime), library_(library)
{
}

NDArray CuPyNumericRuntime::create_array(const legate::Type& type)
{
  auto store = legate_runtime_->create_store(type);
  return NDArray(std::move(store));
}

NDArray CuPyNumericRuntime::create_array(std::vector<uint64_t> shape,
                                         const legate::Type& type,
                                         bool optimize_scalar)
{
  auto store = legate_runtime_->create_store(legate::Shape{shape}, type, optimize_scalar);
  return NDArray(std::move(store));
}

NDArray CuPyNumericRuntime::create_array(legate::LogicalStore&& store)
{
  return NDArray(std::move(store));
}

NDArray CuPyNumericRuntime::create_array(const legate::Type& type, int32_t dim)
{
  auto store = legate_runtime_->create_store(type, dim);
  return NDArray(std::move(store));
}

legate::LogicalStore CuPyNumericRuntime::create_scalar_store(const Scalar& value)
{
  return legate_runtime_->create_store(value);
}

legate::Type CuPyNumericRuntime::get_argred_type(const legate::Type& value_type)
{
  auto finder = argred_types_.find(value_type.code());
  if (finder != argred_types_.end()) {
    return finder->second;
  }

  auto argred_type = legate::struct_type({legate::int64(), value_type}, true /*align*/);
  argred_types_.insert({value_type.code(), argred_type});
  return argred_type;
}

legate::AutoTask CuPyNumericRuntime::create_task(CuPyNumericOpCode op_code)
{
  return legate_runtime_->create_task(library_, legate::LocalTaskID{op_code});
}

legate::ManualTask CuPyNumericRuntime::create_task(CuPyNumericOpCode op_code,
                                                   const legate::tuple<std::uint64_t>& launch_shape)
{
  return legate_runtime_->create_task(library_, legate::LocalTaskID{op_code}, launch_shape);
}

void CuPyNumericRuntime::submit(legate::AutoTask&& task)
{
  legate_runtime_->submit(std::move(task));
}

void CuPyNumericRuntime::submit(legate::ManualTask&& task)
{
  legate_runtime_->submit(std::move(task));
}

uint32_t CuPyNumericRuntime::get_next_random_epoch() { return next_epoch_++; }

/*static*/ CuPyNumericRuntime* CuPyNumericRuntime::get_runtime() { return runtime_; }

/*static*/ void CuPyNumericRuntime::initialize(legate::Runtime* legate_runtime,
                                               legate::Library library)
{
  runtime_ = new CuPyNumericRuntime(legate_runtime, library);
}

namespace {

std::uint32_t extract_env(const char* env_name,
                          std::uint32_t default_value,
                          std::uint32_t test_value)
{
  auto parse_value = [](const char* value_char) {
    auto value_sv = std::string_view{value_char};

    std::uint32_t result{};
    if (auto&& [_, ec] = std::from_chars(value_sv.begin(), value_sv.end(), result);
        ec != std::errc{}) {
      throw std::runtime_error{std::make_error_code(ec).message()};
    }

    return result;
  };

  if (const auto* env_value = std::getenv(env_name); env_value) {
    return parse_value(env_value);
  }

  if (const auto* is_in_test_mode = std::getenv("LEGATE_TEST");
      is_in_test_mode && parse_value(is_in_test_mode)) {
    return test_value;
  }

  return default_value;
}

}  // namespace

}  // namespace cupynumeric

extern "C" {

unsigned cupynumeric_max_eager_volume()
{
  static const auto min_gpu_chunk = cupynumeric::extract_env(
    "CUPYNUMERIC_MIN_GPU_CHUNK", MIN_GPU_CHUNK_DEFAULT, MIN_GPU_CHUNK_TEST);
  static const auto min_cpu_chunk = cupynumeric::extract_env(
    "CUPYNUMERIC_MIN_CPU_CHUNK", MIN_CPU_CHUNK_DEFAULT, MIN_CPU_CHUNK_TEST);
  static const auto min_omp_chunk = cupynumeric::extract_env(
    "CUPYNUMERIC_MIN_OMP_CHUNK", MIN_OMP_CHUNK_DEFAULT, MIN_OMP_CHUNK_TEST);

  auto machine = legate::get_machine();

  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    return min_gpu_chunk;
  }
  if (machine.count(legate::mapping::TaskTarget::OMP) > 0) {
    return min_omp_chunk;
  }
  return min_cpu_chunk;
}

unsigned cupynumeric_matmul_cache_size()
{
  static const auto max_cache_size = cupynumeric::extract_env(
    "CUPYNUMERIC_MATMUL_CACHE_SIZE", MATMUL_CACHE_SIZE_DEFAULT, MATMUL_CACHE_SIZE_TEST);
  return max_cache_size;
}

}  // extern "C"
