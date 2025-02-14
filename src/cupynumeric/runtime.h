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

#pragma once

#include <memory>

#include "legate.h"

#include "cupynumeric/cupynumeric_c.h"
#include "cupynumeric/typedefs.h"

namespace cupynumeric {

class NDArray;

class CuPyNumericRuntime {
 private:
  CuPyNumericRuntime(legate::Runtime* legate_runtime, legate::Library library);

 public:
  NDArray create_array(const legate::Type& type);
  NDArray create_array(std::vector<uint64_t> shape,
                       const legate::Type& type,
                       bool optimize_scalar = true);
  NDArray create_array(legate::LogicalStore&& store);
  NDArray create_array(const legate::Type& type, int32_t dim);
  legate::LogicalStore create_scalar_store(const Scalar& value);

 public:
  legate::Type get_argred_type(const legate::Type& value_type);

 public:
  legate::AutoTask create_task(CuPyNumericOpCode op_code);
  legate::ManualTask create_task(CuPyNumericOpCode op_code,
                                 const legate::tuple<std::uint64_t>& launch_shape);
  void submit(legate::AutoTask&& task);
  void submit(legate::ManualTask&& task);

 public:
  uint32_t get_next_random_epoch();

 public:
  legate::Library get_library() const { return library_; }

 public:
  static CuPyNumericRuntime* get_runtime();
  static void initialize(legate::Runtime* legate_runtime, legate::Library library);

 private:
  static CuPyNumericRuntime* runtime_;

 private:
  legate::Runtime* legate_runtime_;
  legate::Library library_;
  uint32_t next_epoch_{0};
  std::unordered_map<legate::Type::Code, legate::Type> argred_types_;
};

[[nodiscard]] bool is_in_test_mode();

}  // namespace cupynumeric
