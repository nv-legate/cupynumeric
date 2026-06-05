/*
 * Copyright 2026 NVIDIA Corporation
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
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "legate.h"
#include "cupynumeric/ndarray.h"

namespace cupynumeric::ufunc {

class NativeArrayHandle {
 public:
  NativeArrayHandle(NDArray array, bool writeable);
  static NativeArrayHandle allocate_result(std::vector<uint64_t> shape,
                                           const legate::Type& dtype,
                                           bool writeable = true);
  static NativeArrayHandle from_store_raw_handle(std::uintptr_t raw_handle, bool writeable);

  NativeArrayHandle(const NativeArrayHandle&)            = default;
  NativeArrayHandle& operator=(const NativeArrayHandle&) = default;

  NativeArrayHandle(NativeArrayHandle&&)            = default;
  NativeArrayHandle& operator=(NativeArrayHandle&&) = default;

  bool writeable() const noexcept;
  void check_writeable() const;
  int32_t dim() const;
  std::size_t size() const;
  std::vector<uint64_t> shape() const;
  legate::Type dtype() const;
  legate::LogicalStore store();
  void copy_store_to_raw_handle(std::uintptr_t raw_handle);

  NativeArrayHandle converted_to(const legate::Type& dtype) const;
  void convert_from(const NativeArrayHandle& input);
  void launch_unary_op(int32_t op_code,
                       const NativeArrayHandle& input,
                       const std::vector<legate::Scalar>& extra_args = {});
  void launch_binary_op(int32_t op_code,
                        const NativeArrayHandle& lhs,
                        const NativeArrayHandle& rhs);
  void launch_binary_reduction(int32_t op_code,
                               const NativeArrayHandle& lhs,
                               const NativeArrayHandle& rhs);
  void launch_unary_reduction(int32_t op_code, const NativeArrayHandle& input);

 private:
  NDArray array_;
  bool writeable_{true};
};

}  // namespace cupynumeric::ufunc
