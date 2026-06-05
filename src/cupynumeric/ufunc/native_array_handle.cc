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

#include "cupynumeric/ufunc/native_array_handle.h"

#include "cupynumeric/runtime.h"

#include <stdexcept>
#include <utility>

namespace cupynumeric::ufunc {

NativeArrayHandle::NativeArrayHandle(NDArray array, bool writeable)
  : array_{std::move(array)}, writeable_{writeable}
{
}

NativeArrayHandle NativeArrayHandle::allocate_result(std::vector<uint64_t> shape,
                                                     const legate::Type& dtype,
                                                     bool writeable)
{
  auto array = CuPyNumericRuntime::get_runtime()->create_array(std::move(shape), dtype);
  return NativeArrayHandle{std::move(array), writeable};
}

NativeArrayHandle NativeArrayHandle::from_store_raw_handle(std::uintptr_t raw_handle,
                                                           bool writeable)
{
  if (raw_handle == 0) {
    throw std::invalid_argument{"Cannot create NativeArrayHandle from a null LogicalStore handle"};
  }

  auto* store    = reinterpret_cast<legate::LogicalStore*>(raw_handle);
  auto store_cpy = legate::LogicalStore{*store};
  auto array     = CuPyNumericRuntime::get_runtime()->create_array(std::move(store_cpy));
  return NativeArrayHandle{std::move(array), writeable};
}

bool NativeArrayHandle::writeable() const noexcept { return writeable_; }

void NativeArrayHandle::check_writeable() const
{
  if (!writeable_) {
    throw std::invalid_argument{"Cannot write through a read-only NativeArrayHandle"};
  }
}

int32_t NativeArrayHandle::dim() const { return array_.dim(); }

std::size_t NativeArrayHandle::size() const { return array_.size(); }

std::vector<uint64_t> NativeArrayHandle::shape() const { return array_.shape(); }

legate::Type NativeArrayHandle::dtype() const { return array_.type(); }

legate::LogicalStore NativeArrayHandle::store() { return array_.get_store(); }

void NativeArrayHandle::copy_store_to_raw_handle(std::uintptr_t raw_handle)
{
  if (raw_handle == 0) {
    throw std::invalid_argument{
      "Cannot copy NativeArrayHandle store to a null LogicalStore handle"};
  }

  auto* store = reinterpret_cast<legate::LogicalStore*>(raw_handle);
  *store      = array_.get_store();
  store->allow_out_of_order_destruction();
}

NativeArrayHandle NativeArrayHandle::converted_to(const legate::Type& dtype) const
{
  auto converted = array_.as_type(dtype);
  return NativeArrayHandle{std::move(converted), true};
}

void NativeArrayHandle::convert_from(const NativeArrayHandle& input)
{
  check_writeable();
  if (dtype() == input.dtype()) {
    array_.assign(input.array_);
    return;
  }
  auto converted = input.array_.as_type(dtype());
  array_.assign(converted);
}

void NativeArrayHandle::launch_unary_op(int32_t op_code,
                                        const NativeArrayHandle& input,
                                        const std::vector<legate::Scalar>& extra_args)
{
  check_writeable();
  array_.unary_op(op_code, input.array_, extra_args);
}

void NativeArrayHandle::launch_binary_op(int32_t op_code,
                                         const NativeArrayHandle& lhs,
                                         const NativeArrayHandle& rhs)
{
  check_writeable();
  array_.binary_op(op_code, lhs.array_, rhs.array_);
}

void NativeArrayHandle::launch_binary_reduction(int32_t op_code,
                                                const NativeArrayHandle& lhs,
                                                const NativeArrayHandle& rhs)
{
  check_writeable();
  array_.binary_reduction(op_code, lhs.array_, rhs.array_);
}

void NativeArrayHandle::launch_unary_reduction(int32_t op_code, const NativeArrayHandle& input)
{
  check_writeable();
  array_.unary_reduction(op_code, input.array_);
}

}  // namespace cupynumeric::ufunc
