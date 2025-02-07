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

namespace cupynumeric {

template <typename T, int32_t DIM>
legate::AccessorRO<T, DIM> NDArray::get_read_accessor()
{
  auto mapped = store_.get_physical_store();
  return mapped.read_accessor<T, DIM>();
}

template <typename T, int32_t DIM>
legate::AccessorWO<T, DIM> NDArray::get_write_accessor()
{
  auto mapped = store_.get_physical_store();
  return mapped.write_accessor<T, DIM>();
}

}  // namespace cupynumeric
