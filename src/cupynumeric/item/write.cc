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

#include "cupynumeric/item/write.h"
#include "cupynumeric/item/write_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename VAL, int DIM>
struct WriteImplBody<VariantKind::CPU, VAL, DIM> {
  void operator()(const AccessorWO<VAL, 1>& out, const AccessorRO<VAL, DIM>& value) const
  {
    out[0] = value[Point<DIM>::ZEROES()];
  }
};

/*static*/ void WriteTask::cpu_variant(TaskContext context)
{
  write_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  WriteTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
