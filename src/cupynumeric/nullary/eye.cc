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

#include "cupynumeric/nullary/eye.h"
#include "cupynumeric/nullary/eye_template.inl"

namespace cupynumeric {

using namespace legate;

template <typename VAL>
struct EyeImplBody<VariantKind::CPU, VAL> {
  void operator()(const AccessorWO<VAL, 2>& out,
                  const Point<2>& start,
                  const coord_t distance) const
  {
    for (coord_t idx = 0; idx < distance; idx++) {
      out[start[0] + idx][start[1] + idx] = VAL{1};
    }
  }
};

/*static*/ void EyeTask::cpu_variant(TaskContext context)
{
  eye_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  EyeTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
