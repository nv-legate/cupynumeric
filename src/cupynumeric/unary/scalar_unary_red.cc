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

#include "cupynumeric/unary/scalar_unary_red.h"
#include "cupynumeric/unary/scalar_unary_red_template.inl"

namespace cupynumeric {

/*static*/ void ScalarUnaryRedTask::cpu_variant(TaskContext context)
{
  scalar_unary_red_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
const auto reg_ = []() -> char {
  ScalarUnaryRedTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
