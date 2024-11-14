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

#include "cupynumeric_task.h"
#include "arg.h"
#include "arg.inl"

namespace cupynumeric {

#define REGISTER_REDOPS(OP)                              \
  {                                                      \
    context.register_reduction_operator<OP<float>>();    \
    context.register_reduction_operator<OP<double>>();   \
    context.register_reduction_operator<OP<int8_t>>();   \
    context.register_reduction_operator<OP<int16_t>>();  \
    context.register_reduction_operator<OP<int32_t>>();  \
    context.register_reduction_operator<OP<int64_t>>();  \
    context.register_reduction_operator<OP<uint8_t>>();  \
    context.register_reduction_operator<OP<uint16_t>>(); \
    context.register_reduction_operator<OP<uint32_t>>(); \
    context.register_reduction_operator<OP<uint64_t>>(); \
    context.register_reduction_operator<OP<bool>>();     \
    context.register_reduction_operator<OP<__half>>();   \
  }

void register_reduction_operators(legate::LibraryContext& context)
{
  REGISTER_REDOPS(ArgmaxReduction);
  REGISTER_REDOPS(ArgminReduction);
}

}  // namespace cupynumeric
