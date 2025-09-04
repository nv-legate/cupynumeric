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

#include "cupynumeric/item/read.h"
#include "cupynumeric/item/read_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL>
static __global__ void __launch_bounds__(1, 1)
  read_value(const AccessorWO<VAL, 1> out, const AccessorRO<VAL, 1> in)
{
  out[0] = in[0];
}

template <typename VAL>
struct ReadImplBody<VariantKind::GPU, VAL> {
  TaskContext context;
  explicit ReadImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, 1> out, AccessorRO<VAL, 1> in) const
  {
    auto stream = context.get_task_stream();
    read_value<VAL><<<1, 1, 0, stream>>>(out, in);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ReadTask::gpu_variant(TaskContext context)
{
  read_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
