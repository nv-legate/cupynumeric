/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/item/write.h"
#include "cunumeric/item/write_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(1, 1)
  write_value(const AccessorWO<VAL, 1> out, const AccessorRO<VAL, DIM> value)
{
  out[0] = value[Point<DIM>::ZEROES()];
}

template <typename VAL, int DIM>
struct WriteImplBody<VariantKind::GPU, VAL, DIM> {
  void operator()(const AccessorWO<VAL, 1>& out, const AccessorRO<VAL, DIM>& value) const
  {
    auto stream = get_cached_stream();
    write_value<VAL, DIM><<<1, 1, 0, stream>>>(out, value);
    CUNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void WriteTask::gpu_variant(TaskContext context)
{
  write_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
