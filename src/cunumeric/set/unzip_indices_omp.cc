/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/set/unzip_indices.h"
#include "cunumeric/set/unzip_indices_template.inl"

namespace cunumeric {

using namespace legate;

template <Type::Code CODE>
struct UniqueImplBody<VariantKind::OMP, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, 1>& values,
                  const AccessorWO<int64_t, 1>& indices,
                  const AccessorRO<ZippedIndex<VAL>, 1>& in,
                  const Rect<1> input_shape)
  {
#pragma omp parallel for schedule(static)
    for (coord_t i = input_shape.lo[0]; i < input_shape.hi[0] + 1; i++) {
      values[i]  = in[i].value;
      indices[i] = in[i].index;
    }
  }
};

/*static*/ void UnzipIndicesTask::omp_variant(TaskContext& context)
{
  unzip_indices_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
