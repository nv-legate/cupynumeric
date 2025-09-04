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

#include "cupynumeric/scan/scan_global.h"
#include "cupynumeric/scan/scan_global_template.inl"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <omp.h>

namespace cupynumeric {

using namespace legate;

template <ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanGlobalImplBody<VariantKind::OMP, OP_CODE, CODE, DIM> {
  TaskContext context;
  explicit ScanGlobalImplBody(TaskContext context) : context(context) {}

  using OP  = ScanOp<OP_CODE, CODE>;
  using VAL = type_of<CODE>;

  void operator()(OP func,
                  const AccessorRW<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& sum_vals,
                  const Pitches<DIM - 1>& out_pitches,
                  const Rect<DIM>& out_rect,
                  const Pitches<DIM - 1>& sum_vals_pitches,
                  const Rect<DIM>& sum_vals_rect,
                  const DomainPoint& partition_index) const
  {
    auto outptr = out.ptr(out_rect.lo);
    auto volume = out_rect.volume();

    if (partition_index[DIM - 1] == 0) {
      // first partition has nothing to do and can return;
      return;
    }

    auto stride = out_rect.hi[DIM - 1] - out_rect.lo[DIM - 1] + 1;
    for (uint64_t index = 0; index < volume; index += stride) {
      // get the corresponding ND index to use for sum_val
      auto sum_valsp = out_pitches.unflatten(index, out_rect.lo);
      // first element on scan axis
      sum_valsp[DIM - 1]     = 0;
      auto sum_valsp_end     = sum_valsp;
      sum_valsp_end[DIM - 1] = partition_index[DIM - 1];
      auto global_prefix     = thrust::reduce(thrust::omp::par,
                                          &sum_vals[sum_valsp],
                                          &sum_vals[sum_valsp_end],
                                          (VAL)ScanOp<OP_CODE, CODE>::nan_identity,
                                          func);
      // apply global_prefix to out
#pragma omp parallel for schedule(static)
      for (uint64_t i = index; i < index + stride; i++) {
        outptr[i] = func(outptr[i], global_prefix);
      }
    }
  }
};

/*static*/ void ScanGlobalTask::omp_variant(TaskContext context)
{
  scan_global_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
