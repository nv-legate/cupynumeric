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

#include "cupynumeric/scan/scan_local.h"
#include "cupynumeric/scan/scan_local_template.inl"
#include "cupynumeric/unary/isnan.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

namespace cupynumeric {

using namespace legate;

template <ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanLocalImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = ScanOp<OP_CODE, CODE>;
  using VAL = type_of<CODE>;

  void operator()(OP func,
                  const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  legate::PhysicalStore& sum_vals,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    auto outptr = out.ptr(rect.lo);
    auto inptr  = in.ptr(rect.lo);
    auto volume = rect.volume();

    auto stride = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1]   = 1;  // one element along scan axis

    auto sum_valsptr = sum_vals.create_output_buffer<VAL, DIM>(extents, true);

    for (uint64_t index = 0; index < volume; index += stride) {
      thrust::inclusive_scan(
        thrust::host, inptr + index, inptr + index + stride, outptr + index, func);
      // get the corresponding ND index with base zero to use for sum_val
      auto sum_valp = pitches.unflatten(index, Point<DIM>::ZEROES());
      // only one element on scan axis
      sum_valp[DIM - 1] = 0;
      // write out the partition sum
      sum_valsptr[sum_valp] = outptr[index + stride - 1];
    }
  }
};

template <ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanLocalNanImplBody<VariantKind::CPU, OP_CODE, CODE, DIM> {
  using OP  = ScanOp<OP_CODE, CODE>;
  using VAL = type_of<CODE>;

  struct convert_nan_func {
    VAL operator()(VAL x) const
    {
      return cupynumeric::is_nan(x) ? (VAL)ScanOp<OP_CODE, CODE>::nan_identity : x;
    }
  };

  void operator()(OP func,
                  const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  legate::PhysicalStore& sum_vals,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    auto outptr = out.ptr(rect.lo);
    auto inptr  = in.ptr(rect.lo);
    auto volume = rect.volume();

    auto stride = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1]   = 1;  // one element along scan axis

    auto sum_valsptr = sum_vals.create_output_buffer<VAL, DIM>(extents, true);

    for (uint64_t index = 0; index < volume; index += stride) {
      thrust::inclusive_scan(
        thrust::host,
        thrust::make_transform_iterator(inptr + index, convert_nan_func()),
        thrust::make_transform_iterator(inptr + index + stride, convert_nan_func()),
        outptr + index,
        func);
      // get the corresponding ND index with base zero to use for sum_val
      auto sum_valp = pitches.unflatten(index, Point<DIM>::ZEROES());
      // only one element on scan axis
      sum_valp[DIM - 1] = 0;
      // write out the partition sum
      sum_valsptr[sum_valp] = outptr[index + stride - 1];
    }
  }
};

/*static*/ void ScanLocalTask::cpu_variant(TaskContext context)
{
  scan_local_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScanLocalTask::register_variants();
}
}  // namespace

}  // namespace cupynumeric
