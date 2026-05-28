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
#include "cupynumeric/utilities/thrust_util.h"

#include "cupynumeric/cuda_help.h"

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <type_traits>

namespace cupynumeric {

using namespace legate;

namespace {

template <typename VAL, typename OP, typename IndexT>
void scan_impl(
  OP func, size_t volume, VAL* outptr, const VAL* inptr, IndexT stride, cudaStream_t stream)
{
  auto accessor_key = [stride] __host__ __device__(IndexT indx) -> IndexT { return indx / stride; };
  auto key_traversor_start =
    thrust::make_transform_iterator(thrust::make_counting_iterator<IndexT>(0), accessor_key);

  thrust::inclusive_scan_by_key(DEFAULT_POLICY.on(stream),
                                key_traversor_start,
                                key_traversor_start + volume,
                                inptr,
                                outptr,
                                thrust::equal_to<IndexT>{},
                                func);
}

template <typename VAL, typename OP, typename IndexT, typename CONVERT_FUNC>
void nan_scan_impl(OP func,
                   size_t volume,
                   VAL* outptr,
                   const VAL* inptr,
                   IndexT stride,
                   cudaStream_t stream,
                   CONVERT_FUNC convert_nan_func)
{
  auto accessor_key = [stride] __host__ __device__(IndexT indx) -> IndexT { return indx / stride; };
  auto key_traversor_start =
    thrust::make_transform_iterator(thrust::make_counting_iterator<IndexT>(0), accessor_key);

  thrust::inclusive_scan_by_key(DEFAULT_POLICY.on(stream),
                                key_traversor_start,
                                key_traversor_start + volume,
                                thrust::make_transform_iterator(inptr, convert_nan_func),
                                outptr,
                                thrust::equal_to<IndexT>{},
                                func);
}

template <typename VAL, typename IndexT>
void copy_boundary_values(
  size_t volume, VAL* sum_valsptr, VAL* outptr, IndexT stride, cudaStream_t stream)
{
  thrust::transform(DEFAULT_POLICY.on(stream),
                    thrust::make_counting_iterator<IndexT>(0),
                    thrust::make_counting_iterator<IndexT>(volume / stride),
                    sum_valsptr,
                    [stride, outptr] __device__(auto axial_indx) {
                      return outptr[axial_indx * stride + stride - 1];
                    });
}

}  // namespace

template <ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanLocalImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  TaskContext context;
  explicit ScanLocalImplBody(TaskContext context) : context(context) {}

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

    auto stream = context.get_task_stream();

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1]   = 1;  // one element along scan axis

    scan_impl(func, volume, outptr, inptr, stride, stream);

    auto sum_valsptr =
      sum_vals.create_output_buffer<VAL, DIM>(extents, true).ptr(Point<DIM>::ZEROES());

    copy_boundary_values(volume, sum_valsptr, outptr, stride, stream);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanLocalNanImplBody<VariantKind::GPU, OP_CODE, CODE, DIM> {
  TaskContext context;
  explicit ScanLocalNanImplBody(TaskContext context) : context(context) {}

  using OP  = ScanOp<OP_CODE, CODE>;
  using VAL = type_of<CODE>;

  struct convert_nan_func {
    __device__ VAL operator()(VAL x)
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

    auto stream = context.get_task_stream();

    Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
    extents[DIM - 1]   = 1;  // one element along scan axis

    auto sum_valsptr =
      sum_vals.create_output_buffer<VAL, DIM>(extents, true).ptr(Point<DIM>::ZEROES());

    nan_scan_impl(func, volume, outptr, inptr, stride, stream, convert_nan_func{});

    copy_boundary_values(volume, sum_valsptr, outptr, stride, stream);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ScanLocalTask::gpu_variant(TaskContext context)
{
  scan_local_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
