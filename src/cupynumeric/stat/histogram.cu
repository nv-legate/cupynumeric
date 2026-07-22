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

#include "cupynumeric/stat/histogram.h"
#include "cupynumeric/stat/histogram_template.inl"

#include "cupynumeric/cuda_help.h"

#include "cupynumeric/stat/histogram.cuh"
#include "cupynumeric/stat/histogram_impl.h"

#include "cupynumeric/utilities/thrust_util.h"
#include <cupynumeric/utilities/thrust_allocator.h>

#include <tuple>

// #define _DEBUG
#ifdef _DEBUG
#include <iostream>
#include <iterator>
#include <algorithm>
#include <numeric>

#include <thrust/host_vector.h>
#endif

namespace cupynumeric {

namespace {

template <typename SrcType, typename BinType, typename WeightType>
void cub_histogram_range(const SrcType* src_ptr,
                         std::size_t src_size,
                         const BinType* bins_ptr,
                         std::int32_t bins_size,
                         WeightType* result_ptr,
                         ThrustAllocator& alloc,
                         cudaStream_t stream)
{
  size_t workspace_size = 0;
  // first call to get workspace size
  cub::DeviceHistogram::HistogramRange(nullptr,
                                       workspace_size,
                                       src_ptr,
                                       reinterpret_cast<unsigned long long*>(result_ptr),
                                       bins_size,
                                       bins_ptr,
                                       src_size,
                                       stream);

  auto workspace_ptr = alloc.allocate(workspace_size);
  assert(workspace_ptr);

  cub::DeviceHistogram::HistogramRange(workspace_ptr,
                                       workspace_size,
                                       src_ptr,
                                       reinterpret_cast<unsigned long long*>(result_ptr),
                                       bins_size,
                                       bins_ptr,
                                       src_size,
                                       stream);

  alloc.deallocate(workspace_ptr, workspace_size);
}

}  // namespace

template <Type::Code CODE>
struct HistogramNoWeightImplBody<VariantKind::GPU, CODE> {
  TaskContext context;
  explicit HistogramNoWeightImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  template <typename BinType, typename WeightType>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto alloc   = ThrustAllocator(Memory::Kind::GPU_FB_MEM);
    auto stream  = context.get_task_stream();
    auto exe_pol = DEFAULT_POLICY(alloc).on(stream);

    auto [src_size, src_ptr]       = detail::accessors::get_accessor_ptr(src, src_rect);
    auto [bins_size, bins_ptr]     = detail::accessors::get_accessor_ptr(bins, bins_rect);
    auto [result_size, result_ptr] = detail::accessors::get_accessor_ptr(result, result_rect);

    static_assert(!std::is_floating_point_v<WeightType>,
                  "weight-less histogram result type must be integral");

    // cub takes number of bins as int32_t
    if (bins_size < std::numeric_limits<std::int32_t>::max()) {
      cub_histogram_range(src_ptr,
                          src_size,
                          bins_ptr,
                          static_cast<std::int32_t>(bins_size),
                          result_ptr,
                          alloc,
                          stream);
    } else {
      detail::histogram_no_weight_thrust(
        exe_pol, src, src_rect, bins, bins_rect, result, result_rect);
    }
  }
};

/*static*/ void HistogramNoWeightTask::gpu_variant(TaskContext context)
{
  histogram_impl_type_dispatch(context, HistogramNoWeightImpl<VariantKind::GPU>{context});
}

template <Type::Code CODE>
struct HistogramWeightedImplBody<VariantKind::GPU, CODE> {
  TaskContext context;
  explicit HistogramWeightedImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  template <typename BinType, typename WeightType>
  void operator()(const AccessorRO<VAL, 1>& src,
                  const Rect<1>& src_rect,
                  const AccessorRO<BinType, 1>& bins,
                  const Rect<1>& bins_rect,
                  const AccessorRO<WeightType, 1>& weights,
                  const Rect<1>& weights_rect,
                  const AccessorRD<SumReduction<WeightType>, true, 1>& result,
                  const Rect<1>& result_rect) const
  {
    auto alloc   = ThrustAllocator(Memory::Kind::GPU_FB_MEM);
    auto stream  = context.get_task_stream();
    auto exe_pol = DEFAULT_POLICY(alloc).on(stream);

    detail::histogram_weighted_thrust(exe_pol,
                                      src,
                                      src_rect,
                                      bins,
                                      bins_rect,
                                      weights,
                                      weights_rect,
                                      result,
                                      result_rect,
                                      static_cast<cudaStream_t>(stream));
  }
};

/*static*/ void HistogramWeightedTask::gpu_variant(TaskContext context)
{
  histogram_impl_type_dispatch(context, HistogramWeightedImpl<VariantKind::GPU>{context});
}

}  // namespace cupynumeric
