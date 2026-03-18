/* Copyright 2026 NVIDIA Corporation
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

#pragma once

#include <cupynumeric/pitches.h>
#include <cupynumeric/unary/unary_op.h>
#include <cupynumeric/utilities/thrust_util.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

namespace cupynumeric {

using namespace legate;

template <typename T, int32_t OUT_DIM>
struct ScatterFunctorDense {
  T* out;
  const Point<OUT_DIM> out_strides;
  const T* src;
  const Point<OUT_DIM>* indices;

  LEGATE_HOST_DEVICE void operator()(size_t tid) const
  {
    out[indices[tid].dot(out_strides)] = src[tid];
  }
};

template <typename T, int32_t OUT_DIM, int32_t SRC_DIM>
struct ScatterFunctor {
  AccessorWO<T, OUT_DIM> out;
  AccessorRO<T, SRC_DIM> src;
  AccessorRO<Point<OUT_DIM>, SRC_DIM> indices;
  Pitches<SRC_DIM - 1> pitches;
  Point<SRC_DIM> lo;

  LEGATE_HOST_DEVICE void operator()(size_t tid) const
  {
    auto p          = pitches.unflatten(tid, lo);
    out[indices[p]] = src[p];
  }
};

template <typename exec_policy_t, typename T, int32_t OUT_DIM, int32_t SRC_DIM>
struct ScatterImplBody {
  void operator()(const exec_policy_t& policy,
                  PhysicalStore output,
                  PhysicalStore source,
                  PhysicalStore indices)
  {
    auto out_rect = output.shape<OUT_DIM>();
    auto src_rect = source.shape<SRC_DIM>();
    auto idx_rect = indices.shape<SRC_DIM>();

    assert(src_rect == idx_rect);

    auto out_acc = output.write_accessor<T, OUT_DIM>();
    auto src_acc = source.read_accessor<T, SRC_DIM>();
    auto idx_acc = indices.read_accessor<Point<OUT_DIM>, SRC_DIM>();

    Pitches<SRC_DIM - 1> src_pitches;
    size_t volume = src_pitches.flatten(src_rect);

    if (volume == 0) {
      return;
    }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    // Check to see if this is dense or not
    bool dense = out_acc.accessor.is_dense_row_major(out_rect) &&
                 src_acc.accessor.is_dense_row_major(src_rect) &&
                 idx_acc.accessor.is_dense_row_major(idx_rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    if (dense) {
      size_t out_strides[OUT_DIM];
      auto outptr = out_acc.ptr(out_rect, out_strides);
      auto srcptr = src_acc.ptr(src_rect);
      auto idxptr = idx_acc.ptr(idx_rect);

      thrust::for_each(
        policy,
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(volume),
        ScatterFunctorDense<T, OUT_DIM>{outptr, Point<OUT_DIM>{out_strides}, srcptr, idxptr});
    } else {
      thrust::for_each(
        policy,
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(volume),
        ScatterFunctor<T, OUT_DIM, SRC_DIM>{out_acc, src_acc, idx_acc, src_pitches, src_rect.lo});
    }
  }
};

template <typename exec_policy_t, typename T>
struct ScatterDimDispatch {
  const exec_policy_t& policy;
  PhysicalStore output;
  PhysicalStore source;
  PhysicalStore indices;

  template <int32_t OUT_DIM, int32_t SRC_DIM>
  void operator()()
  {
    ScatterImplBody<exec_policy_t, T, OUT_DIM, SRC_DIM>{}(policy, output, source, indices);
  }
};

template <typename exec_policy_t>
struct ScatterTypeDispatch {
  const exec_policy_t& policy;

  template <Type::Code CODE>
  void operator()(PhysicalStore output, PhysicalStore source, PhysicalStore indices) const
  {
    using T = type_of<CODE>;
    ScatterDimDispatch<exec_policy_t, T> impl{policy, output, source, indices};
    cupynumeric::double_dispatch(source.dim(), output.dim(), impl);
  }
};

template <typename exec_policy_t>
static void scatter_template(TaskContext& context, const exec_policy_t& policy)
{
  PhysicalStore output  = context.output(0);
  PhysicalStore source  = context.input(0);
  PhysicalStore indices = context.input(1);

  type_dispatch(
    source.type().code(), ScatterTypeDispatch<exec_policy_t>{policy}, output, source, indices);
}

}  // namespace cupynumeric