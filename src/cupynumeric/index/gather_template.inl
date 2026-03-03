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

template <typename T, int32_t OUT_DIM, int32_t SRC_DIM>
struct GatherFunctor {
  AccessorWO<T, OUT_DIM> out;
  AccessorRO<T, SRC_DIM> src;
  AccessorRO<Point<SRC_DIM>, OUT_DIM> indices;
  Pitches<OUT_DIM - 1> pitches;
  Point<OUT_DIM> lo;

  LEGATE_HOST_DEVICE void operator()(size_t tid) const
  {
    auto p = pitches.unflatten(tid, lo);
    out[p] = src[indices[p]];
  }
};

template <typename exec_policy_t, typename T, int32_t OUT_DIM, int32_t SRC_DIM>
struct GatherImplBody {
  void operator()(const exec_policy_t& policy,
                  PhysicalStore output,
                  PhysicalStore source,
                  PhysicalStore indices)
  {
    auto out_rect = output.shape<OUT_DIM>();
    auto idx_rect = indices.shape<OUT_DIM>();
    assert(out_rect == idx_rect);
    auto out_acc = output.write_accessor<T, OUT_DIM>();
    auto src_acc = source.read_accessor<T, SRC_DIM>();
    auto idx_acc = indices.read_accessor<Point<SRC_DIM>, OUT_DIM>();

    Pitches<OUT_DIM - 1> out_pitches;
    size_t volume = out_pitches.flatten(out_rect);

    if (volume == 0) {
      return;
    }

    thrust::for_each(
      policy,
      thrust::counting_iterator<size_t>(0),
      thrust::counting_iterator<size_t>(volume),
      GatherFunctor<T, OUT_DIM, SRC_DIM>{out_acc, src_acc, idx_acc, out_pitches, out_rect.lo});
  }
};

template <typename exec_policy_t, typename T>
struct GatherDimDispatch {
  const exec_policy_t& policy;
  PhysicalStore output;
  PhysicalStore source;
  PhysicalStore indices;

  template <int32_t OUT_DIM, int32_t SRC_DIM>
  void operator()()
  {
    GatherImplBody<exec_policy_t, T, OUT_DIM, SRC_DIM>{}(policy, output, source, indices);
  }
};

template <typename exec_policy_t>
struct GatherTypeDispatch {
  const exec_policy_t& policy;

  template <Type::Code CODE>
  void operator()(PhysicalStore output, PhysicalStore source, PhysicalStore indices) const
  {
    using T = type_of<CODE>;
    GatherDimDispatch<exec_policy_t, T> impl{policy, output, source, indices};
    cupynumeric::double_dispatch(source.dim(), output.dim(), impl);
  }
};

template <typename exec_policy_t>
static void gather_template(TaskContext& context, const exec_policy_t& policy)
{
  PhysicalStore output  = context.output(0);
  PhysicalStore source  = context.input(0);
  PhysicalStore indices = context.input(1);

  type_dispatch(
    source.type().code(), GatherTypeDispatch<exec_policy_t>{policy}, output, source, indices);
}

}  // namespace cupynumeric
