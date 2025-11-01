/* Copyright 2025 NVIDIA Corporation
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
#include <cupynumeric/utilities/thrust_util.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <algorithm>

namespace cupynumeric {

using namespace legate;

namespace detail {

inline constexpr int64_t LEGATE_HOST_DEVICE mod(int64_t x, int64_t m)
{
  int64_t y = x % m;
  if (y < 0) {
    y += m;
  }
  return y;
}

}  // namespace detail

// This task handles both take() and take_along_axis() with the same code.
// The Python layer uses promote() to broadcast indices appropriately:
// - For take(): indices are promoted uniformly, so ind[a,b,c,d] == ind[0,0,c,0]
// - For take_along_axis(): indices are promoted on axis, so if e.g. axis is 1
//   then ind[a,b,c,d] == ind[a,0,c,d]
template <typename exec_policy_t, Type::Code CODE, int DIM>
struct TakeImplBody {
  TaskContext context;
  explicit TakeImplBody(TaskContext context) : context(context) {}

  using VAL         = type_of<CODE>;
  using SourceArray = AccessorRO<VAL, DIM>;
  using Indices     = AccessorRO<int64_t, DIM>;
  using ResultArray = AccessorWO<VAL, DIM>;

  template <bool clip>
  struct TakeFunctor {
    SourceArray src;
    Indices ind;
    ResultArray res;
    Pitches<DIM - 1> pitches;
    Point<DIM> lo;
    int8_t axis;
    int64_t m;
    size_t axis_pitch;

    LEGATE_HOST_DEVICE
    void operator()(size_t idx) const
    {
      auto res_point = pitches.unflatten(idx, lo);
      auto take_idx  = ind[res_point];
      auto i         = clip ? std::clamp<int64_t>(take_idx, 0, m - 1) : detail::mod(take_idx, m);
      const VAL* axis_start = &src[res_point];

      // NOTE(tisaac, 2025-10-31): here we could have done
      //
      //     auto src_point = res_point;
      //
      //     src_point[axis] = i;
      //     res[res_point]  = src[src_point];
      //
      // but the line `src_point[axis] = i` with `axis` being a value that is not known at compile
      // time resulted in significantly slower execution on the GPU.

      res[res_point] = axis_start[i * axis_pitch];
    }
  };

  void operator()(const exec_policy_t& policy,
                  const SourceArray& src,
                  const Indices& ind,
                  const ResultArray& res,
                  const Rect<DIM>& shape,
                  const int8_t axis,
                  const coord_t m,
                  const size_t axis_pitch,
                  const bool clip)
  {
    Pitches<DIM - 1> pitches{};

    auto volume = pitches.flatten(shape);
    if (volume <= 0) {
      return;
    }

    if (clip) {
      thrust::for_each(policy,
                       thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(volume),
                       TakeFunctor<true>{src, ind, res, pitches, shape.lo, axis, m, axis_pitch});
    } else {
      thrust::for_each(policy,
                       thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(volume),
                       TakeFunctor<false>{src, ind, res, pitches, shape.lo, axis, m, axis_pitch});
    }
  }
};

template <typename exec_policy_t>
struct TakeImpl {
  TaskContext context;
  explicit TakeImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(TakeArgs& args, const exec_policy_t& policy) const
  {
    using VAL = type_of<CODE>;

    auto src_shape = args.src.shape<DIM>();
    auto ind_shape = args.ind.shape<DIM>();
    auto res_shape = args.res.shape<DIM>();
    auto src       = args.src.read_accessor<VAL, DIM>();
    auto ind       = args.ind.read_accessor<int64_t, DIM>();
    auto res       = args.res.write_accessor<VAL, DIM>();

    Point<DIM> work_lo;
    Point<DIM> work_hi;

    for (size_t d = 0; d < DIM; d++) {
      work_lo[d] = std::max(src_shape.lo[d], std::max(ind_shape.lo[d], res_shape.lo[d]));
      work_hi[d] = std::min(src_shape.hi[d], std::min(ind_shape.hi[d], res_shape.hi[d]));
    }

    // the work is associated with the result/indices arrays, and for these arrays 'axis' is a
    // phony dimension, so we set the working interval to [0,0] on this dimension
    auto m             = work_hi[args.axis] + 1 - work_lo[args.axis];
    work_lo[args.axis] = 0;
    work_hi[args.axis] = 0;
    Rect<DIM> work_shape{work_lo, work_hi};

    // see NOTE in TakeFunctor(): the kernel is significantly faster on the GPU if we
    // precompute the pitch on the axis of interest
    size_t axis_pitch = 0;
    if (src_shape.hi[args.axis] > src_shape.lo[args.axis]) {
      auto axis_start = work_lo;
      auto up_one     = work_lo;

      axis_start[args.axis] = work_lo[args.axis];
      up_one[args.axis]     = work_lo[args.axis] + 1;

      axis_pitch = &src[up_one] - &src[axis_start];
    }

    TakeImplBody<exec_policy_t, CODE, DIM>{context}(
      policy, src, ind, res, work_shape, args.axis, m, axis_pitch, args.clip);
  }
};

template <typename exec_policy_t>
static void take_template(TaskContext& context, const exec_policy_t& policy)
{
  TakeArgs args{context.input(0),
                context.input(1),
                context.output(0),
                context.scalar(0).value<std::int8_t>(),
                context.scalar(1).value<bool>()};

  double_dispatch(args.res.dim(), args.res.code(), TakeImpl<exec_policy_t>{context}, args, policy);
}

}  // namespace cupynumeric
