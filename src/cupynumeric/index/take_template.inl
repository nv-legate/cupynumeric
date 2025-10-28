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
// - For take_along_axis(): indices are promoted on dim 1, so ind[a,b,c,d] == ind[a,0,c,d]
// Thus, reading ind[res_point[0], 0, j, res_point[2]] works correctly for both cases.
template <typename exec_policy_t, Type::Code CODE, bool clip>
struct TakeImplBody {
  TaskContext context;
  explicit TakeImplBody(TaskContext context) : context(context) {}

  using VAL         = type_of<CODE>;
  using SourceArray = AccessorRO<VAL, 4>;
  using Indices     = AccessorRO<int64_t, 4>;
  using ResultArray = AccessorWO<VAL, 4>;

  struct TakeFunctorAB {
    SourceArray src;
    Indices ind;
    ResultArray res;
    Pitches<2> pitches;
    Point<3> lo;
    int64_t m;

    LEGATE_HOST_DEVICE
    void operator()(size_t idx) const
    {
      auto res_point = pitches.unflatten(idx, lo);
      auto j         = res_point[1];
      auto take_idx  = ind[Point<4>(res_point[0], 0, j, res_point[2])];
      auto i         = clip ? std::clamp<int64_t>(take_idx, 0, m - 1) : detail::mod(take_idx, m);

      res[Point<4>(res_point[0], 0, j, res_point[2])] =
        src[Point<4>(res_point[0], i, 0, res_point[2])];
    }
  };

  struct TakeFunctor1B {
    SourceArray src;
    Indices ind;
    ResultArray res;
    Pitches<1> pitches;
    Point<3> lo;
    int64_t m;

    LEGATE_HOST_DEVICE
    void operator()(size_t idx) const
    {
      Point<2> lo_sub{lo[1], lo[2]};

      auto res_point = pitches.unflatten(idx, lo_sub);
      auto j         = res_point[0];
      auto take_idx  = ind[Point<4>(lo[0], 0, j, res_point[1])];
      auto i         = clip ? std::clamp<int64_t>(take_idx, 0, m - 1) : detail::mod(take_idx, m);

      res[Point<4>(lo[0], 0, j, res_point[1])] = src[Point<4>(lo[0], i, 0, res_point[1])];
    }
  };

  struct TakeFunctorA1 {
    SourceArray src;
    Indices ind;
    ResultArray res;
    Pitches<1> pitches;
    Point<3> lo;
    int64_t m;

    LEGATE_HOST_DEVICE
    void operator()(size_t idx) const
    {
      Point<2> lo_sub{lo[0], lo[1]};

      auto res_point = pitches.unflatten(idx, lo_sub);
      auto j         = res_point[1];
      auto take_idx  = ind[Point<4>(res_point[0], 0, j, lo[2])];
      auto i         = clip ? std::clamp<int64_t>(take_idx, 0, m - 1) : detail::mod(take_idx, m);

      res[Point<4>(res_point[0], 0, j, lo[2])] = src[Point<4>(res_point[0], i, 0, lo[2])];
    }
  };

  struct TakeFunctor11 {
    SourceArray src;
    Indices ind;
    ResultArray res;
    Point<3> lo;
    int64_t m;

    LEGATE_HOST_DEVICE
    void operator()(size_t idx) const
    {
      auto j        = lo[1] + idx;
      auto take_idx = ind[Point<4>(lo[0], 0, j, lo[2])];
      auto i        = clip ? std::clamp<int64_t>(take_idx, 0, m - 1) : detail::mod(take_idx, m);

      res[Point<4>(lo[0], 0, j, lo[2])] = src[Point<4>(lo[0], i, 0, lo[2])];
    }
  };

  void operator()(const exec_policy_t& policy,
                  const SourceArray& src,
                  const Indices& ind,
                  const ResultArray& res,
                  const Rect<4>& shape)
  {
    // the second dimension is fictitious for the result
    Point<3> lo(shape.lo[0], shape.lo[2], shape.lo[3]);

    auto a = shape.hi[0] + 1 - shape.lo[0];
    auto m = shape.hi[1] + 1 - shape.lo[1];
    auto b = shape.hi[3] + 1 - shape.lo[3];

    if (a != 1 && b != 1) {
      Rect<3> res_reduced_shape(Point<3>(shape.lo[0], shape.lo[2], shape.lo[3]),
                                Point<3>(shape.hi[0], shape.hi[2], shape.hi[3]));
      Pitches<2> res_pitches{};
      auto res_volume = res_pitches.flatten(res_reduced_shape);
      if (res_volume <= 0) {
        return;
      }

      thrust::for_each(policy,
                       thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(res_volume),
                       TakeFunctorAB{src, ind, res, res_pitches, lo, m});
    } else if (a != 1 && b == 1) {
      Rect<2> res_reduced_shape(Point<2>(shape.lo[0], shape.lo[2]),
                                Point<2>(shape.hi[0], shape.hi[2]));
      Pitches<1> res_pitches{};
      auto res_volume = res_pitches.flatten(res_reduced_shape);

      thrust::for_each(policy,
                       thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(res_volume),
                       TakeFunctorA1{src, ind, res, res_pitches, lo, m});
    } else if (a == 1 && b != 1) {
      Rect<2> res_reduced_shape(Point<2>(shape.lo[2], shape.lo[3]),
                                Point<2>(shape.hi[2], shape.hi[3]));
      Pitches<1> res_pitches{};
      auto res_volume = res_pitches.flatten(res_reduced_shape);

      thrust::for_each(policy,
                       thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(res_volume),
                       TakeFunctor1B{src, ind, res, res_pitches, lo, m});
    } else {
      auto res_volume = shape.hi[2] + 1 - shape.lo[2];

      thrust::for_each(policy,
                       thrust::counting_iterator<size_t>(0),
                       thrust::counting_iterator<size_t>(res_volume),
                       TakeFunctor11{src, ind, res, lo, m});
    }
  }
};

template <typename exec_policy_t>
struct TakeImpl {
  TaskContext context;
  explicit TakeImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(TakeArgs& args, const exec_policy_t& policy) const
  {
    using VAL = type_of<CODE>;

    auto src_shape = args.src.shape<4>();
    auto ind_shape = args.ind.shape<4>();
    auto res_shape = args.res.shape<4>();

    Point<4> work_lo;
    Point<4> work_hi;

    for (size_t i = 0; i < 4; i++) {
      work_lo[i] = std::max(src_shape.lo[i], std::max(ind_shape.lo[i], res_shape.lo[i]));
      work_hi[i] = std::min(src_shape.hi[i], std::min(ind_shape.hi[i], res_shape.hi[i]));
    }
    Rect<4> work_shape{work_lo, work_hi};

    auto src = args.src.read_accessor<VAL, 4>();
    auto ind = args.ind.read_accessor<int64_t, 4>();
    auto res = args.res.write_accessor<VAL, 4>();
    if (args.clip) {
      TakeImplBody<exec_policy_t, CODE, true>{context}(policy, src, ind, res, work_shape);
    } else {
      TakeImplBody<exec_policy_t, CODE, false>{context}(policy, src, ind, res, work_shape);
    }
  }
};

template <typename exec_policy_t>
static void take_template(TaskContext& context, const exec_policy_t& policy)
{
  TakeArgs args{
    context.input(0), context.input(1), context.output(0), context.scalar(0).value<bool>()};

  type_dispatch(args.res.code(), TakeImpl<exec_policy_t>{context}, args, policy);
}

}  // namespace cupynumeric
