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

template <typename exec_policy_t, Type::Code CODE, bool clip>
struct TakeImplBody {
  TaskContext context;
  explicit TakeImplBody(TaskContext context) : context(context) {}

  using VAL         = type_of<CODE>;
  using SourceArray = AccessorRO<VAL, 4>;
  using Indices     = AccessorRO<int64_t, 1>;
  using ResultArray = AccessorWO<VAL, 4>;

  struct TakeFunctor {
   public:
    TakeFunctor(
      SourceArray src, Indices ind, ResultArray res, Pitches<2> pitches, Point<3> lo, int64_t m)
      : src{src}, ind{ind}, res{res}, pitches{pitches}, lo{lo}, m{m} {};

    LEGATE_HOST_DEVICE
    void operator()(size_t idx) const
    {
      auto res_point = pitches.unflatten(idx, lo);
      auto j         = res_point[1];
      auto take_idx  = ind[j];
      auto i         = clip ? std::clamp<int64_t>(take_idx, 0, m - 1) : detail::mod(take_idx, m);

      res[Point<4>(res_point[0], 0, j, res_point[2])] =
        src[Point<4>(res_point[0], i, 0, res_point[2])];
    }

   private:
    SourceArray src;
    Indices ind;
    ResultArray res;
    Point<3> lo;
    Pitches<2> pitches;
    int64_t m;
  };

  void operator()(const exec_policy_t& policy,
                  const SourceArray& src,
                  const Indices& ind,
                  const ResultArray& res,
                  const Rect<4>& shape)
  {
    // the second dimension is fictitious for the result
    // TODO(tisaac): correct this when alignment on a subset of dimensions is possible
    Rect<3> res_reduced_shape(Point<3>(shape.lo[0], shape.lo[2], shape.lo[3]),
                              Point<3>(shape.hi[0], shape.hi[2], shape.hi[3]));
    Point<3> res_reduced_lo(shape.lo[0], shape.lo[2], shape.lo[3]);

    Pitches<2> res_pitches{};
    auto res_volume = res_pitches.flatten(res_reduced_shape);
    auto m          = shape.hi[1] + 1 - shape.lo[1];

    // TODO(tisaac): optimize for cases (j == 1 && n == 1), (j == 1), and (n == 1)
    thrust::for_each(policy,
                     thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(res_volume),
                     TakeFunctor(src, ind, res, res_pitches, res_reduced_lo, m));
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
    auto ind_shape = args.ind.shape<1>();
    auto res_shape = args.res.shape<4>();

    assert(src_shape == res_shape);
    assert(ind_shape.hi[0] == src_shape.hi[2]);
    assert(ind_shape.lo[0] == src_shape.lo[2]);

    auto src = args.src.read_accessor<VAL, 4>();
    auto ind = args.ind.read_accessor<int64_t, 1>();
    auto res = args.res.write_accessor<VAL, 4>();
    if (args.clip) {
      TakeImplBody<exec_policy_t, CODE, true>{context}(policy, src, ind, res, src_shape);
    } else {
      TakeImplBody<exec_policy_t, CODE, false>{context}(policy, src, ind, res, src_shape);
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
