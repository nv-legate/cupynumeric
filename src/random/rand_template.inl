/* Copyright 2021 NVIDIA Corporation
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

#include "arg.h"
#include "core.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, typename RNG, typename VAL, int DIM>
struct RandImplBody;

template <RandGenCode GEN_CODE, VariantKind KIND>
struct RandImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<RandomGenerator<GEN_CODE, CODE>::valid> * = nullptr>
  void operator()(RandArgs &args) const
  {
    using VAL = legate_type_of<CODE>;
    using RNG = RandomGenerator<GEN_CODE, CODE>;

    auto rect = args.out.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out     = args.out.write_accessor<VAL, DIM>(rect);
    auto strides = args.strides.to_point<DIM>();

    RNG rng(args.epoch, args.args);
    RandImplBody<KIND, RNG, VAL, DIM>{}(out, rng, strides, pitches, rect);
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!RandomGenerator<GEN_CODE, CODE>::valid> * = nullptr>
  void operator()(RandArgs &args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct RandDispatch {
  template <RandGenCode GEN_CODE>
  void operator()(RandArgs &args) const
  {
    double_dispatch(args.out.dim(), args.out.code(), RandImpl<GEN_CODE, KIND>{}, args);
  }
};

template <VariantKind KIND>
static void rand_template(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context context,
                          Runtime *runtime)
{
  Deserializer ctx(task, regions);
  RandArgs args;
  deserialize(ctx, args);
  op_dispatch(args.gen_code, RandDispatch<KIND>{}, args);
}

}  // namespace numpy
}  // namespace legate
