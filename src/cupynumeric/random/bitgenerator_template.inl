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

#pragma once

#include "cupynumeric/arg.h"
#include "cupynumeric/pitches.h"

#include "bitgenerator_util.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND>
struct BitGeneratorImplBody;

template <VariantKind KIND>
struct BitGeneratorImpl {
  void operator()(BitGeneratorArgs& args) const
  {
    BitGeneratorImplBody<KIND>{}(args.bitgen_op,
                                 args.generatorID,
                                 args.generatorType,
                                 args.seed,
                                 args.flags,
                                 args.distribution,
                                 args.strides,
                                 args.intparams,
                                 args.floatparams,
                                 args.doubleparams,
                                 args.output,
                                 args.args);
  }
};

template <VariantKind KIND>
static void bitgenerator_template(TaskContext& context)
{
  auto inputs        = context.inputs();
  auto outputs       = context.outputs();
  auto scalars       = context.scalars();
  auto bitgen_op     = scalars[0].value<BitGeneratorOperation>();
  auto generatorID   = scalars[1].value<int32_t>();
  auto generatorType = scalars[2].value<BitGeneratorType>();
  auto seed          = scalars[3].value<uint64_t>();
  auto flags         = scalars[4].value<uint32_t>();

  BitGeneratorDistribution distribution{};
  std::vector<int64_t> intparams;
  std::vector<float> floatparams;
  std::vector<double> doubleparams;

  DomainPoint strides;  // optional parameter
  legate::Span<const int32_t> todestroy;
  switch (bitgen_op) {
    case BitGeneratorOperation::DESTROY:  // gather same parameters as CREATE
    case BitGeneratorOperation::CREATE: {
      if (scalars.size() > 5) {
        todestroy = scalars[5].values<int32_t>();
      }
      break;
    }
    case BitGeneratorOperation::RAND_RAW: {
      if (scalars.size() > 5) {
        strides = scalars[5].value<DomainPoint>();
      }
      break;
    }
    case BitGeneratorOperation::DISTRIBUTION: {
      if (scalars.size() < 10) {
        ::fprintf(stderr, "FATAL: not enough parameters\n");
        ::exit(1);
      }
      distribution    = scalars[5].value<BitGeneratorDistribution>();
      auto _intparams = scalars[7].values<int64_t>();
      intparams.insert(intparams.end(), _intparams.begin(), _intparams.end());
      auto _floatparams = scalars[8].values<float>();
      floatparams.insert(floatparams.end(), _floatparams.begin(), _floatparams.end());
      auto _doubleparams = scalars[9].values<double>();
      doubleparams.insert(doubleparams.end(), _doubleparams.begin(), _doubleparams.end());
      break;
    }
    default: LEGATE_ABORT("Unknown bitgenerator op");
  }

  std::vector<PhysicalStore> extra_args;
  for (auto& input : inputs) {
    extra_args.push_back(std::move(input));
  }

  std::vector<PhysicalStore> optional_output;
  for (auto& output : outputs) {
    optional_output.push_back(std::move(output));
  }

  // destroy ?
  for (auto& idx : todestroy) {
    auto dargs = BitGeneratorArgs::destroy(idx);
    BitGeneratorImpl<KIND>{}(dargs);
  }

  BitGeneratorArgs args(bitgen_op,
                        generatorID,
                        generatorType,
                        seed,
                        flags,
                        distribution,
                        strides,
                        std::move(intparams),
                        std::move(floatparams),
                        std::move(doubleparams),
                        std::move(optional_output),
                        std::move(extra_args));
  BitGeneratorImpl<KIND>{}(args);
}

}  // namespace cupynumeric
