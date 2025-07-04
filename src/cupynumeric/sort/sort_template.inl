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

// Useful for IDEs
#include "cupynumeric/sort/sort.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct SortImplBody;

static int get_rank(Domain domain, DomainPoint index_point)
{
  int domain_index = 0;
  auto hi          = domain.hi();
  auto lo          = domain.lo();
  for (int i = 0; i < domain.get_dim(); ++i) {
    if (i > 0) {
      domain_index *= hi[i] - lo[i] + 1;
    }
    domain_index += index_point[i];
  }
  return domain_index;
}

template <VariantKind KIND>
struct SortImpl {
  template <Type::Code CODE, int32_t DIM>
  void operator()(SortArgs& args, TaskContext& context, std::vector<comm::Communicator> comms) const
  {
    auto rect = args.input.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    int64_t segment_size  = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;
    size_t segment_size_l = segment_size > 0 ? segment_size : 0;

    /*
     * Assumptions:
     * 1. Sort is always requested for the 'last' dimension within rect
     * 2. We have product_of_all_other_dimensions independent sort ranges
     * 3. if data distributed accross sort dimension we perform sample sort
     */

    // we shall not return on empty rectangle in case of distributed sort data
    // as the process needs to participate in collective communication
    // to identify rank-index to sort participant mapping
    if ((segment_size_l == args.segment_size_g || !args.is_index_space) && rect.empty()) {
      if (args.output.is_unbound_store()) {
        args.output.bind_empty_data();
      }
      return;
    }

    SortImplBody<KIND, CODE, DIM>()(context,
                                    args.input,
                                    args.output,
                                    pitches,
                                    rect,
                                    volume,
                                    segment_size_l,
                                    args.segment_size_g,
                                    args.argsort,
                                    args.stable,
                                    args.is_index_space,
                                    args.local_rank,
                                    args.num_ranks,
                                    args.num_sort_ranks,
                                    comms);
  }
};

template <VariantKind KIND>
static void sort_template(TaskContext& context)
{
  auto shape_span       = context.scalar(1).values<int64_t>();
  size_t segment_size_g = shape_span[shape_span.size() - 1];
  auto domain           = context.get_launch_domain();
  size_t local_rank     = get_rank(domain, context.get_task_index());
  size_t num_ranks      = domain.get_volume();
  size_t num_sort_ranks = domain.hi()[domain.get_dim() - 1] - domain.lo()[domain.get_dim() - 1] + 1;

  bool is_index_space = !context.is_single_task() && context.communicators().size() > 0;
  assert(is_index_space || num_ranks == 1);

  SortArgs args{context.input(0),
                context.output(0),
                context.scalar(0).value<bool>(),  // argsort
                context.scalar(2).value<bool>(),  // stable
                segment_size_g,
                is_index_space,
                local_rank,
                num_ranks,
                num_sort_ranks};
  double_dispatch(
    args.input.dim(), args.input.code(), SortImpl<KIND>{}, args, context, context.communicators());
}

}  // namespace cupynumeric
