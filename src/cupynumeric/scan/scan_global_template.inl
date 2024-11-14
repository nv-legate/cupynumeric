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

#include "cupynumeric/scan/scan_util.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, ScanCode OP_CODE, Type::Code CODE, int DIM>
struct ScanGlobalImplBody;

template <VariantKind KIND, ScanCode OP_CODE>
struct ScanGlobalImpl {
  template <Type::Code CODE, int DIM>
  void operator()(ScanGlobalArgs& args) const
  {
    using OP  = ScanOp<OP_CODE, CODE>;
    using VAL = type_of<CODE>;

    auto out_rect      = args.out.shape<DIM>();
    auto sum_vals_rect = args.sum_vals.shape<DIM>();

    Pitches<DIM - 1> out_pitches;
    size_t volume = out_pitches.flatten(out_rect);
    Pitches<DIM - 1> sum_vals_pitches;

    if (volume == 0) {
      return;
    }

    auto out      = args.out.read_write_accessor<VAL, DIM>(out_rect);
    auto sum_vals = args.sum_vals.read_accessor<VAL, DIM>(sum_vals_rect);

    OP func;
    ScanGlobalImplBody<KIND, OP_CODE, CODE, DIM>()(func,
                                                   out,
                                                   sum_vals,
                                                   out_pitches,
                                                   out_rect,
                                                   sum_vals_pitches,
                                                   sum_vals_rect,
                                                   args.partition_index);
  }
};

template <VariantKind KIND>
struct ScanGlobalDispatch {
  template <ScanCode OP_CODE>
  void operator()(ScanGlobalArgs& args) const
  {
    return double_dispatch(args.out.dim(), args.out.code(), ScanGlobalImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void scan_global_template(TaskContext& context)
{
  auto task_index = context.get_task_index();
  ScanGlobalArgs args{
    context.input(1), context.output(0), context.scalar(0).value<ScanCode>(), task_index};
  op_dispatch(args.op_code, ScanGlobalDispatch<KIND>{}, args);
}

}  // namespace cupynumeric
