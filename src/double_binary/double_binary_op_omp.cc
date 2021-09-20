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

#include "double_binary/double_binary_op.h"
#include "double_binary_op_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <DoubleBinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct DoubleBinaryOpImplBody<VariantKind::OMP, OP_CODE, CODE, DIM> {
  using OP  = DoubleBinaryOp<OP_CODE, CODE>;
  using ARG = legate_type_of<CODE>;
  using RES = std::result_of_t<OP(ARG, ARG)>;

  void operator()(OP func,
                  AccessorWO<RES, DIM> out,
                  AccessorRW<RES, DIM> temp,
                  AccessorRO<ARG, DIM> in1,
                  AccessorRO<ARG, DIM> in2,
                  AccessorRO<ARG, DIM> in3,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto tempptr = temp.ptr(rect);
      auto in1ptr = in1.ptr(rect);
      auto in2ptr = in2.ptr(rect);
      auto in3ptr = in3.ptr(rect);
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) { 
          tempptr[idx] = func(in1ptr[idx], in2ptr[idx]);
          outptr[idx] = func(tempptr[idx], in3ptr[idx]);
          //outptr[idx] = func(func(in1ptr[idx], in2ptr[idx]), in3ptr[idx]);
      }
    } else {
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) {
        auto p = pitches.unflatten(idx, rect.lo);
        //out[p] = func(func(in1[p], in2[p]), in3[p]);
        temp[p] = func(in1[p], in2[p]);
        out[p] = func(temp[p], in3[p]);
      }
    }
  }
};

/*static*/ void DoubleBinaryOpTask::omp_variant(TaskContext& context)
{
  double_binary_op_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
