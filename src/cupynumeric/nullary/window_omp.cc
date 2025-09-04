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

#include "cupynumeric/nullary/window.h"
#include "cupynumeric/nullary/window_template.inl"

namespace cupynumeric {

using namespace legate;

template <WindowOpCode OP_CODE>
struct WindowImplBody<VariantKind::OMP, OP_CODE> {
  TaskContext context;
  explicit WindowImplBody(TaskContext context) : context(context) {}

  void operator()(
    AccessorWO<double, 1> out, const Rect<1>& rect, bool dense, int64_t M, double beta) const
  {
    WindowOp<OP_CODE> gen(M, beta);
    if (dense) {
      auto* outptr = out.ptr(rect);
      auto base    = rect.lo[0];
#pragma omp parallel for schedule(static)
      for (int64_t idx = base; idx <= rect.hi[0]; ++idx) {
        outptr[idx - base] = gen(idx);
      }
    } else
#pragma omp parallel for schedule(static)
      for (int64_t idx = rect.lo[0]; idx <= rect.hi[0]; ++idx) {
        out[idx] = gen(idx);
      }
  }
};

/*static*/ void WindowTask::omp_variant(TaskContext context)
{
  window_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
