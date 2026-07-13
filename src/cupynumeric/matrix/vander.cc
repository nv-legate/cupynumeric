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

#include "cupynumeric/matrix/vander.h"
#include "cupynumeric/matrix/vander_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct VanderImplBody<VariantKind::CPU, CODE> {
  using VAL = type_of<CODE>;

  TaskContext context;
  explicit VanderImplBody(TaskContext context) : context(context) {}

  void operator()(const AccessorWO<VAL, 2>& out,
                  const AccessorRO<VAL, 2>& in,
                  const Pitches<1>& pitches,
                  const Point<2>& lo,
                  size_t volume,
                  int64_t N,
                  bool increasing) const
  {
    for (size_t idx = 0; idx < volume; ++idx) {
      auto p              = pitches.unflatten(idx, lo);
      const int64_t power = increasing ? static_cast<int64_t>(p[1]) : (N - 1 - p[1]);
      out[p]              = vander_pow<VAL>(in[p], power);
    }
  }
};

/*static*/ void VanderTask::cpu_variant(TaskContext context)
{
  vander_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  VanderTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
