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

#include "cupynumeric/index/repeat.h"
#include "cupynumeric/index/repeat_template.inl"

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE, int DIM>
struct RepeatImplBody<VariantKind::CPU, CODE, DIM> {
  TaskContext context;
  explicit RepeatImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, DIM>& in,
                  const int64_t repeats,
                  const int32_t axis,
                  const Rect<DIM>& in_rect) const
  {
    auto out_rect = out_array.shape<DIM>();
    auto out      = out_array.write_accessor<VAL, DIM>(out_rect);
    Pitches<DIM - 1> pitches;

    auto out_volume = pitches.flatten(out_rect);
    for (size_t idx = 0; idx < out_volume; ++idx) {
      auto out_p = pitches.unflatten(idx, out_rect.lo);
      auto in_p  = out_p;
      in_p[axis] /= repeats;
      out[out_p] = in[in_p];
    }
  }

  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, 1>& in,
                  const AccessorRO<int64_t, 1>& repeats,
                  const int32_t axis,
                  const Rect<1>& in_rect) const
  {
    Pitches<0> in_pitches;
    auto volume = in_pitches.flatten(in_rect);
    Point<1> extents(0);
    for (size_t idx = 0; idx < volume; ++idx) {
      auto point = in_pitches.unflatten(idx, in_rect.lo);
      extents[0] += repeats[point];
    }

    auto out = out_array.create_output_buffer<VAL, 1>(extents, true);

    int64_t out_idx = 0;
    for (size_t in_idx = 0; in_idx < volume; ++in_idx) {
      auto p = in_pitches.unflatten(in_idx, in_rect.lo);
      for (int64_t r = 0; r < repeats[p]; r++) {
        out[out_idx++] = in[p];
      }
    }
  }

  template <int32_t _DIM = DIM, std::enable_if_t<(_DIM > 1)>* = nullptr>
  void operator()(legate::PhysicalStore& out_array,
                  const AccessorRO<VAL, _DIM>& in,
                  const AccessorRO<int64_t, _DIM>& repeats,
                  const int32_t axis,
                  const Rect<_DIM>& in_rect) const
  {
    int64_t sum  = 0;
    auto p       = in_rect.lo;
    auto offsets = create_buffer<int64_t>(in_rect.hi[axis] - in_rect.lo[axis] + 1);

    int64_t off_idx = 0;
    for (int64_t idx = in_rect.lo[axis]; idx <= in_rect.hi[axis]; ++idx) {
      p[axis]            = idx;
      offsets[off_idx++] = sum;
      sum += repeats[p];
    }

    Point<DIM> extents = in_rect.hi - in_rect.lo + Point<DIM>::ONES();
    extents[axis]      = sum;

    auto out = out_array.create_output_buffer<VAL, DIM>(extents, true);

    Pitches<DIM - 1> in_pitches;
    auto in_volume = in_pitches.flatten(in_rect);

    int64_t axis_base = in_rect.lo[axis];
    for (size_t idx = 0; idx < in_volume; ++idx) {
      auto in_p  = in_pitches.unflatten(idx, in_rect.lo);
      auto out_p = in_p - in_rect.lo;

      int64_t off_start = offsets[in_p[axis] - axis_base];
      int64_t off_end   = off_start + repeats[in_p];

      auto in_v = in[in_p];
      for (int64_t out_idx = off_start; out_idx < off_end; ++out_idx) {
        out_p[axis] = out_idx;
        out[out_p]  = in_v;
      }
    }
  }
};

/*static*/ void RepeatTask::cpu_variant(TaskContext context)
{
  repeat_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  RepeatTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
