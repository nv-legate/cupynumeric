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

#include "cupynumeric/index/pad.h"
#include "cupynumeric/index/pad_template.inl"

#include <algorithm>
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/utilities/thrust_util.h"

namespace cupynumeric {

using namespace legate;

// Kernel for CONSTANT mode - extracts value from accessor
template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  pad_constant_kernel(AccessorRW<VAL, DIM> output,
                      const Point<DIM> center_lo,
                      const Point<DIM> center_hi,
                      const Point<DIM> output_lo,
                      const Pitches<DIM - 1> pitches,
                      const size_t volume,
                      const AccessorRO<VAL, 1> const_acc,
                      const int64_t rows,
                      const int64_t cols)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }

  auto out_p = pitches.unflatten(idx, output_lo);

  bool in_center = true;
  VAL fill_value{};
  for (int d = 0; d < DIM; ++d) {
    if (out_p[d] < center_lo[d]) {
      // Lower padding for axis d: choose the "before" value according to metadata.
      if (rows == 1 && cols == 1) {
        // Scalar constant shared by all sides.
        fill_value = const_acc[0];
      } else if (rows == 1 && cols == 2) {
        // Single (before, after) pair broadcast to every axis; index 0 is the lower side.
        fill_value = const_acc[0];
      } else if (rows == DIM && cols == 1) {
        // Per-axis value applied to both sides of that axis.
        fill_value = const_acc[d];
      } else if (rows == DIM && cols == 2) {
        // Per-axis before/after pairs; stride by cols to reach the current axis.
        const int64_t offset = static_cast<int64_t>(d) * cols;
        fill_value           = const_acc[offset];
      } else {
        // Metadata unexpectedly shaped; fall back to the first element.
        fill_value = const_acc[0];
      }
      in_center = false;
    } else if (out_p[d] > center_hi[d]) {
      // Upper padding for axis d: pick the "after" value when available.
      if (rows == 1 && cols == 1) {
        fill_value = const_acc[0];
      } else if (rows == 1 && cols == 2) {
        // Index 1 corresponds to the upper side of the shared pair.
        fill_value = const_acc[1];
      } else if (rows == DIM && cols == 1) {
        fill_value = const_acc[d];
      } else if (rows == DIM && cols == 2) {
        // Offset plus one jumps to the "after" entry for this axis.
        const int64_t offset = static_cast<int64_t>(d) * cols + 1;
        fill_value           = const_acc[offset];
      } else {
        fill_value = const_acc[0];
      }
      in_center = false;
    }
  }

  if (!in_center) {
    output[out_p] = fill_value;
  }
}

// Kernel for EDGE mode - replicates edges from center
template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  pad_edge_kernel(AccessorRW<VAL, DIM> output,
                  const Point<DIM> center_lo,
                  const Point<DIM> center_hi,
                  const Point<DIM> output_lo,
                  const Pitches<DIM - 1> pitches,
                  const size_t volume)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }

  auto out_p = pitches.unflatten(idx, output_lo);

  // Check if in center
  bool in_center = true;
  for (int d = 0; d < DIM; ++d) {
    if (out_p[d] < center_lo[d] || out_p[d] > center_hi[d]) {
      in_center = false;
      break;
    }
  }

  if (!in_center) {
    // Map to nearest edge of center
    Point<DIM> edge_p = out_p;
    for (int d = 0; d < DIM; ++d) {
      if (edge_p[d] < center_lo[d]) {
        edge_p[d] = center_lo[d];
      } else if (edge_p[d] > center_hi[d]) {
        edge_p[d] = center_hi[d];
      }
    }
    output[out_p] = output[edge_p];
  }
}

template <Type::Code CODE, int DIM>
struct PadImplBody<VariantKind::GPU, CODE, DIM> {
  TaskContext context;
  explicit PadImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  // Overload for CONSTANT mode
  void operator()(AccessorRW<VAL, DIM>& output,
                  PadMode mode,
                  Span<const std::pair<int64_t, int64_t>> pad_width,
                  Span<const int64_t> inner_shape,
                  int64_t constant_rows,
                  int64_t constant_cols,
                  const AccessorRO<VAL, 1>& const_acc,
                  const Rect<DIM>& output_rect) const
  {
    (void)mode;
    Point<DIM> center_lo_global, center_hi_global;
    for (int i = 0; i < DIM; ++i) {
      center_lo_global[i] = static_cast<Legion::coord_t>(pad_width[i].first);
      center_hi_global[i] = center_lo_global[i] + static_cast<Legion::coord_t>(inner_shape[i]) - 1;
    }

    Pitches<DIM - 1> pitches;
    auto volume         = pitches.flatten(output_rect);
    auto stream         = context.get_task_stream();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // CONSTANT mode - pass accessor to kernel
    pad_constant_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(output,
                                                                  center_lo_global,
                                                                  center_hi_global,
                                                                  output_rect.lo,
                                                                  pitches,
                                                                  volume,
                                                                  const_acc,
                                                                  constant_rows,
                                                                  constant_cols);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }

  // Overload for EDGE mode
  void operator()(AccessorRW<VAL, DIM>& output,
                  PadMode mode,
                  Span<const std::pair<int64_t, int64_t>> pad_width,
                  Span<const int64_t> inner_shape,
                  const Rect<DIM>& output_rect) const
  {
    // Calculate center region in GLOBAL coordinates and intersect with tile
    Point<DIM> center_lo, center_hi;
    for (int i = 0; i < DIM; ++i) {
      const Legion::coord_t center_lo_global = static_cast<Legion::coord_t>(pad_width[i].first);
      const Legion::coord_t center_hi_global =
        center_lo_global + static_cast<Legion::coord_t>(inner_shape[i]) - 1;

      center_lo[i] = std::max(center_lo_global, output_rect.lo[i]);
      center_hi[i] = std::min(center_hi_global, output_rect.hi[i]);
    }

    Pitches<DIM - 1> pitches;
    auto volume         = pitches.flatten(output_rect);
    auto stream         = context.get_task_stream();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // EDGE mode - kernel reads from center
    pad_edge_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      output, center_lo, center_hi, output_rect.lo, pitches, volume);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void PadTask::gpu_variant(TaskContext context)
{
  pad_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
