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

#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/pitches.h"

#include <algorithm>

namespace cupynumeric {

enum class PadMode : int32_t {
  CONSTANT = 0,
  EDGE     = 1,
};

struct PadArgs {
  legate::PhysicalStore output;          // Input already copied to center, fill padding regions
  legate::PhysicalStore constant_input;  // Broadcast constant value (for CONSTANT mode)
  PadMode mode;
  legate::Span<const std::pair<int64_t, int64_t>> pad_width;
  legate::Span<const int64_t> inner_shape;  // Unpadded input shape for center bounds
  int64_t constant_rows;
  int64_t constant_cols;
};

namespace detail {

template <int DIM>
inline void compute_center_bounds(const legate::Span<const std::pair<int64_t, int64_t>>& pad_width,
                                  const legate::Span<const int64_t>& inner_shape,
                                  legate::Point<DIM>& center_lo_global,
                                  legate::Point<DIM>& center_hi_global)
{
  // Convert pad widths and inner shape into the global coordinates of the
  // unpadded region. Each output point inside these bounds belongs to the
  // untouched center of the array.
  for (int i = 0; i < DIM; ++i) {
    const Legion::coord_t lo = static_cast<Legion::coord_t>(pad_width[i].first);
    center_lo_global[i]      = lo;
    center_hi_global[i]      = lo + static_cast<Legion::coord_t>(inner_shape[i]) - 1;
  }
}

template <int DIM>
inline bool intersect_center_with_tile(const legate::Point<DIM>& center_lo_global,
                                       const legate::Point<DIM>& center_hi_global,
                                       const legate::Rect<DIM>& output_rect,
                                       legate::Point<DIM>& center_lo_tile,
                                       legate::Point<DIM>& center_hi_tile)
{
  // Intersect the global center bounds with a tile's rectangle to determine
  // whether the tile contains any part of the original data. Tiles consisting
  // purely of padding return false.
  bool has_center = true;
  for (int i = 0; i < DIM; ++i) {
    center_lo_tile[i] = std::max(center_lo_global[i], output_rect.lo[i]);
    center_hi_tile[i] = std::min(center_hi_global[i], output_rect.hi[i]);
    if (center_lo_tile[i] > center_hi_tile[i]) {
      has_center = false;
    }
  }
  return has_center;
}

template <int DIM, typename VAL>
inline VAL fetch_constant_value(
  const legate::AccessorRO<VAL, 1>& const_acc, int64_t rows, int64_t cols, int axis, bool upper)
{
  // Interpret the flattened constant buffer (rows x cols) produced by Python.
  // The layout matches NumPy's semantics for scalar, per-axis, and before/after
  // pairs. When the metadata is unexpected fall back to the first element.
  if (rows == 1 && cols == 1) {
    // Scalar constant: same value for every side of every axis.
    return const_acc[0];
  }
  if (rows == 1 && cols == 2) {
    // One (before, after) pair broadcast across all axes.
    return const_acc[upper ? 1 : 0];
  }
  if (rows == DIM && cols == 1) {
    // One value per axis, shared between its lower and upper sides.
    return const_acc[axis];
  }
  if (rows == DIM && cols == 2) {
    // Per-axis (before, after) pairs; stride by cols to find the entry.
    const int64_t offset = static_cast<int64_t>(axis) * cols + (upper ? 1 : 0);
    return const_acc[offset];
  }
  return const_acc[0];
}

template <int DIM, typename VAL>
struct ConstantPadPointFunctor {
  legate::AccessorRW<VAL, DIM> output;
  legate::AccessorRO<VAL, 1> const_acc;
  legate::Point<DIM> center_lo_global;
  legate::Point<DIM> center_hi_global;
  int64_t rows;
  int64_t cols;
  Pitches<DIM - 1> pitches;
  legate::Point<DIM> output_lo;

  ConstantPadPointFunctor(legate::AccessorRW<VAL, DIM> output,
                          legate::AccessorRO<VAL, 1> const_acc,
                          const legate::Point<DIM>& center_lo_global,
                          const legate::Point<DIM>& center_hi_global,
                          int64_t rows,
                          int64_t cols,
                          const Pitches<DIM - 1>& pitches,
                          const legate::Point<DIM>& output_lo)
    : output(output),
      const_acc(const_acc),
      center_lo_global(center_lo_global),
      center_hi_global(center_hi_global),
      rows(rows),
      cols(cols),
      pitches(pitches),
      output_lo(output_lo)
  {
  }

  inline void operator()(size_t idx) const
  {
    auto point = pitches.unflatten(idx, output_lo);

    bool in_center = true;
    VAL fill_value{};
    for (int d = 0; d < DIM; ++d) {
      if (point[d] < center_lo_global[d]) {
        fill_value = fetch_constant_value<DIM, VAL>(const_acc, rows, cols, d, /*upper=*/false);
        in_center  = false;
      } else if (point[d] > center_hi_global[d]) {
        fill_value = fetch_constant_value<DIM, VAL>(const_acc, rows, cols, d, /*upper=*/true);
        in_center  = false;
      }
    }

    if (!in_center) {
      output[point] = fill_value;
    }
  }
};

template <int DIM, typename VAL>
struct EdgePadPointFunctor {
  legate::AccessorRW<VAL, DIM> output;
  bool has_center;
  legate::Point<DIM> center_lo_tile;
  legate::Point<DIM> center_hi_tile;
  Pitches<DIM - 1> pitches;
  legate::Point<DIM> output_lo;

  EdgePadPointFunctor(legate::AccessorRW<VAL, DIM> output,
                      bool has_center,
                      const legate::Point<DIM>& center_lo_tile,
                      const legate::Point<DIM>& center_hi_tile,
                      const Pitches<DIM - 1>& pitches,
                      const legate::Point<DIM>& output_lo)
    : output(output),
      has_center(has_center),
      center_lo_tile(center_lo_tile),
      center_hi_tile(center_hi_tile),
      pitches(pitches),
      output_lo(output_lo)
  {
  }

  inline void operator()(size_t idx) const
  {
    auto point = pitches.unflatten(idx, output_lo);

    bool in_center = has_center;
    if (has_center) {
      for (int d = 0; d < DIM; ++d) {
        if (point[d] < center_lo_tile[d] || point[d] > center_hi_tile[d]) {
          in_center = false;
          break;
        }
      }
    }

    if (!in_center) {
      legate::Point<DIM> edge_point = point;
      for (int d = 0; d < DIM; ++d) {
        if (edge_point[d] < center_lo_tile[d]) {
          edge_point[d] = center_lo_tile[d];
        } else if (edge_point[d] > center_hi_tile[d]) {
          edge_point[d] = center_hi_tile[d];
        }
      }
      output[point] = output[edge_point];
    }
  }
};

}  // namespace detail

class PadTask : public CuPyNumericTask<PadTask> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_PAD}};

 public:
  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace cupynumeric
