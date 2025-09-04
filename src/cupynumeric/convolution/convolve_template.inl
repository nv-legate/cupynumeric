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
#include "cupynumeric/convolution/convolve.h"
#include "cupynumeric/pitches.h"

#include <map>

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE, int DIM>
struct ConvolveImplBody;

template <VariantKind KIND>
struct ConvolveImpl {
  TaskContext context;
  explicit ConvolveImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM, std::enable_if_t<(DIM <= 3)>* = nullptr>
  void operator()(ConvolveArgs& args) const
  {
    using VAL        = type_of<CODE>;
    auto subrect     = args.out.shape<DIM>();
    auto filter_rect = args.filter.shape<DIM>();

    if (subrect.empty()) {
      return;
    }

    auto input_subrect = subrect;
    for (uint32_t idx = 1; idx < args.inputs.size(); ++idx) {
      auto image_subrect = args.inputs[idx].shape<DIM>();
      input_subrect      = input_subrect.union_bbox(image_subrect);
    }

    auto out    = args.out.write_accessor<VAL, DIM>(subrect);
    auto filter = args.filter.read_accessor<VAL, DIM>(filter_rect);
    // This is valid only because we colocate all shifted images with the main tile
    auto input = args.inputs[0].read_accessor<VAL, DIM>(input_subrect);

    Rect<DIM> root_rect(args.root_domain);
    ConvolveImplBody<KIND, CODE, DIM>{context}(
      out, filter, input, root_rect, subrect, filter_rect, args.method);
  }

  template <Type::Code CODE, int DIM, std::enable_if_t<!(DIM <= 3)>* = nullptr>
  void operator()(ConvolveArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void convolve_template(TaskContext& context)
{
  ConvolveArgs args;

  auto inputs  = context.inputs();
  auto outputs = context.outputs();

  args.out    = std::move(outputs[0]);
  args.filter = std::move(inputs[0]);
  for (uint32_t idx = 1; idx < inputs.size(); ++idx) {
    args.inputs.push_back(std::move(inputs[idx]));
  }

  auto shape           = context.scalar(0).value<DomainPoint>();
  args.root_domain.dim = shape.dim;
  for (int32_t dim = 0; dim < shape.dim; ++dim) {
    args.root_domain.rect_data[dim]             = 0;
    args.root_domain.rect_data[dim + shape.dim] = shape[dim] - 1;
  }

  args.method = static_cast<CuPyNumericConvolveMethod>(context.scalar(1).value<std::int32_t>());

  double_dispatch(args.out.dim(), args.out.code(), ConvolveImpl<KIND>{context}, args);
}

template <typename VAL, int DIM>
static unsigned compute_output_tile(Point<DIM>& tile,
                                    const Point<DIM>& bounds,
                                    const size_t min_last_elements,
                                    const size_t target_volume)
{
  for (int d = 0; d < DIM; d++) {
    tile[d] = 1;
  }
  // Better be a powers of 2
  assert((min_last_elements & (min_last_elements - 1)) == 0);
  assert((target_volume & (target_volume - 1)) == 0);
  unsigned volume = 1;
  // Try to make the last dimension at least the min last elements for locality
  for (uint32_t idx = 1; idx < min_last_elements; idx *= 2) {
    tile[DIM - 1] *= 2;
    if (bounds[DIM - 1] < tile[DIM - 1]) {
      tile[DIM - 1] /= 2;
      break;
    } else {
      volume *= 2;
    }
    if (volume == target_volume) {
      break;
    }
  }
  int last_dim = DIM - 1;
  // Round-robin powers of 2 onto the other dimensions until
  // we hit the max or get all the dimensions balanced
  if (DIM > 1) {
    for (uint32_t idx = 1; idx < min_last_elements; idx *= 2) {
      for (int d = DIM - 2; d >= 0; d--) {
        tile[d] *= 2;
        if (bounds[d] < tile[d]) {
          tile[d] /= 2;
        } else {
          volume *= 2;
          last_dim = d;
          if (volume == target_volume) {
            break;
          }
        }
      }
      if (volume == target_volume) {
        break;
      }
    }
  }
  // If we still have more to go round-robin powers of 2 over
  // all the dimensions
  int unchanged = 0;
  while (volume < target_volume) {
    if (last_dim == 0) {
      last_dim = DIM - 1;
    } else {
      last_dim--;
    }
    tile[last_dim] *= 2;
    if (bounds[last_dim] < tile[last_dim]) {
      tile[last_dim] /= 2;
      unchanged++;
      if (unchanged == DIM) {
        break;
      }
    } else {
      volume *= 2;
      unchanged = 0;
    }
  }
  return volume;
}

template <typename VAL, int DIM>
static unsigned compute_filter_tile(Point<DIM>& tile,
                                    const Point<DIM>& bounds,
                                    const Point<DIM>& output,
                                    const size_t max_size)
{
  for (int d = 0; d < DIM; d++) {
    tile[d] = 1;
  }
  // Try doubling dimensions until we can't make it any larger
  unsigned result = 0;
  bool done       = false;
  while (!done) {
    done = true;
    for (int d = DIM - 1; d >= 0; d--) {
      // Skip if it will exceed our bounds
      if (bounds[d] < (2 * tile[d])) {
        continue;
      }
      unsigned filter_size = sizeof(VAL);
      tile[d] *= 2;
      for (int d2 = 0; d2 < DIM; d2++) {
        filter_size *= tile[d2];
      }
      unsigned input_size = sizeof(VAL);
      for (int d2 = 0; d2 < DIM; d2++) {
        input_size *= (output[d2] + 2 * (tile[d2] / 2));
      }
      unsigned total_size = filter_size + input_size;
      if (total_size <= max_size) {
        result = total_size;
        done   = false;
      } else {
        tile[d] /= 2;
      }
    }
  }
  // Then try incrementally increasing dimensions until we can't
  // make it any larger
  done = false;
  while (!done) {
    done = true;
    for (int d = DIM - 1; d >= 0; d--) {
      // Skip if it will exceed our bounds
      if (bounds[d] == tile[d]) {
        continue;
      }
      unsigned filter_size = sizeof(VAL);
      tile[d]++;
      for (int d2 = 0; d2 < DIM; d2++) {
        filter_size *= tile[d2];
      }
      unsigned input_size = sizeof(VAL);
      for (int d2 = 0; d2 < DIM; d2++) {
        input_size *= (output[d2] + 2 * (tile[d2] / 2));
      }
      unsigned total_size = filter_size + input_size;
      if (total_size <= max_size) {
        result = total_size;
        done   = false;
      } else {
        tile[d]--;
      }
    }
  }
  return result;
}

template <typename VAL, int DIM>
static unsigned roundup_tile(Point<DIM>& tile,
                             const Point<DIM>& bounds,
                             const Point<DIM>& padding,
                             const unsigned max_size)
{
  if (DIM == 1) {
    // In this single case we can just solve for this directly
    unsigned elements = max_size / sizeof(VAL);
    assert(elements > padding[0]);
    if (tile[0] < (elements - padding[0])) {
      tile[0] = elements - padding[0];
      if (bounds[0] < tile[0]) {
        tile[0] = bounds[0];
      }
    }
    return (tile[0] + padding[0]) * sizeof(VAL);
  } else {
    // Compute the initial size
    // Shrink the tile to the bounds if necessary
    unsigned result = sizeof(VAL);
    for (int d = 0; d < DIM; d++) {
      if (bounds[d] < tile[d]) {
        tile[d] = bounds[d];
      }
      result *= std::max(static_cast<coord_t>(1), tile[d] + padding[d]);
    }
    // Find the two smallest dimensions and increase one of them
    // until we hit the second smallest one or exceed max_smem_size
    unsigned skipdims = 0;
    bool all_same     = true;
    while (true) {
      int d1 = DIM - 1, d2 = -1;
      int t1 = tile[d1], t2 = 0;
      while (t1 == bounds[d1]) {
        skipdims |= (1 << d1);
        if (--d1 < 0) {
          return result;  // all dims at their bounds so we're done
        }
        t1 = tile[d1];
      }
      for (int d = d1 - 1; d >= 0; d--) {
        if (skipdims & (1 << d)) {
          continue;
        }
        // Skip any dimension that is at its bound
        if (tile[d] == bounds[d]) {
          skipdims |= (1 << d);
          continue;
        }
        if (tile[d] < t1) {
          d2 = d1;
          t2 = t1;
          d1 = d;
          t1 = tile[d];
        } else if ((d2 < 0) || (tile[d] < t2)) {
          d2 = d;
          t2 = tile[d];
        }
      }
      if (d2 == -1) {
        // All the other dimensions are at their bounds, check that
        // the last dimension is also at its bound if not solve
        unsigned pitch = sizeof(VAL);
        for (int d = 0; d < DIM; d++) {
          if (d != d1) {
            pitch *= std::max(static_cast<coord_t>(1), tile[d] + padding[d]);
          }
        }
        // Make sure the last dimension is as large as it can go too
        if (tile[d1] < bounds[d1]) {
          unsigned elements = max_size / pitch;
          assert(elements > padding[d1]);
          assert(tile[d1] < (elements - padding[d1]));
          tile[d1] = elements - padding[d1];
          if (bounds[d1] < tile[d1]) {
            tile[d1] = bounds[d1];
          }
        }
        return pitch * std::max(static_cast<coord_t>(1), tile[d1] + padding[d1]);
      }
      // If we ever get two dimensions of the same size then see what dimension
      // has the next largest value. If we can't find one that is larger then
      // we know that there is no smallest dimension so we can march all the
      // dimensions together at this point
      if (t1 == t2) {
        d2 = -1;
        for (int d = 0; d < DIM; d++) {
          if (d == d1) {
            continue;
          }
          if (tile[d] <= tile[d1]) {
            continue;
          }
          if ((d2 == -1) || (tile[d] < tile[d2])) {
            d2 = d;
            t2 = tile[d];
          }
        }
        if (d2 == -1) {
          break;
        }
      }
      // Solve for the max we can walk
      unsigned pitch = sizeof(VAL);
      for (int d = 0; d < DIM; d++) {
        if (d != d1) {
          pitch *= std::max(static_cast<coord_t>(1), tile[d] + padding[d]);
        }
      }
      unsigned elements = max_size / pitch;
      if ((elements <= padding[d1]) || (t1 >= (elements - padding[d1]))) {
        skipdims |= (1 << d1);
        continue;
      }
      int bound = elements - padding[d1];
      if (bounds[d1] < bound) {
        tile[d1] = bounds[d1];
        result   = pitch * std::max(static_cast<coord_t>(1), tile[d1] + padding[d1]);
      } else if (bound < t2) {
        tile[d1] = bound;
        result   = pitch * std::max(static_cast<coord_t>(1), bound + padding[d1]);
        all_same = false;
        break;
      } else {
        tile[d1] = t2;
        result   = pitch * std::max(static_cast<coord_t>(1), t2 + padding[d1]);
      }
    }
    if (all_same) {
      // Step all the dimensions together until we hit
      // the shared memory upper bound we're targetting
      // This algorithm is in theory slow, but the max
      // memory sizes of caches are "small" and the amount
      // of memory will grow polynomially in the number
      // of dimensions so it should converge quickly
      while (true) {
        unsigned next_size = sizeof(VAL);
        for (int d = 0; d < DIM; d++) {
          if (skipdims & (1 << d)) {
            next_size *= std::max(static_cast<coord_t>(1), tile[d] + padding[d]);
          } else if (tile[d] == bounds[d]) {
            next_size *= std::max(static_cast<coord_t>(1), tile[d] + padding[d]);
            skipdims |= (1 << d);
          } else {
            next_size *= (tile[d] + 1 + padding[d]);
          }
        }
        if ((next_size > max_size) || (next_size == result)) {
          break;
        }
        result = next_size;
        for (int d = 0; d < DIM; d++) {
          if (skipdims & (1 << d)) {
            continue;
          }
          tile[d]++;
        }
      }
    }
    return result;
  }
}

}  // namespace cupynumeric
