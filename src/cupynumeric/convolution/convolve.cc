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

#include "cupynumeric/divmod.h"
#include "cupynumeric/convolution/convolve.h"
#include "cupynumeric/convolution/convolve_template.inl"

namespace cupynumeric {

// This is the easy to understand functional specification of the
// algorithm, but it is commented out in favor of the faster one
// that is blocked for caches
#if 0
template <Type::Code CODE, int DIM>
struct ConvolveImplBody<VariantKind::CPU, CODE, DIM> {
  TaskContext context;
  explicit ConvolveImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> filter,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& root_rect,
                  const Rect<DIM>& subrect,
                  const Rect<DIM>& filter_rect) const
  {
    const Point<DIM> one = Point<DIM>::ONES();
    Point<DIM> extents = filter_rect.hi + one;
    Point<DIM> centers;
    for (int d = 0; d < DIM; d++)
      centers[d] = extents[d] / 2;
    Point<DIM> output = subrect.lo;
    const size_t output_volume = subrect.volume();
    const size_t filter_volume = filter_rect.volume();
    for (unsigned p = 0; p < output_volume; p++) {
      VAL acc{0};
      Point<DIM> filter_point = filter_rect.lo;
      for (unsigned f = 0; f < filter_volume; f++) {
        Point<DIM> input = output + extents - filter_point - one - centers;
        if (root_rect.contains(input))
          acc += in[input] * filter[filter_point];
        // Step to the next filter point
        for (int d = DIM-1; d >= 0; d--) {
          filter_point[d]++;
          if (filter_rect.hi[d] < filter_point[d])
            filter_point[d] = filter_rect.lo[d];
          else
            break;
        }
      }
      out[output] = acc;
      // Step to the next point
      for (int d = DIM-1; d >= 0; d--) {
        output[d]++;
        if (subrect.hi[d] < output[d])
          output[d] = subrect.lo[d];
        else
          break;
      }
    }
  }
};
#endif

template <Type::Code CODE, int DIM>
struct ConvolveImplBody<VariantKind::CPU, CODE, DIM> {
  TaskContext context;
  explicit ConvolveImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> filter,
                  AccessorRO<VAL, DIM> in,
                  const Rect<DIM>& root_rect,
                  const Rect<DIM>& subrect,
                  const Rect<DIM>& filter_rect,
                  CuPyNumericConvolveMethod method) const
  {
    const Point<DIM> one = Point<DIM>::ONES();
    Point<DIM> extents   = filter_rect.hi - filter_rect.lo + one;
    Point<DIM> centers;
    for (int d = 0; d < DIM; d++) {
      centers[d] = extents[d] / 2;
    }

    // Compute the tiles for the L2 cache
    Point<DIM> l2_output_tile, l2_filter_tile;
    const Point<DIM> output_bounds = subrect.hi - subrect.lo + one;
    // Try to fit the output in 1/4 of the L2 cache and
    // and the input and filter in the other 3/4
    compute_output_tile<VAL, DIM>(l2_output_tile,
                                  output_bounds,
                                  CACHE_LINE_SIZE / sizeof(VAL),
                                  L2_CACHE_SIZE / sizeof(VAL) / 4);
    const Point<DIM> filter_bounds = filter_rect.hi - filter_rect.lo + one;
    compute_filter_tile<VAL, DIM>(
      l2_filter_tile, filter_bounds, l2_output_tile, 3 * L2_CACHE_SIZE / 4);
    unsigned total_l2_filters = 1;
    for (int d = 0; d < DIM; d++) {
      total_l2_filters *= ((extents[d] + l2_filter_tile[d] - 1) / l2_filter_tile[d]);
    }
    unsigned total_l2_outputs = 1;
    for (int d = 0; d < DIM; d++) {
      total_l2_outputs *= ((output_bounds[d] + l2_output_tile[d] - 1) / l2_output_tile[d]);
    }

    // Compute the tiles for the L1 cache
    Point<DIM> l1_output_tile, l1_filter_tile;
    // Try to fit the output in the 1/4 of the L1 cache and
    // the filter and input in the other 3/4 of the L1 cache
    compute_output_tile<VAL, DIM>(l1_output_tile,
                                  output_bounds,
                                  CACHE_LINE_SIZE / sizeof(VAL),
                                  L1_CACHE_SIZE / sizeof(VAL) / 4);
    compute_filter_tile<VAL, DIM>(
      l1_filter_tile, filter_bounds, l1_output_tile, 3 * L1_CACHE_SIZE / 4);
    unsigned total_l1_filters = 1;
    for (int d = 0; d < DIM; d++) {
      total_l1_filters *= ((l2_filter_tile[d] + l1_filter_tile[d] - 1) / l1_filter_tile[d]);
    }
    unsigned total_l1_outputs = 1;
    for (int d = 0; d < DIM; d++) {
      total_l1_outputs *= ((l2_output_tile[d] + l1_output_tile[d] - 1) / l1_output_tile[d]);
    }

    // Zero out the output data since we're going to be doing sum accumulations
    Point<DIM> output         = subrect.lo;
    const size_t total_points = subrect.volume();
    for (size_t p = 0; p < total_points; p++) {
      out[output] = VAL{0};
      for (int d = DIM - 1; d >= 0; d--) {
        output[d]++;
        if (subrect.hi[d] < output[d]) {
          output[d] = subrect.lo[d];
        } else {
          break;
        }
      }
    }

    // Iterate over the L2 filter tiles
    Point<DIM> l2_filter = filter_rect.lo;
    for (unsigned l2_fidx = 0; l2_fidx < total_l2_filters; l2_fidx++) {
      Rect<DIM> l2_filter_rect(l2_filter, l2_filter + l2_filter_tile - one);
      unsigned local_l1_filters = total_l1_filters;
      // Make sure we don't overflow our boundaries
      if (!filter_rect.contains(l2_filter_rect)) {
        l2_filter_rect   = filter_rect.intersection(l2_filter_rect);
        local_l1_filters = 1;
        for (int d = 0; d < DIM; d++) {
          local_l1_filters *=
            ((l2_filter_rect.hi[d] - l2_filter_rect.lo[d] + l1_filter_tile[d]) / l1_filter_tile[d]);
        }
      }
      // Now iterate the tiles for the L2 outputs
      Point<DIM> l2_output = subrect.lo;
      for (unsigned l2_outidx = 0; l2_outidx < total_l2_outputs; l2_outidx++) {
        Rect<DIM> l2_output_rect(l2_output, l2_output + l2_output_tile - one);
        unsigned local_l1_outputs = total_l1_outputs;
        if (!subrect.contains(l2_output_rect)) {
          l2_output_rect   = subrect.intersection(l2_output_rect);
          local_l1_outputs = 1;
          for (int d = 0; d < DIM; d++) {
            local_l1_outputs *= ((l2_output_rect.hi[d] - l2_output_rect.lo[d] + l1_output_tile[d]) /
                                 l1_output_tile[d]);
          }
        }
        // Do a quick check here to see if all the inputs are contained for
        // this particular tile
        Rect<DIM> l2_input_rect(l2_output_rect.lo + extents - l2_filter_rect.hi - one - centers,
                                l2_output_rect.hi + extents - l2_filter_rect.lo - one - centers);
        const bool input_contained = root_rect.contains(l2_input_rect);
        // Iterate the L1 output tiles this output rect
        Point<DIM> l1_output = l2_output;
        for (unsigned l1_outidx = 0; l1_outidx < local_l1_outputs; l1_outidx++) {
          Rect<DIM> l1_output_rect(l1_output, l1_output + l1_output_tile - one);
          l1_output_rect = l2_output_rect.intersection(l1_output_rect);
          // Iterate the L1 filters for this L1 output rect
          Point<DIM> l1_filter = l2_filter;
          for (unsigned l1_fidx = 0; l1_fidx < local_l1_filters; l1_fidx++) {
            Rect<DIM> l1_filter_rect(l1_filter, l1_filter + l1_filter_tile - one);
            l1_filter_rect                  = l2_filter_rect.intersection(l1_filter_rect);
            const unsigned l1_filter_points = l1_filter_rect.volume();
            // Now we can iterate all the points in the output volume and
            // compute the their partial accumulations to the output value
            const unsigned l1_output_points = l1_output_rect.volume();
            output                          = l1_output_rect.lo;
            for (unsigned pidx = 0; pidx < l1_output_points; pidx++) {
              VAL acc{0};
              Point<DIM> filter_point = l1_filter_rect.lo;
              for (unsigned fidx = 0; fidx < l1_filter_points; fidx++) {
                Point<DIM> input = output + extents - filter_point - one - centers;
                if (input_contained || root_rect.contains(input)) {
                  acc += in[input] * filter[filter_point];
                }
                // Step to the next filter point
                for (int d = DIM - 1; d >= 0; d--) {
                  filter_point[d]++;
                  if (l1_filter_rect.hi[d] < filter_point[d]) {
                    filter_point[d] = l1_filter_rect.lo[d];
                  } else {
                    break;
                  }
                }
              }
              out[output] += acc;
              // Step to the next output point
              for (int d = DIM - 1; d >= 0; d--) {
                output[d]++;
                if (l1_output_rect.hi[d] < output[d]) {
                  output[d] = l1_output_rect.lo[d];
                } else {
                  break;
                }
              }
            }
            // Step to the next L1 filter
            for (int d = DIM - 1; d >= 0; d--) {
              l1_filter[d] += l1_filter_tile[d];
              if (l2_filter_rect.hi[d] < l1_filter[d]) {
                l1_filter[d] = l2_filter_rect.lo[d];
              } else {
                break;
              }
            }
          }
          // Step to the next L1 output tile
          for (int d = DIM - 1; d >= 0; d--) {
            l1_output[d] += l1_output_tile[d];
            if (l2_output_rect.hi[d] < l1_output[d]) {
              l1_output[d] = l2_output_rect.lo[d];
            } else {
              break;
            }
          }
        }
        // Step to the next output tile
        for (int d = DIM - 1; d >= 0; d--) {
          l2_output[d] += l2_output_tile[d];
          if (subrect.hi[d] < l2_output[d]) {
            l2_output[d] = subrect.lo[d];
          } else {
            break;
          }
        }
      }
      // Step to the next l2 filter
      for (int d = DIM - 1; d >= 0; d--) {
        l2_filter[d] += l2_filter_tile[d];
        if (filter_rect.hi[d] < l2_filter[d]) {
          l2_filter[d] = filter_rect.lo[d];
        } else {
          break;
        }
      }
    }
  }
};

/*static*/ void ConvolveTask::cpu_variant(TaskContext context)
{
  convolve_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  ConvolveTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
