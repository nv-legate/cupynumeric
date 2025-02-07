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
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/convolution/convolve.h"
#include "cupynumeric/convolution/convolve_template.inl"

namespace cupynumeric {

using namespace legate;

////////////////////////////////////
// Direct convolution implementation
////////////////////////////////////

// Convolution should be able to hit FMA throughput limits
// on the GPU due to the amount of FLOPs needed to be performed
// given the amount of data loaded. This is especially true of
// larger convolution filters. In order to hit these limits though
// we need to make sure that the GPU is fed data appropriately.
// We have two different kernels to handle different sized filters.

// Small Tile Case
// In the small tile case, a reasonable tile input including the
// all the boundary values for a given filter tile can fit in the
// shared memory of the SM, allowing the threadblock to fully
// compute an entire tile of output points in a single pass.
// If the tile is small enough, we even try to get multiple CTAs/SM
// in order to better pipeline data loading with compute.

// Large Tile Case
// For inputs where the filter is very large and it is impossible
// to fit a reasonable sized tile into shared memory, we tile both
// the output and the filter and make multiple passes over the data
// to create reasonable sized input tiles that fit in shared memory.
// If possible we also attempt to tile for the L2 cache as well so
// that threadblocks walking through memory together can hopefully
// hit in the L2 more often than not when loading data

template <int DIM>
struct ConvolutionInitArgs {
 public:
  FastDivmodU64 pitches[DIM];
};

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 4)
  convolution_init(const AccessorWO<VAL, DIM> out,
                   const Point<DIM> subrect_lo,
                   const ConvolutionInitArgs<DIM> args,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  Point<DIM> point = subrect_lo;
#pragma unroll
  for (int d = 0; d < DIM; d++) {
    point[d] += args.pitches[d].divmod(offset, offset);
  }
  out[point] = VAL{0};
}

// We want to run the convolution kernel with as large a shared memory
// tile as possible to avoid duplicate loading of data and maximize
// compute intensity. Therefore we're always going to run with 1 CTA
// per SM, but we still want enough thread-level parallelism, so we
// set this to the maximum number of warps in a threadblock
// Note that a lot of this code assumes this is a power of 2
#define CONVOLUTION_THREADS 1024
// The idea behind THREAD_OUTPUTS is to figure out how many registers
// we will be willing to assign to hold the partial output accumulations
// in each thread without using too many registers. Every GPU (with one
// exception) has 64K 32-bit registers per SM. We key off that and want
// to allocate a quarter of those registers for holding the partial accumulations
// We assume here that sizeof(VAL) is a power of 2
#define THREAD_OUTPUTS(TYPE) 1
//((4/*bytes/reg*/ * ((65536/8)/CONVOLUTION_THREADS)/*regs/thread*/) / sizeof(TYPE))

template <int DIM, int POINTS>
struct ConvolutionLargeTileArgs {
  FastDivmod l1_output_tile_pitches[DIM];
  FastDivmod l1_input_pitches[DIM];
  FastDivmod l1_filter_pitches[DIM];
  FastDivmod l1_output_pitches[DIM];
  Point<DIM> l2_output_limits;
  Point<DIM, unsigned> point_offsets[POINTS];
  Point<DIM, unsigned> l2_output_tile;
  Point<DIM, unsigned> l2_filter_tile;
  Point<DIM, unsigned> l1_output_tile;
  Point<DIM, unsigned> l1_filter_tile;
  unsigned total_l2_outputs;
  unsigned total_l1_outputs;
  unsigned total_l1_filters;
  unsigned total_l1_points;
  unsigned l1_filter_points;
  unsigned l1_input_points;
  unsigned shared_input_offset;
  unsigned uniform_input_stride;
  unsigned shared_input_bound;
};

template <typename VAL, int DIM, int POINTS>
__global__ static void __launch_bounds__(CONVOLUTION_THREADS, 1)
  convolution_large_tile(const AccessorWO<VAL, DIM> out,
                         const AccessorRO<VAL, DIM> filter,
                         const AccessorRO<VAL, DIM> in,
                         const Rect<DIM> root_rect,
                         const Rect<DIM> subrect,
                         const Rect<DIM> l2_filter_rect,
                         const Point<DIM> l2_input_start,
                         const Point<DIM> l2_input_stop,
                         const Point<DIM> l1_input_start,
                         const Point<DIM, unsigned> zero,
                         const Point<DIM, unsigned> one,
                         const ConvolutionLargeTileArgs<DIM, POINTS> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this is illegal C++, but there's no other way to do it
  VAL* sharedmem = (VAL*)buffer;
  Point<DIM, unsigned> thread_offset;
  int offset = threadIdx.x;
#pragma unroll
  for (int d = 0; d < DIM; d++) {
    thread_offset[d] = args.l1_output_pitches[d].divmod(offset, offset);
  }
  Point<DIM> l2_output_offset = zero;
  for (unsigned l2_outidx = 0; l2_outidx < args.total_l2_outputs; l2_outidx++) {
    // Do a quick check here to see if all the inputs are contained for this tile
    // l2_input_start = subrect.lo + args.extents - l2_filter_rect.hi - one - centers
    // l2_input_stop = subrect.lo + l2_output_tile - one + args.extents - l2_filter_rect.lo - one -
    // centers
    const Rect<DIM> l2_input_rect(l2_input_start + l2_output_offset,
                                  l2_input_stop + l2_output_offset);
    const bool input_contained = root_rect.contains(l2_input_rect);
    // Iterate the L1 output tiles that this threadblock should compute for the L2 output
    for (unsigned l1_outidx = blockIdx.x; l1_outidx < args.total_l1_outputs;
         l1_outidx += gridDim.x) {
      Point<DIM, unsigned> l1_output_offset;
      offset = l1_outidx;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        l1_output_offset[d] =
          args.l1_output_tile_pitches[d].divmod(offset, offset) * args.l1_output_tile[d];
      }
      // Handle the boundary case where an L1 tile is not contained in the L2 tile
      // becasue the L2 tile is overlapping a boundary. Note this decisions is the
      // same for all the threads in the threadblock so no bad divergence
      bool output_contained = true;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        if ((subrect.lo[d] + l2_output_offset[d] + l1_output_offset[d]) <= subrect.hi[d]) {
          continue;
        }
        output_contained = false;
        break;
      }
      if (!output_contained) {
        continue;
      }
      // Initialize our point data
      VAL acc[POINTS];
#pragma unroll
      for (int p = 0; p < POINTS; p++) {
        acc[p] = VAL{0};
      }
      // Iterate over the l1 filter tiles
      Point<DIM, unsigned> l1_filter_offset = zero;
      for (unsigned l1_fidx = 0; l1_fidx < args.total_l1_filters; l1_fidx++) {
        // Wait for any previous readers to be done
        __syncthreads();
// Load the filter into shared memory
// Unroll this a few times to get some memory level parallelims
#pragma unroll 4
        for (unsigned fidx = threadIdx.x; fidx < args.l1_filter_points; fidx += blockDim.x) {
          Point<DIM> filter_point = l2_filter_rect.lo + l1_filter_offset;
          offset                  = fidx;
#pragma unroll
          for (int d = 0; d < DIM; d++) {
            filter_point[d] += args.l1_filter_pitches[d].divmod(offset, offset);
          }
          if (l2_filter_rect.contains(filter_point)) {
            sharedmem[fidx] = filter[filter_point];
          } else {
            sharedmem[fidx] = VAL{0};
          }
        }
        // Load the input into shared memory
        // Compute the input start point
        // input_start = subrect.lo + extents - l2_filter_rect.lo - l1_filter_tile - centers
        Point<DIM> input_start = l1_input_start + l2_output_offset + l1_output_offset;
        input_start -= l1_filter_offset;
// Unroll this a few times to get some memory level parallelism
#pragma unroll 4
        for (unsigned idx = threadIdx.x; idx < args.l1_input_points; idx += blockDim.x) {
          Point<DIM> input_point = input_start;
          offset                 = idx;
#pragma unroll
          for (int d = 0; d < DIM; d++) {
            input_point[d] += args.l1_input_pitches[d].divmod(offset, offset);
          }
          if (input_contained || root_rect.contains(input_point)) {
            sharedmem[args.shared_input_offset + idx] = in[input_point];
          } else {
            sharedmem[args.shared_input_offset + idx] = VAL{0};
          }
        }
        // Wait for everything to be loaded into shared memory
        __syncthreads();
        // Iterate the points in the filter
        // We can safely iterate all the filter points and input points
        // because we wrote zeros into shared memory for everything that
        // was out of bounds
        Point<DIM, unsigned> filter_point = zero;
        if (args.uniform_input_stride) {
          // Each point is a constant offset in shared from the others
          unsigned input_offset = args.shared_input_offset;
#pragma unroll
          for (int d = 0; d < DIM; d++) {
            input_offset +=
              args.l1_input_pitches[d].divisor * (thread_offset[d] + args.l1_filter_tile[d] - 1);
          }
          if (args.shared_input_bound) {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
              unsigned point_offset  = input_offset;
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                if (args.shared_input_bound <= point_offset) {
                  break;
                }
                acc[p] = acc[p] + filter_value * sharedmem[point_offset];
                point_offset += args.uniform_input_stride;
              }
// Step to the next filter point and update the input stride
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                input_offset -= args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  input_offset += args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          } else {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
              unsigned point_offset  = input_offset;
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                acc[p] = acc[p] + filter_value * sharedmem[point_offset];
                point_offset += args.uniform_input_stride;
              }
// Step to the next filter point and update the input stride
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                input_offset -= args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  input_offset += args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          }
        } else {
          // Need to compute the input offset uniquely for each point
          Point<DIM, unsigned> input_point = thread_offset + args.l1_filter_tile - one;
          unsigned point_offsets[POINTS];
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            point_offsets[p] = args.shared_input_offset;
#pragma unroll
            for (int d = 0; d < DIM; d++) {
              point_offsets[p] +=
                (input_point[d] + args.point_offsets[p][d]) * args.l1_input_pitches[d].divisor;
            }
          }
          unsigned filter_offset = 0;
          if (args.shared_input_bound) {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                unsigned point_offset = point_offsets[p] - filter_offset;
                if (args.shared_input_bound <= point_offset) {
                  continue;
                }
                acc[p] = acc[p] + filter_value * sharedmem[point_offset];
              }
// Step to the next filter point
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                filter_offset += args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  filter_offset -= args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          } else {
            for (unsigned fidx = 0; fidx < args.l1_filter_points; fidx++) {
              // Use shared memory broadcasting functionality to avoid bank conflicts
              const VAL filter_value = sharedmem[fidx];
#pragma unroll
              for (int p = 0; p < POINTS; p++) {
                unsigned point_offset = point_offsets[p] - filter_offset;
                acc[p]                = acc[p] + filter_value * sharedmem[point_offset];
              }
// Step to the next filter point
#pragma unroll
              for (int d = DIM - 1; d >= 0; d--) {
                filter_point[d]++;
                filter_offset += args.l1_input_pitches[d].divisor;
                if (filter_point[d] == args.l1_filter_tile[d]) {
                  filter_offset -= args.l1_filter_tile[d] * args.l1_input_pitches[d].divisor;
                  filter_point[d] = 0;
                } else {
                  break;
                }
              }
            }
          }
        }
// Step to the next L1 filter tile
#pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
          l1_filter_offset[d] += args.l1_filter_tile[d];
          if (args.l2_filter_tile[d] <= l1_filter_offset[d]) {
            l1_filter_offset[d] = 0;
          } else {
            break;
          }
        }
      }
      // Now we can stream our accumulators back to the output
      Point<DIM> output = subrect.lo + l2_output_offset + (l1_output_offset + thread_offset);
      if (input_contained) {
        // If the input was contained, then so is the output
        if (args.total_l1_points) {
          unsigned index = threadIdx.x;
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            if (args.total_l1_points <= index) {
              break;
            }
            VAL* ptr = out.ptr(output + args.point_offsets[p]);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
            index += blockDim.x;
          }
        } else {
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            VAL* ptr = out.ptr(output + args.point_offsets[p]);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
          }
        }
      } else {
        // Input was not contained, so the output might not be either, do checks
        if (args.total_l1_points) {
          unsigned index = threadIdx.x;
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            if (args.total_l1_points <= index) {
              break;
            }
            Point<DIM> point = output + args.point_offsets[p];
            if (!subrect.contains(point)) {
              break;
            }
            VAL* ptr = out.ptr(point);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
            index += blockDim.x;
          }
        } else {
#pragma unroll
          for (int p = 0; p < POINTS; p++) {
            Point<DIM> point = output + args.point_offsets[p];
            if (!subrect.contains(point)) {
              continue;
            }
            VAL* ptr = out.ptr(point);
            // Make sure we don't pollute the L2 cache
            VAL value = load_streaming<VAL>(ptr);
            store_streaming<VAL>(ptr, value + acc[p]);
          }
        }
      }
    }
// Step to the next output tile
#pragma unroll
    for (int d = DIM - 1; d >= 0; d--) {
      l2_output_offset[d] += args.l2_output_tile[d];
      if (args.l2_output_limits[d] <= l2_output_offset[d]) {
        l2_output_offset[d] = 0;
      } else {
        break;
      }
    }
  }
}

template <int DIM>
struct ConvolutionSmallTileArgs {
  FastDivmodU64 grid_pitches[DIM];
  FastDivmodU64 block_pitches[DIM];
  FastDivmodU64 input_pitches[DIM];
  unsigned block_tiles[DIM];
  unsigned filter_centers[DIM];
  unsigned filter_extents[DIM];
  Point<DIM> delta_lo, delta_hi;
  size_t filter_volume;
  size_t tile_volume;
  size_t input_volume;
};

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(512, 2)
  convolution_small_tile1(const AccessorWO<VAL, DIM> out,
                          const AccessorRO<VAL, DIM> filter,
                          const AccessorRO<VAL, DIM> in,
                          const Rect<DIM> root_rect,
                          const Rect<DIM> subrect,
                          const Rect<DIM> filter_rect,
                          const ConvolutionSmallTileArgs<DIM> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this illegal C++, but there's no other way to do it
  VAL* input = (VAL*)buffer;
  // Compute the origin point of the block
  size_t offset          = blockIdx.x;
  Point<DIM> block_point = subrect.lo;
#pragma unroll
  for (int d = 0; d < DIM; d++) {
    block_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  }
  // Load in the shared memory for this block
  Point<DIM> tile_point;
  const Rect<DIM> input_bounds(block_point - args.delta_lo, block_point + args.delta_hi);
  const bool input_contained = root_rect.contains(input_bounds);
  if (input_contained) {
// All the points are contained, so no need for point-wise tests
// Unroll this four times to try to pipeline loads
#pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        tile_point[d] = args.input_pitches[d].divmod(offset, offset);
      }
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  } else {
// Need to do point-wise tests
// Unroll this four times to try to pipeline loads
#pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        tile_point[d] = args.input_pitches[d].divmod(offset, offset);
      }
      if (!root_rect.contains(input_bounds.lo + tile_point)) {
        continue;
      }
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  }
  // Wait for everything to be loaded into shared memory
  __syncthreads();
  // Loop over points in the tile and compute the outputs
  coord_t f_coords[DIM];
  Point<DIM> out_point, in_point, filter_point;
  for (unsigned idx = threadIdx.x; idx < args.tile_volume; idx += blockDim.x) {
    // Compute the local coordinates
    offset = idx;
#pragma unroll
    for (int d = 0; d < DIM; d++) {
      tile_point[d] = args.block_pitches[d].divmod(offset, offset);
      out_point[d]  = block_point[d] + tile_point[d];
    }
    if (!subrect.contains(out_point)) {
      continue;
    }
#pragma unroll
    for (int d = 0; d < DIM; d++) {
      f_coords[d] = 0;
    }
    VAL acc{0};
    for (unsigned idx = 0; idx < args.filter_volume; idx++) {
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
      }
      if (input_contained || root_rect.contains(in_point)) {
        offset = 0;
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          offset += (tile_point[d] + f_coords[d]) * args.input_pitches[d].divisor;
        }
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
        }
        acc = acc + input[offset] * filter[filter_point];
      }
// Step the filter coordinates
#pragma unroll
      for (int d = DIM - 1; d >= 0; d--) {
        f_coords[d]++;
        if (f_coords[d] == args.filter_extents[d]) {
          f_coords[d] = 0;
        } else {
          break;
        }
      }
    }
    store_streaming(out.ptr(out_point), acc);
  }
}

// This version of the kernel is identical to the one above but with
// different launch bounds to handle a bigger CTA with more shared memory
template <typename VAL, int DIM>
__global__ static void __launch_bounds__(1024, 1)
  convolution_small_tile2(const AccessorWO<VAL, DIM> out,
                          const AccessorRO<VAL, DIM> filter,
                          const AccessorRO<VAL, DIM> in,
                          const Rect<DIM> root_rect,
                          const Rect<DIM> subrect,
                          const Rect<DIM> filter_rect,
                          const ConvolutionSmallTileArgs<DIM> args)
{
  // Deal with compiler shared memory stupidity
  extern __shared__ uint8_t buffer[];
  // Technically this illegal C++, but there's no other way to do it
  VAL* input = (VAL*)buffer;
  // Compute the origin point of the block
  size_t offset          = blockIdx.x;
  Point<DIM> block_point = subrect.lo;
#pragma unroll
  for (int d = 0; d < DIM; d++) {
    block_point[d] += args.grid_pitches[d].divmod(offset, offset) * args.block_tiles[d];
  }
  // Load in the shared memory for this block
  Point<DIM> tile_point;
  const Rect<DIM> input_bounds(block_point - args.delta_lo, block_point + args.delta_hi);
  const bool input_contained = root_rect.contains(input_bounds);
  if (input_contained) {
// All the points are contained, so no need for point-wise tests
// Unroll this four times to try to pipeline loads
#pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        tile_point[d] = args.input_pitches[d].divmod(offset, offset);
      }
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  } else {
// Need to do point-wise tests
// Unroll this four times to try to pipeline loads
#pragma unroll 4
    for (unsigned idx = threadIdx.x; idx < args.input_volume; idx += blockDim.x) {
      offset = idx;
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        tile_point[d] = args.input_pitches[d].divmod(offset, offset);
      }
      if (!root_rect.contains(input_bounds.lo + tile_point)) {
        continue;
      }
      VAL value = in[input_bounds.lo + tile_point];
      // Write the value into shared memory
      input[idx] = value;
    }
  }
  // Wait for everything to be loaded into shared memory
  __syncthreads();
  // Loop over points in the tile and compute the outputs
  coord_t f_coords[DIM];
  Point<DIM> out_point, in_point, filter_point;
  for (unsigned idx = threadIdx.x; idx < args.tile_volume; idx += blockDim.x) {
    // Compute the local coordinates
    offset = idx;
#pragma unroll
    for (int d = 0; d < DIM; d++) {
      tile_point[d] = args.block_pitches[d].divmod(offset, offset);
      out_point[d]  = block_point[d] + tile_point[d];
    }
    if (!subrect.contains(out_point)) {
      continue;
    }
#pragma unroll
    for (int d = 0; d < DIM; d++) {
      f_coords[d] = 0;
    }
    VAL acc{0};
    for (unsigned idx = 0; idx < args.filter_volume; idx++) {
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        in_point[d] = out_point[d] + f_coords[d] - args.filter_centers[d];
      }
      if (input_contained || root_rect.contains(in_point)) {
        offset = 0;
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          offset += (tile_point[d] + f_coords[d]) * args.input_pitches[d].divisor;
        }
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          filter_point[d] = args.filter_extents[d] - f_coords[d] - 1;
        }
        acc = acc + input[offset] * filter[filter_point];
      }
// Step the filter coordinates
#pragma unroll
      for (int d = DIM - 1; d >= 0; d--) {
        f_coords[d]++;
        if (f_coords[d] == args.filter_extents[d]) {
          f_coords[d] = 0;
        } else {
          break;
        }
      }
    }
    store_streaming(out.ptr(out_point), acc);
  }
}

template <typename VAL, int DIM>
__host__ static inline void launch_small_tile_kernel(AccessorWO<VAL, DIM> out,
                                                     AccessorRO<VAL, DIM> filter,
                                                     AccessorRO<VAL, DIM> in,
                                                     const Rect<DIM>& root_rect,
                                                     const Rect<DIM>& subrect,
                                                     const Rect<DIM>& filter_rect,
                                                     const cudaDeviceProp& properties,
                                                     const unsigned extents[DIM],
                                                     const unsigned centers[DIM],
                                                     Point<DIM>& tile,
                                                     unsigned smem_size,
                                                     size_t max_smem_size)
{
  // Make the tile as big as possible so that it fits in shared memory
  // Try to keep it rectangular to minimize surface-to-volume ratio
  // and improve the reuse of data
  // If the current tile is less than half the shared memory in the SM then
  // decrease the upper bound so we can get 2 CTAs/SM
  bool halved              = false;
  const unsigned half_smem = properties.sharedMemPerMultiprocessor / 2;
  if ((smem_size <= (half_smem)) && (half_smem < max_smem_size)) {
    max_smem_size = half_smem;
    halved        = true;
  }
  Point<DIM> padding;
  for (int d = 0; d < DIM; d++) {
    padding[d] = 2 * centers[d];
  }
  Point<DIM> bounds = subrect.hi - subrect.lo + Point<DIM>::ONES();
  smem_size         = roundup_tile<VAL, DIM>(tile, bounds, padding, max_smem_size);
  // At this point we've got the tile size that we're going to compute
  // and the amount of dynamic shared memory that we need
  // Compute the arguments needed for the kernel launch
  ConvolutionSmallTileArgs<DIM> args;
  size_t blocks        = 1;
  size_t tile_pitch    = 1;
  unsigned input_pitch = 1;
  args.filter_volume   = 1;
  for (int d = DIM - 1; d >= 0; d--) {
    size_t blocks_along_dim = ((subrect.hi[d] - subrect.lo[d]) + tile[d]) / tile[d];
    args.grid_pitches[d]    = FastDivmodU64(blocks);
    blocks *= blocks_along_dim;
    args.block_tiles[d]   = tile[d];
    args.block_pitches[d] = FastDivmodU64(tile_pitch);
    tile_pitch *= tile[d];
    args.delta_lo[d]      = centers[d];
    args.delta_hi[d]      = tile[d] + centers[d] - 1;
    args.input_pitches[d] = FastDivmodU64(input_pitch);
    input_pitch *= (args.delta_lo[d] + args.delta_hi[d] + 1);
    args.filter_centers[d] = centers[d];
    args.filter_extents[d] = extents[d];
    args.filter_volume *= extents[d];
  }
  args.tile_volume  = tile_pitch;
  args.input_volume = input_pitch;
  assert((input_pitch * sizeof(VAL)) == smem_size);
  auto stream = get_cached_stream();
  if (halved) {
    if (tile_pitch < 512) {
      convolution_small_tile1<VAL, DIM><<<blocks, tile_pitch, smem_size, stream>>>(
        out, filter, in, root_rect, subrect, filter_rect, args);
    } else {
      convolution_small_tile1<VAL, DIM><<<blocks, 512, smem_size, stream>>>(
        out, filter, in, root_rect, subrect, filter_rect, args);
    }
  } else {
    if (tile_pitch < 1024) {
      convolution_small_tile2<VAL, DIM><<<blocks, tile_pitch, smem_size, stream>>>(
        out, filter, in, root_rect, subrect, filter_rect, args);
    } else {
      convolution_small_tile2<VAL, DIM><<<blocks, 1024, smem_size, stream>>>(
        out, filter, in, root_rect, subrect, filter_rect, args);
    }
  }
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <typename VAL, int32_t DIM>
__host__ void direct_convolution(AccessorWO<VAL, DIM> out,
                                 AccessorRO<VAL, DIM> filter,
                                 AccessorRO<VAL, DIM> in,
                                 const Rect<DIM>& root_rect,
                                 const Rect<DIM>& subrect,
                                 const Rect<DIM>& filter_rect)
{
  constexpr int THREADVALS = THREAD_OUTPUTS(VAL);
  // Get the maximum amount of shared memory per threadblock
  int device           = get_device_ordinal();
  auto& properties     = get_device_properties();
  size_t max_smem_size = properties.sharedMemPerBlockOptin;

  // Only need to do these calls the first time on each device so
  // we use a bit mask to track which devices we've done it for
  static unsigned long long mask = 0;
  if (!(mask & (1 << device))) {
    if (properties.sharedMemPerBlock < max_smem_size) {
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile1<VAL, DIM>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  max_smem_size));
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile2<VAL, DIM>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  max_smem_size));
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetAttribute(convolution_large_tile<VAL, DIM, THREADVALS>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  max_smem_size));
    }
    if (sizeof(VAL) >= 8) {
      // Only need to set this on the first invocation
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile1<VAL, DIM>,
                                                        cudaSharedMemBankSizeEightByte));
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile2<VAL, DIM>,
                                                        cudaSharedMemBankSizeEightByte));
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetSharedMemConfig(
        convolution_large_tile<VAL, DIM, THREADVALS>, cudaSharedMemBankSizeEightByte));
    }
    // Make sure we have enough bits for every device
    assert(device < (8 * sizeof(mask)));
    // Make sure not to race with updates from other GPUs
    __sync_fetch_and_add(&mask, (1 << device));
  }
  unsigned extents[DIM];
  unsigned centers[DIM];
  for (int d = 0; d < DIM; d++) {
    assert(filter_rect.lo[d] == 0);
    extents[d] = filter_rect.hi[d] + 1;
    centers[d] = static_cast<coord_t>(extents[d] / 2);
  }
  Point<DIM> tile;
  for (int d = DIM - 1; d >= 0; d--) {
    // Make sure that each tile is at least double the size of the filter
    // so that we can get some savings in bandwidth needed
    tile[d] = 2 * centers[d];
    if (d == (DIM - 1)) {
      // In order to maximize bandwidth, we want to make sure we're loading at
      // least 128B of contiguous memory along the last axis (row-major) of input
      const unsigned min_contig_elmts = 128 / sizeof(VAL);
      if ((tile[d] + 2 * centers[d]) < min_contig_elmts) {
        tile[d] = min_contig_elmts - 2 * centers[d];
      }
    }
  }
  unsigned smem_size = sizeof(VAL);
  for (int d = 0; d < DIM; d++) {
    smem_size *= std::max(static_cast<coord_t>(1), (tile[d] + 2 * centers[d]));
  }
  if (smem_size <= max_smem_size) {
    // Small tile case:
    launch_small_tile_kernel<VAL, DIM>(out,
                                       filter,
                                       in,
                                       root_rect,
                                       subrect,
                                       filter_rect,
                                       properties,
                                       extents,
                                       centers,
                                       tile,
                                       smem_size,
                                       max_smem_size);
  } else {
    // Large tile case:
    // If we're going to do this, we need to initialize the output to zeros
    // so we can kick that off to the GPU while we figure out how to launch
    // the rest of the kernels to do the convolution
    size_t strides[DIM];
    VAL* out_ptr = out.ptr(subrect, strides);
    // Check to see if the output is dense
    bool out_dense   = true;
    size_t out_pitch = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      if (strides[d] != out_pitch) {
        out_dense = false;
        break;
      }
      out_pitch *= strides[d];
    }
    if (out_dense) {
      size_t bytes = sizeof(VAL) * out_pitch;
      CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(out_ptr, 0, bytes));
    } else {
      out_pitch = 1;
      ConvolutionInitArgs<DIM> args;
      for (int d = DIM - 1; d >= 0; d--) {
        args.pitches[d] = FastDivmodU64(out_pitch);
        out_pitch *= (subrect.hi[d] - subrect.lo[d] + 1);
      }
      size_t blocks = (out_pitch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      convolution_init<VAL, DIM><<<blocks, THREADS_PER_BLOCK>>>(out, subrect.lo, args, out_pitch);
    }
    // Figure out the shape of the L1 output tile based on the number of
    // points that we can fit into registers
    Point<DIM> l1_output_tile;
    const unsigned max_l1_output_volume = CONVOLUTION_THREADS * THREADVALS;
    // Make sure the max_l1_output_volume doesn't consume more than half of shared memory
    unsigned target_l1_output_volume = max_l1_output_volume;
    while ((max_smem_size / 2) < (target_l1_output_volume * sizeof(VAL))) {
      target_l1_output_volume /= 2;
    }
    const Point<DIM> output_bounds = subrect.hi - subrect.lo + Point<DIM>::ONES();
    const unsigned l1_output_volume =
      compute_output_tile<VAL, DIM>(l1_output_tile,
                                    output_bounds,
                                    128 /*cache line size*/ / sizeof(VAL),
                                    target_l1_output_volume);
    // At this point we've got our output tile, compute how big a filter
    // tile we can make and still fit both the filter tile and the
    // input tile into the maximum amount of shared memory for this GPU
    Point<DIM> l1_filter_tile;
    const Point<DIM> filter_bounds = filter_rect.hi - filter_rect.lo + Point<DIM>::ONES();
    unsigned dynamic_smem =
      compute_filter_tile<VAL, DIM>(l1_filter_tile, filter_bounds, l1_output_tile, max_smem_size);
    unsigned input_smem_offset = 1;
    for (int d = 0; d < DIM; d++) {
      input_smem_offset *= l1_filter_tile[d];
    }
    // Tile the number of SMs on this GPU to compute the shape of the
    // L2 output tile for this kernel
    // We assume here that the number of SMs is easily factorable
    // into primes of 2, 3, and 5. It would be strange if we have a
    // GPU with a number of SMs these days that can't be factored
    // this way. If we do report a warning.
    unsigned l2_tiles[DIM];
    for (int d = 0; d < DIM; d++) {
      l2_tiles[d] = 1;
    }
    if (DIM > 1) {
      unsigned twos = 0, threes = 0, fives = 0;
      unsigned remainder = properties.multiProcessorCount;
      while ((remainder > 1) && ((remainder % 2) == 0)) {
        twos++;
        remainder /= 2;
      }
      while ((remainder > 1) && ((remainder % 3) == 0)) {
        threes++;
        remainder /= 3;
      }
      while ((remainder > 1) && ((remainder % 5) == 0)) {
        fives++;
        remainder /= 5;
      }
      if (remainder > 1) {
        fprintf(stdout,
                "WARNING: %d is an unusual number of SMs "
                "for GPU convolution. Please report your GPU kind and "
                "the number of SMs in a Legate NumPy issue.",
                properties.multiProcessorCount);
        l2_tiles[DIM - 1] = remainder;
      }
      for (unsigned idx = 0; idx < fives; idx++) {
        int smallest = 0;
        for (int d = 1; d < DIM; d++) {
          if (l2_tiles[smallest] < l2_tiles[d]) {
            continue;
          }
          smallest = d;
        }
        l2_tiles[smallest] *= 5;
      }
      for (unsigned idx = 0; idx < threes; idx++) {
        int smallest = 0;
        for (int d = 1; d < DIM; d++) {
          if (l2_tiles[smallest] < l2_tiles[d]) {
            continue;
          }
          smallest = d;
        }
        l2_tiles[smallest] *= 3;
      }
      for (unsigned idx = 0; idx < twos; idx++) {
        int smallest = 0;
        for (int d = 1; d < DIM; d++) {
          if (l2_tiles[smallest] < l2_tiles[d]) {
            continue;
          }
          smallest = d;
        }
        l2_tiles[smallest] *= 2;
      }
    } else {
      l2_tiles[0] = properties.multiProcessorCount;
    }
    // Now that we've got a tiling of the l1 output blocks across
    // the SMs compute how big it is in memory and see if it is less
    // than a quarter of the L2 cache so we can block for the L2
    Point<DIM> l2_output_tile;
    size_t l2_output_tile_size = sizeof(VAL);
    for (int d = 0; d < DIM; d++) {
      l2_output_tile[d] = l2_tiles[d] * l1_output_tile[d];
      l2_output_tile_size *= l2_output_tile[d];
    }
    Point<DIM> l2_filter_tile;
    size_t total_l2_filters = 1;
    if (l2_output_tile_size <= (static_cast<size_t>(properties.l2CacheSize) / 4)) {
      for (int d = 0; d < DIM; d++) {
        l2_filter_tile[d] = 1;
      }
      // Compute the L2 filter tile size so that the L2 filter and the
      // corresponding L2 input tile will fit in the L2 cache
      compute_filter_tile<VAL, DIM>(
        l2_filter_tile, filter_bounds, l2_output_tile, 3 * properties.l2CacheSize / 4);
      for (int d = 0; d < DIM; d++) {
        total_l2_filters *= (filter_bounds[d] + l2_filter_tile[d] - 1) / l2_filter_tile[d];
      }
    } else {
      // It's likely this tile is too big to block for the L2 cache
      // so we're not going to bother blocking for the L2 and just
      // run everything out of the framebuffer memory. The upside is
      // that we'll only need to make a single pass over the input
      for (int d = 0; d < DIM; d++) {
        l2_filter_tile[d] = filter_rect.hi[d] - filter_rect.lo[d] + 1;
      }
    }
    // Construct the arguments for the kernel launches
    ConvolutionLargeTileArgs<DIM, THREADVALS> args;
    int pitch = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      args.l1_input_pitches[d] = FastDivmod(pitch);
      pitch *= (l1_output_tile[d] + 2 * (l1_filter_tile[d] / 2));
    }
    pitch = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      args.l1_filter_pitches[d] = FastDivmod(pitch);
      pitch *= l1_filter_tile[d];
    }
    pitch = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      args.l1_output_pitches[d] = FastDivmod(pitch);
      pitch *= l1_output_tile[d];
    }
    args.l2_output_tile      = l2_output_tile;
    args.l2_filter_tile      = l2_filter_tile;
    args.l1_output_tile      = l1_output_tile;
    args.l1_filter_tile      = l1_filter_tile;
    args.l2_output_limits    = output_bounds;
    args.shared_input_offset = input_smem_offset;
    args.total_l2_outputs    = 1;
    args.total_l1_outputs    = 1;
    args.total_l1_filters    = 1;
    args.l1_filter_points    = 1;
    args.l1_input_points     = 1;
    pitch                    = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      args.total_l2_outputs *= (output_bounds[d] + l2_output_tile[d] - 1) / l2_output_tile[d];
      args.l1_output_tile_pitches[d] = FastDivmod(pitch);
      pitch *= (l2_output_tile[d] + l1_output_tile[d] - 1) / l1_output_tile[d];
      args.total_l1_filters *= (l2_filter_tile[d] + l1_filter_tile[d] - 1) / l1_filter_tile[d];
      args.l1_filter_points *= l1_filter_tile[d];
      args.l1_input_points *= (l1_output_tile[d] + 2 * (l1_filter_tile[d] / 2));
    }
    args.total_l1_outputs = pitch;
    // Figure out how to tile the points across the l1_output_tile
    if (DIM > 1) {
      unsigned regsteps[DIM];
      for (int d = 0; d < DIM; d++) {
        regsteps[d] = 0;
      }
      unsigned remainder = THREADVALS;
      // Handle the case here where we aren't going to use all
      // the points in the registers so we need to scale back
      if (l1_output_volume < max_l1_output_volume) {
        assert((max_l1_output_volume % l1_output_volume) == 0);
        remainder /= (max_l1_output_volume / l1_output_volume);
        if (remainder == 0) {
          remainder = 1;
        }
      }
      for (int d = 0; d < DIM; d++) {
        if (remainder == 1) {
          regsteps[d] = l1_output_tile[d];
        } else if (remainder <= l1_output_tile[d]) {
          // All powers of two so should always divide
          assert((l1_output_tile[d] % remainder) == 0);
          regsteps[d] = l1_output_tile[d] / remainder;
          remainder   = 1;
        } else {
          // All powers of two so should always divide
          assert((remainder % l1_output_tile[d]) == 0);
          regsteps[d] = 1;
          remainder /= l1_output_tile[d];
        }
      }
      assert(remainder == 1);
      Point<DIM, unsigned> offset = Point<DIM, unsigned>::ZEROES();
      for (int p = 0; p < THREADVALS; p++) {
        args.point_offsets[p] = offset;
        // Step to the next offset
        for (int d = DIM - 1; d >= 0; d--) {
          offset[d] += regsteps[d];
          if (offset[d] == l1_output_tile[d]) {
            if ((d == 0) && (p != (THREADVALS - 1))) {
              // Allow overflow in this case to handle the case
              // where we have more points than we need for the l1 output tile
              assert(l1_output_volume < max_l1_output_volume);
            } else {
              offset[d] = 0;
            }
          } else {
            break;
          }
        }
      }
      args.uniform_input_stride = regsteps[0] * args.l1_input_pitches[0].divisor;
      // Check to make sure this is the uniform input stride case
      for (int d = 1; d < DIM; d++) {
        if (regsteps[d] == l1_output_tile[d]) {
          continue;
        }
        args.uniform_input_stride = 0;
        break;
      }
    } else {
      assert(THREADVALS <= l1_output_tile[0]);
      unsigned remainder = THREADVALS;
      // Handle the case here where we aren't going to use all
      // the points in the registers so we need to scale back
      if (l1_output_volume < max_l1_output_volume) {
        assert((max_l1_output_volume % l1_output_volume) == 0);
        remainder /= (max_l1_output_volume / l1_output_volume);
        if (remainder == 0) {
          remainder = 1;
        }
      }
      assert((l1_output_tile[0] % remainder) == 0);
      unsigned regstep = l1_output_tile[0] / remainder;
      for (int p = 0; p < THREADVALS; p++) {
        args.point_offsets[p][0] = p * regstep;
      }
      args.uniform_input_stride = regstep * args.l1_input_pitches[0].divisor;
    }
    if (l1_output_volume < max_l1_output_volume) {
      args.shared_input_bound = dynamic_smem / sizeof(VAL);
      args.total_l1_points    = l1_output_volume;
    } else {
      args.shared_input_bound = 0;
      args.total_l1_points    = 0;
    }
    // Launch as many kernels as we need to walk over the entire filter
    // Given the L2 filter tile that we came up with
    auto stream                     = get_cached_stream();
    const Point<DIM, unsigned> zero = Point<DIM, unsigned>::ZEROES();
    const Point<DIM, unsigned> one  = Point<DIM, unsigned>::ONES();
    if (total_l2_filters > 1) {
      Point<DIM> l2_filter_lo = filter_rect.lo;
      for (unsigned idx = 0; idx < total_l2_filters; idx++) {
        Rect<DIM> l2_filter_rect(l2_filter_lo, l2_filter_lo + l2_filter_tile - one);
        l2_filter_rect = l2_filter_rect.intersection(filter_rect);
        const Point<DIM> l1_input_start =
          subrect.lo + Point<DIM>(extents) - l2_filter_lo - l1_filter_tile - Point<DIM>(centers);
        const Point<DIM> l2_input_start =
          subrect.lo + Point<DIM>(extents) - l2_filter_rect.hi - one - Point<DIM>(centers);
        const Point<DIM> l2_input_stop = subrect.lo + l2_output_tile - one + Point<DIM>(extents) -
                                         l2_filter_rect.lo - one - Point<DIM>(centers);
        convolution_large_tile<VAL, DIM, THREADVALS>
          <<<properties.multiProcessorCount, CONVOLUTION_THREADS, dynamic_smem, stream>>>(
            out,
            filter,
            in,
            root_rect,
            subrect,
            l2_filter_rect,
            l2_input_start,
            l2_input_stop,
            l1_input_start,
            zero,
            one,
            args);
        // Step to the next filter
        for (int d = DIM - 1; d >= 0; d--) {
          l2_filter_lo[d] += l2_filter_tile[d];
          if (filter_rect.hi[d] < l2_filter_lo[d]) {
            l2_filter_lo[d] = filter_rect.lo[d];
          } else {
            break;
          }
        }
      }
    } else {
      assert(total_l2_filters == 1);
      const Point<DIM> l1_input_start =
        subrect.lo + Point<DIM>(extents) - filter_rect.lo - l1_filter_tile - Point<DIM>(centers);
      const Point<DIM> l2_input_start = subrect.lo - Point<DIM>(centers);
      const Point<DIM> l2_input_stop  = subrect.lo + l2_output_tile - one + Point<DIM>(extents) -
                                       filter_rect.lo - one - Point<DIM>(centers);
      convolution_large_tile<VAL, DIM, THREADVALS>
        <<<properties.multiProcessorCount, CONVOLUTION_THREADS, dynamic_smem, stream>>>(
          out,
          filter,
          in,
          root_rect,
          subrect,
          filter_rect,
          l2_input_start,
          l2_input_stop,
          l1_input_start,
          zero,
          one,
          args);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
}

///////////////////////////////////////
// FFT-based convolution implementation
///////////////////////////////////////

template <int DIM>
struct FFTPitches {
  size_t pitches[DIM];
  __host__ inline size_t& operator[](unsigned idx) { return pitches[idx]; }
  __device__ __forceinline__ size_t operator[](unsigned idx) const { return pitches[idx]; }
};

template <int DIM>
struct CopyPitches {
  FastDivmodU64 pitches[DIM];
  __host__ inline FastDivmodU64& operator[](unsigned idx) { return pitches[idx]; }
  __device__ __forceinline__ const FastDivmodU64& operator[](unsigned idx) const
  {
    return pitches[idx];
  }
};

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 4)
  copy_into_buffer(const AccessorRO<VAL, DIM> accessor,
                   const Buffer<VAL, DIM> buffer,
                   const Point<DIM> lo,
                   const CopyPitches<DIM> copy_pitches,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  Point<DIM> point;
  for (int d = 0; d < DIM; d++) {
    point[d] = copy_pitches[d].divmod(offset, offset);
  }
  buffer[point] = accessor[lo + point];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 4)
  copy_from_buffer(const VAL* buffer,
                   const AccessorWO<VAL, DIM> accessor,
                   const Point<DIM> buffer_lo,
                   const Point<DIM> accessor_lo,
                   const CopyPitches<DIM> copy_pitches,
                   const FFTPitches<DIM> fft_pitches,
                   const size_t volume,
                   const VAL scaling)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  Point<DIM> point;
  size_t buffer_offset = 0;
  for (int d = 0; d < DIM; d++) {
    point[d] = copy_pitches[d].divmod(offset, offset);
    buffer_offset += (buffer_lo[d] + point[d]) * fft_pitches[d];
  }
  accessor[accessor_lo + point] = scaling * buffer[buffer_offset];
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, 4)
  complex_multiply(complex<VAL>* inout, complex<VAL>* in, const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  inout[offset] *= in[offset];
}

__host__ inline void cufft_execute_forward(cufftHandle plan, float* idata, float* odata)
{
  CHECK_CUFFT(cufftExecR2C(plan, (cufftReal*)idata, (cufftComplex*)odata));
}

__host__ inline void cufft_execute_forward(cufftHandle plan, double* idata, double* odata)
{
  CHECK_CUFFT(cufftExecD2Z(plan, (cufftDoubleReal*)idata, (cufftDoubleComplex*)odata));
}

__host__ inline void cufft_execute_backward(cufftHandle plan, float* idata, float* odata)
{
  CHECK_CUFFT(cufftExecC2R(plan, (cufftComplex*)idata, (cufftReal*)odata));
}

__host__ inline void cufft_execute_backward(cufftHandle plan, double* idata, double* odata)
{
  CHECK_CUFFT(cufftExecZ2D(plan, (cufftDoubleComplex*)idata, (cufftDoubleReal*)odata));
}

template <typename VAL>
struct ForwardPlanType;

template <>
struct ForwardPlanType<float> {
  static constexpr cufftType value = CUFFT_R2C;
};

template <>
struct ForwardPlanType<double> {
  static constexpr cufftType value = CUFFT_D2Z;
};

template <typename VAL>
struct BackwardPlanType;

template <>
struct BackwardPlanType<float> {
  static constexpr cufftType value = CUFFT_C2R;
};

template <>
struct BackwardPlanType<double> {
  static constexpr cufftType value = CUFFT_Z2D;
};

template <typename VAL, int DIM>
__host__ static inline void cufft_convolution(AccessorWO<VAL, DIM> out,
                                              AccessorRO<VAL, DIM> filter,
                                              AccessorRO<VAL, DIM> in,
                                              const Rect<DIM>& root_rect,
                                              const Rect<DIM>& subrect,
                                              const Rect<DIM>& filter_rect,
                                              CuPyNumericConvolveMethod method)
{
  int device           = get_device_ordinal();
  auto& properties     = get_device_properties();
  size_t max_smem_size = properties.sharedMemPerBlockOptin;

  // Only need to do these calls the first time on each device so
  // we use a bit mask to track which devices we've done it for
  static unsigned long long mask = 0;
  if (!(mask & (1 << device))) {
    if (properties.sharedMemPerBlock < max_smem_size) {
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile1<VAL, DIM>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  max_smem_size));
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetAttribute(convolution_small_tile2<VAL, DIM>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  max_smem_size));
    }
    if (sizeof(VAL) >= 8) {
      // Only need to set this on the first invocation
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile1<VAL, DIM>,
                                                        cudaSharedMemBankSizeEightByte));
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetSharedMemConfig(convolution_small_tile2<VAL, DIM>,
                                                        cudaSharedMemBankSizeEightByte));
    }
    // Make sure we have enough bits for every device
    assert(device < (8 * sizeof(mask)));
    // Make sure not to race with updates from other GPUs
    __sync_fetch_and_add(&mask, (1 << device));
  }
  unsigned extents[DIM];
  unsigned centers[DIM];
  for (int d = 0; d < DIM; d++) {
    assert(filter_rect.lo[d] == 0);
    extents[d] = filter_rect.hi[d] + 1;
    centers[d] = static_cast<coord_t>(extents[d] / 2);
  }
  Point<DIM> tile;
  for (int d = DIM - 1; d >= 0; d--) {
    // Make sure that each tile is at least double the size of the filter
    // so that we can get some savings in bandwidth needed
    tile[d] = 2 * centers[d];
    if (d == (DIM - 1)) {
      // In order to maximize bandwidth, we want to make sure we're loading at
      // least 128B of contiguous memory along the last axis (row-major) of input
      const unsigned min_contig_elmts = 128 / sizeof(VAL);
      if ((tile[d] + 2 * centers[d]) < min_contig_elmts) {
        tile[d] = min_contig_elmts - 2 * centers[d];
      }
    }
  }
  unsigned smem_size = sizeof(VAL);
  for (int d = 0; d < DIM; d++) {
    smem_size *= (tile[d] + 2 * centers[d]);
  }
  if (method != CUPYNUMERIC_CONVOLVE_FFT && smem_size <= max_smem_size) {
    launch_small_tile_kernel<VAL, DIM>(out,
                                       filter,
                                       in,
                                       root_rect,
                                       subrect,
                                       filter_rect,
                                       properties,
                                       extents,
                                       centers,
                                       tile,
                                       smem_size,
                                       max_smem_size);
  } else {
    // Instead of doing the large tile case, we can instead do this
    // by transforming both the input and the filter to the frequency
    // domain using an FFT, perform the convolution with a point-wise
    // multiplication, and then transform the result back to the spatial domain
    auto stream = get_cached_stream();
    // First compute how big our temporary allocation needs to be
    // We'll need two of them to store the zero-padded data for the inputs
    const Point<DIM> zero = Point<DIM>::ZEROES();
    const Point<DIM> one  = Point<DIM>::ONES();
    Rect<DIM> offset_bounds;
    for (int d = 0; d < DIM; d++) {
      offset_bounds.lo[d] = subrect.lo[d] - centers[d];
      offset_bounds.hi[d] = subrect.hi[d] + extents[d] - 1 - centers[d];
    }
    Rect<DIM> input_bounds         = root_rect.intersection(offset_bounds);
    const Point<DIM> signal_bounds = input_bounds.hi - input_bounds.lo + one;
    const Point<DIM> filter_bounds = filter_rect.hi - filter_rect.lo + one;
    Point<DIM> fftsize             = signal_bounds + filter_bounds;
    for (int d = 0; d < DIM; d++) {
      // Technically we can shrink this by one and still be sound but we'll
      // only do that if it will make the number even
      if ((fftsize[d] % 2) == 1) {
        fftsize[d]--;
      }
    }
    // Cufft needs the last dimension to have fftsize/2+1 complex elements for
    // the temporary buffer
    // Since we know fftsize is even, we just need to add two to it for the output
    Point<DIM> buffersize = fftsize;
    buffersize[DIM - 1] += 2;
    size_t buffervolume = 1;
    for (int d = 0; d < DIM; d++) {
      buffervolume *= buffersize[d];
    }
    // Zero pad and copy in the input data
    auto signal_buffer = create_buffer<VAL, DIM>(buffersize, Memory::GPU_FB_MEM, 128 /*alignment*/);
    VAL* signal_ptr    = signal_buffer.ptr(zero);
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(signal_ptr, 0, buffervolume * sizeof(VAL), stream));
    // Check to see if the input pointer is dense and we can do this with a CUDA memcpy
    size_t strides[DIM];
    const VAL* input_ptr = in.ptr(input_bounds, strides);
    size_t pitch         = 1;
    CopyPitches<DIM> copy_pitches;
    for (int d = DIM - 1; d >= 0; d--) {
      copy_pitches[d] = FastDivmodU64(pitch);
      pitch *= signal_bounds[d];
    }
    size_t blocks = (pitch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copy_into_buffer<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      in, signal_buffer, input_bounds.lo, copy_pitches, pitch);
    // Zero pad and copy in the filter data
    auto filter_buffer = create_buffer<VAL, DIM>(buffersize, Memory::GPU_FB_MEM, 128 /*alignment*/);
    VAL* filter_ptr    = filter_buffer.ptr(zero);
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(filter_ptr, 0, buffervolume * sizeof(VAL), stream));
    const VAL* filt_ptr = filter.ptr(filter_rect, strides);
    pitch               = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      copy_pitches[d] = FastDivmodU64(pitch);
      pitch *= filter_bounds[d];
    }
    blocks = (pitch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copy_into_buffer<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      filter, filter_buffer, filter_rect.lo, copy_pitches, pitch);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

    auto forward_plan  = get_cufft_plan(ForwardPlanType<VAL>::value, cufftPlanParams(fftsize));
    auto backward_plan = get_cufft_plan(BackwardPlanType<VAL>::value, cufftPlanParams(fftsize));

    // Set the working area for the plans
    auto workarea_size = std::max(forward_plan.workareaSize(), backward_plan.workareaSize());

    // Create the plan and allocate a temporary buffer for it if it needs one
    Buffer<uint8_t, 1> workarea_buffer;
    if (workarea_size > 0) {
      const Point<1> zero1d(0);
      workarea_buffer =
        create_buffer<uint8_t, 1>(workarea_size, Memory::GPU_FB_MEM, 128 /*alignment*/);
      void* workarea = workarea_buffer.ptr(zero1d);
      CHECK_CUFFT(cufftSetWorkArea(forward_plan.handle(), workarea));
      CHECK_CUFFT(cufftSetWorkArea(backward_plan.handle(), workarea));
    }
    // FFT the input data
    cufft_execute_forward(forward_plan.handle(), signal_ptr, signal_ptr);
    // FFT the filter data
    cufft_execute_forward(forward_plan.handle(), filter_ptr, filter_ptr);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

    // Perform the pointwise multiplcation
    {
      size_t volume = (buffervolume / 2);
      blocks        = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      complex_multiply<VAL><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (complex<VAL>*)signal_ptr, (complex<VAL>*)filter_ptr, volume);
    }
    // Inverse FFT for the ouptut
    // Allow this out-of-place for better performance
    cufft_execute_backward(backward_plan.handle(), signal_ptr, filter_ptr);
    // Copy the result data out of the temporary buffer and scale
    // because CUFFT inverse does not perform the scale for us
    pitch = 1;
    FFTPitches<DIM> fft_pitches;
    for (int d = DIM - 1; d >= 0; d--) {
      fft_pitches[d] = pitch;
      pitch *= fftsize[d];
    }
    const VAL scaling_factor = VAL(1) / pitch;
    Point<DIM> buffer_offset;
    for (int d = 0; d < DIM; d++) {
      buffer_offset[d] =
        centers[d] - (((extents[d] % 2) == 0) ? 1 : 0) +
        ((offset_bounds.lo[d] < root_rect.lo[d]) ? (subrect.lo[d] - root_rect.lo[d]) : centers[d]);
    }
    Point<DIM> output_bounds = subrect.hi - subrect.lo + one;
    pitch                    = 1;
    for (int d = DIM - 1; d >= 0; d--) {
      copy_pitches[d] = FastDivmodU64(pitch);
      pitch *= output_bounds[d];
    }
    blocks = (pitch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copy_from_buffer<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      filter_ptr, out, buffer_offset, subrect.lo, copy_pitches, fft_pitches, pitch, scaling_factor);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

#if 0
    // This is useful debugging code for finding the output
    VAL *buffer = (VAL*)malloc(buffervolume*sizeof(VAL));
    CUPYNUMERIC_CHECK_CUDA( cudaMemcpyAsync(buffer, filter_ptr, buffervolume*sizeof(VAL), cudaMemcpyDeviceToHost, stream) );
    CUPYNUMERIC_CHECK_CUDA( cudaStreamSynchronize(stream) );
    for (unsigned idx = 0; idx < buffervolume; idx++) {
      if ((idx % fftsize[DIM-1]) == 0)
        printf("\n");
      printf("%.8g ", buffer[idx]*scaling_factor);
    }
    printf("\n");
    free(buffer);
#endif
  }
}

/////////////
// Dispatcher
/////////////

template <typename VAL, int DIM>
struct CanUseCUFFT {
  static constexpr bool value = 1 <= DIM && DIM <= 3 && std::is_floating_point<VAL>::value;
};

template <Type::Code CODE, int DIM>
struct ConvolveImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  template <typename _VAL,
            int32_t _DIM,
            std::enable_if_t<CanUseCUFFT<_VAL, _DIM>::value>* = nullptr>
  __host__ void dispatch(AccessorWO<_VAL, _DIM> out,
                         AccessorRO<_VAL, _DIM> filter,
                         AccessorRO<_VAL, _DIM> in,
                         const Rect<_DIM>& root_rect,
                         const Rect<_DIM>& subrect,
                         const Rect<_DIM>& filter_rect,
                         CuPyNumericConvolveMethod method) const
  {
    if (method == CUPYNUMERIC_CONVOLVE_DIRECT) {
      direct_convolution<_VAL, _DIM>(out, filter, in, root_rect, subrect, filter_rect);
    } else {
      cufft_convolution<_VAL, _DIM>(out, filter, in, root_rect, subrect, filter_rect, method);
    }
  }

  template <typename _VAL,
            int32_t _DIM,
            std::enable_if_t<!CanUseCUFFT<_VAL, _DIM>::value>* = nullptr>
  __host__ void dispatch(AccessorWO<_VAL, _DIM> out,
                         AccessorRO<_VAL, _DIM> filter,
                         AccessorRO<_VAL, _DIM> in,
                         const Rect<_DIM>& root_rect,
                         const Rect<_DIM>& subrect,
                         const Rect<_DIM>& filter_rect,
                         CuPyNumericConvolveMethod method) const
  {
    direct_convolution<_VAL, _DIM>(out, filter, in, root_rect, subrect, filter_rect);
  }

  __host__ void operator()(AccessorWO<VAL, DIM> out,
                           AccessorRO<VAL, DIM> filter,
                           AccessorRO<VAL, DIM> in,
                           const Rect<DIM>& root_rect,
                           const Rect<DIM>& subrect,
                           const Rect<DIM>& filter_rect,
                           CuPyNumericConvolveMethod method) const
  {
    dispatch(out, filter, in, root_rect, subrect, filter_rect, method);
  }
};

/*static*/ void ConvolveTask::gpu_variant(TaskContext context)
{
  convolve_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
