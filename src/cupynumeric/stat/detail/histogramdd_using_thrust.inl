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

#include <cupynumeric/stat/histogramdd.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <vector>

namespace cupynumeric {
namespace detail {
using namespace legate;

// histogramdd outputs and edges are always doubles in numpy
using output_t = HistogramDDTask::output_t;
using edge_t   = HistogramDDTask::edge_t;

namespace {

template <Type::Code CODE>
inline constexpr bool is_candidate = (is_floating_point<CODE>::value || is_integral<CODE>::value);

template <typename exe_policy_t, typename point_t>
int64_t histogramdd_points_to_bins(const exe_policy_t& policy,
                                   Memory::Kind memkind,
                                   const PhysicalStore& points_store,
                                   const std::vector<PhysicalStore>& edge_stores,
                                   int64_t* points_to_bins_array)
{
  // unpack points
  auto points_rect = points_store.shape<2>();
  auto points_lo   = points_rect.lo;
  auto points_hi   = points_rect.hi;
  auto num_points  = points_hi[0] + 1 - points_lo[0];
  auto num_dims    = points_hi[1] + 1 - points_lo[1];
  auto first_dim   = points_lo[1];
  auto points      = points_store.read_accessor<point_t, 2>(points_rect);

  // unpack edges
  auto edges        = create_buffer<thrust::pair<const edge_t*, size_t>>(num_dims, memkind);
  auto* edges_array = edges.ptr(0);

  int64_t num_bins = 1;

  {  // get each dimension's edges as a (pointer, last_edge) pair
    std::vector<thrust::pair<const edge_t*, size_t>> host_edges_vec;
    for (auto const& dim_edges_store : edge_stores) {
      auto rect      = dim_edges_store.shape<1>();
      auto dim_edges = dim_edges_store.read_accessor<edge_t, 1>(rect);
      size_t strides[1];
      const edge_t* first = dim_edges.ptr(rect, strides);
      assert(strides[0] == 1);
      int64_t last_edge = rect.hi - rect.lo;
      assert(last_edge > 0);
      host_edges_vec.push_back({first, last_edge});
      num_bins *=
        last_edge;  // the index of the last edge is the number of intervals in this dimension
    }
    thrust::copy(policy, host_edges_vec.begin(), host_edges_vec.end(), edges_array);
  }

  // for each point compare its coordinate in each dimension to the edges to
  // figure out the interval it belongs to in that dimension, and
  // simultaneously convert that multindex into a single linear index (assign
  // to the linear index num_bins if it is not in the histogram domain)
  thrust::transform(
    policy,
    thrust::make_counting_iterator<coord_t>(points_lo[0]),
    thrust::make_counting_iterator<coord_t>(points_hi[0] + 1),
    points_to_bins_array,
    [num_dims, num_bins, first_dim, edges_array, points] LEGATE_HOST_DEVICE(coord_t point_idx) {
      int64_t point_bin = 0;
      for (int64_t d = 0; d < num_dims; d++) {
        auto& [dim_edges, last_edge] = edges_array[d];
        auto x                       = (edge_t)points[Point<2>(point_idx, d + first_dim)];

        point_bin *= last_edge;
        if (!(x == x) || x < dim_edges[0] || x > dim_edges[last_edge]) {
          // point is outside of this dimension's edges
          point_bin = num_bins;
          break;
        }
        if (x == dim_edges[last_edge]) {
          point_bin += last_edge - 1;
        } else {
          for (int64_t b = 0; b < last_edge; b++) {
            if (x < dim_edges[b + 1]) {
              point_bin += b;
              break;
            }
          }
        }
      }
      return point_bin;
    });

  return num_bins;
}

template <typename exe_policy_t>
struct HistogrgmDDPointsToBins {
  template <Type::Code CODE, std::enable_if_t<is_candidate<CODE>>* = nullptr>
  int64_t operator()(const exe_policy_t& policy,
                     Memory::Kind memkind,
                     const PhysicalStore& points_store,
                     const std::vector<PhysicalStore>& edge_stores,
                     int64_t* points_to_bins_array)
  {
    using point_t = type_of<CODE>;

    return histogramdd_points_to_bins<exe_policy_t, point_t>(
      policy, memkind, points_store, edge_stores, points_to_bins_array);
  }

  template <Type::Code CODE, std::enable_if_t<!is_candidate<CODE>>* = nullptr>
  int64_t operator()(const exe_policy_t&,
                     Memory::Kind,
                     const PhysicalStore&,
                     const std::vector<PhysicalStore>&,
                     int64_t*)
  {
    assert(false);
    return 0;
  }
};

template <typename exe_policy_t>
size_t histogramdd_bin_counts(const exe_policy_t& policy,
                              Memory::Kind memkind,
                              int64_t num_points,
                              int64_t num_bins,
                              int64_t* points_to_bins_array,
                              int64_t* non_empty_bins_array,
                              output_t* non_empty_weights_array)
{
  thrust::sort(policy, points_to_bins_array, points_to_bins_array + num_points);

  auto new_end = thrust::reduce_by_key(policy,
                                       points_to_bins_array,
                                       points_to_bins_array + num_points,
                                       thrust::make_constant_iterator<output_t>(1.0),
                                       non_empty_bins_array,
                                       non_empty_weights_array);

  return new_end.first - non_empty_bins_array;
}

template <typename exe_policy_t, typename weight_t>
size_t histogramdd_bin_weights(const exe_policy_t& policy,
                               Memory::Kind memkind,
                               int64_t num_points,
                               int64_t num_bins,
                               int64_t* points_to_bins_array,
                               PhysicalStore weights_store,
                               int64_t* non_empty_bins_array,
                               output_t* non_empty_weights_array)
{
  auto weights_rect = weights_store.shape<2>();
  auto weights_acc  = weights_store.read_accessor<weight_t, 2>(weights_rect);

  auto weights_buffer = create_buffer<output_t>(num_points, memkind);
  auto* weights_array = weights_buffer.ptr(0);

  // copy to weights_array, which is mutable, so it can be sorted
  thrust::transform(policy,
                    thrust::make_counting_iterator<int64_t>(0),
                    thrust::make_counting_iterator<int64_t>(num_points),
                    weights_array,
                    [weights_acc, weights_rect] LEGATE_HOST_DEVICE(int64_t i) {
                      return (output_t)weights_acc[Point<2>(weights_rect.lo[0] + i, 0)];
                    });

  thrust::sort_by_key(
    policy, points_to_bins_array, points_to_bins_array + num_points, weights_array);

  auto new_end = thrust::reduce_by_key(policy,
                                       points_to_bins_array,
                                       points_to_bins_array + num_points,
                                       weights_array,
                                       non_empty_bins_array,
                                       non_empty_weights_array);

  return new_end.first - non_empty_bins_array;
}

template <typename exe_policy_t>
struct HistogrgmDDBinWeights {
  template <Type::Code CODE, std::enable_if_t<is_candidate<CODE>>* = nullptr>
  size_t operator()(const exe_policy_t& policy,
                    Memory::Kind memkind,
                    int64_t num_points,
                    int64_t num_bins,
                    int64_t* points_to_bins_array,
                    PhysicalStore weights,
                    int64_t* non_empty_bins_array,
                    output_t* non_empty_weights_array)
  {
    using weight_t = type_of<CODE>;

    return histogramdd_bin_weights<exe_policy_t, weight_t>(policy,
                                                           memkind,
                                                           num_points,
                                                           num_bins,
                                                           points_to_bins_array,
                                                           weights,
                                                           non_empty_bins_array,
                                                           non_empty_weights_array);
  }

  template <Type::Code CODE, std::enable_if_t<!is_candidate<CODE>>* = nullptr>
  size_t operator()(const exe_policy_t&,
                    Memory::Kind,
                    int64_t,
                    int64_t,
                    int64_t*,
                    PhysicalStore,
                    int64_t*,
                    output_t*)
  {
    assert(false);
    return 0;
  }
};

}  // namespace

template <typename exe_policy_t>
void histogramdd_using_thrust(TaskContext& context,
                              const exe_policy_t& policy,
                              Memory::Kind memkind)
{
  auto num_args     = context.num_inputs();
  auto points       = context.input(0).data();
  auto has_weights  = context.scalar(0).value<bool>();
  auto points_shape = points.shape<2>();
  auto num_points   = points_shape.hi[0] - points_shape.lo[0] + 1;
  auto num_dims     = points_shape.hi[1] - points_shape.lo[1] + 1;
  auto hist_store   = context.reduction(0).data();
  auto hist_rect    = hist_store.shape<1>();
  auto hist         = hist_store.reduce_accessor<SumReduction<output_t>, true, 1>(hist_rect);
  size_t first_bin  = has_weights ? 2 : 1;

  // there are either num_dims + 1 arguments (no weights) or num_dims + 2 arguments (weights)
  assert(num_args == num_dims + 1 || num_args == num_dims + 2);

  // Step 1: comput the bin that each point is in
  std::vector<PhysicalStore> bin_edges;
  for (int i = 0; i < num_dims; ++i) {
    bin_edges.emplace_back(context.input(i + first_bin));
  }

  auto points_to_bins        = create_buffer<int64_t>(num_points, memkind);
  auto* points_to_bins_array = points_to_bins.ptr(0);

  auto num_bins = type_dispatch(points.code(),
                                HistogrgmDDPointsToBins<exe_policy_t>{},
                                policy,
                                memkind,
                                points,
                                bin_edges,
                                points_to_bins_array);

  // Step 2: locally sum weight associated with each point into its bin
  auto non_empty_bins        = create_buffer<int64_t>(num_bins + 1, memkind);
  auto* non_empty_bins_array = non_empty_bins.ptr(0);

  auto non_empty_weights        = create_buffer<output_t>(num_bins + 1, memkind);
  auto* non_empty_weights_array = non_empty_weights.ptr(0);

  size_t num_non_empty = [&]() {
    if (num_args == num_dims + 1) {
      // no weights vector
      return histogramdd_bin_counts(policy,
                                    memkind,
                                    num_points,
                                    num_bins,
                                    points_to_bins_array,
                                    non_empty_bins_array,
                                    non_empty_weights_array);
    } else {
      auto weights = context.input(1).data();

      return type_dispatch(weights.code(),
                           HistogrgmDDBinWeights<exe_policy_t>{},
                           policy,
                           memkind,
                           num_points,
                           num_bins,
                           points_to_bins_array,
                           weights,
                           non_empty_bins_array,
                           non_empty_weights_array);
    }
  }();

  // Step 3: sum local weights into output reduction store

  // there is at most one entry per bin, so
  // reductions can be performed in exclusive mode
  thrust::for_each(
    policy,
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_non_empty),
    [num_bins, non_empty_bins_array, non_empty_weights_array, hist_rect, hist] LEGATE_HOST_DEVICE(
      size_t i) {
      auto bin = non_empty_bins_array[i];

      if (bin < num_bins) {
        hist[hist_rect.lo + bin] <<= non_empty_weights_array[i];
      }
    });
}

}  // namespace detail
}  // namespace cupynumeric
