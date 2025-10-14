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

#include <legate/data/buffer.h>
#include <cupynumeric/stat/detail/histogramdd_using_thrust.h>

namespace cupynumeric {

namespace detail {

using namespace legate::mapping;

// use this large alignment because it is sometimes used in cub (and so transitively affects us
// through thrust)
static constexpr size_t DEFAULT_ALIGNMENT = 256;

namespace {
size_t pad(size_t bytes)
{
  size_t blocks = (bytes + DEFAULT_ALIGNMENT - 1) / DEFAULT_ALIGNMENT;
  return blocks * DEFAULT_ALIGNMENT;
}
}  // namespace

std::optional<size_t> histogramdd_using_thrust_allocation_pool_size(const Task& task,
                                                                    StoreTarget memory_kind)
{
  using output_t = HistogramDDTask::output_t;
  using edge_t   = HistogramDDTask::edge_t;

  auto points_shape = task.input(0).data().shape<2>();
  auto has_weights  = task.scalar(0).value<bool>();
  auto num_points   = points_shape.hi[0] + 1 - points_shape.lo[0];
  auto num_dims     = points_shape.hi[1] + 1 - points_shape.lo[1];
  size_t first_dim  = has_weights ? 2 : 1;

  size_t num_bins = 1;

  for (size_t d = 0; d < num_dims; d++) {
    auto dim_edges_shape = task.input(d + first_dim).data().shape<1>();
    auto dim_bins        = dim_edges_shape.hi - dim_edges_shape.lo;  // one more edge than bins

    num_bins *= dim_bins;
  }

  size_t points_to_bins_size    = num_points * sizeof(int64_t);
  size_t edge_list_size         = num_dims * sizeof(thrust::pair<const edge_t*, size_t>);
  size_t weights_buffer_size    = has_weights ? num_points * sizeof(output_t) : 0;
  size_t non_empty_bins_size    = (1 + num_bins) * sizeof(int64_t);
  size_t non_empty_weights_size = (1 + num_bins) * sizeof(output_t);
  size_t thrust_sort_c0         = 2;
  size_t thrust_sort_c1         = 1;
  size_t thrust_sort_c2         = DEFAULT_ALIGNMENT;
  size_t sort_points_size =
    (thrust_sort_c0 * num_points + thrust_sort_c1) * sizeof(int64_t) + thrust_sort_c2;
  size_t sort_weights_size =
    has_weights
      ? ((thrust_sort_c0 * num_points + thrust_sort_c1) * sizeof(output_t) + thrust_sort_c2)
      : 0;
  size_t thrust_rbk_c0 = 1;
  size_t thrust_rbk_c1 = 1;
  size_t thrust_rbk_c2 = DEFAULT_ALIGNMENT;
  size_t reduce_by_key_size =
    (thrust_rbk_c0 * (num_bins + 1) + thrust_rbk_c1) * sizeof(output_t) + thrust_rbk_c2;

  size_t total_size = 0;
  for (const size_t& item : {points_to_bins_size,
                             edge_list_size,
                             weights_buffer_size,
                             non_empty_bins_size,
                             non_empty_weights_size,
                             sort_points_size,
                             sort_weights_size,
                             reduce_by_key_size}) {
    total_size += pad(item);
  }

  switch (memory_kind) {
    case StoreTarget::FBMEM: [[fallthrough]];
    case StoreTarget::SOCKETMEM: [[fallthrough]];
    case StoreTarget::SYSMEM: {
      return total_size;
    }
    default: break;
  }
  return 0;
}

}  // namespace detail

}  // namespace cupynumeric
