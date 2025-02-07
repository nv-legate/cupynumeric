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

#include "legate/data/buffer.h"
#include "cupynumeric/utilities/thrust_allocator.h"
#include "cupynumeric/utilities/thrust_util.h"

#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {
namespace detail {

using namespace legate;
using namespace Legion;

template <class VAL>
void thrust_local_sort(const VAL* values_in,
                       VAL* values_out,
                       const int64_t* indices_in,
                       int64_t* indices_out,
                       const size_t volume,
                       const size_t sort_dim_size,
                       const bool stable,
                       cudaStream_t stream)
{
  auto alloc       = ThrustAllocator(Memory::GPU_FB_MEM);
  auto exec_policy = DEFAULT_POLICY(alloc).on(stream);

  if (values_in != values_out) {
    // not in-place --> need a copy
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      values_out, values_in, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  }
  if (indices_in != indices_out) {
    // not in-place --> need a copy
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      indices_out, values_in, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));
  }

  if (indices_out == nullptr) {
    if (volume == sort_dim_size) {
      if (stable) {
        thrust::stable_sort(exec_policy, values_out, values_out + volume);
      } else {
        thrust::sort(exec_policy, values_out, values_out + volume);
      }
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(exec_policy,
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), values_out));

      if (stable) {
        thrust::stable_sort(
          exec_policy, combined, combined + volume, thrust::less<thrust::tuple<size_t, VAL>>());
      } else {
        thrust::sort(
          exec_policy, combined, combined + volume, thrust::less<thrust::tuple<size_t, VAL>>());
      }

      sort_id.destroy();
    }
  } else {
    if (volume == sort_dim_size) {
      if (stable) {
        thrust::stable_sort_by_key(exec_policy, values_out, values_out + volume, indices_out);
      } else {
        thrust::sort_by_key(exec_policy, values_out, values_out + volume, indices_out);
      }
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(exec_policy,
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), values_out));

      if (stable) {
        thrust::stable_sort_by_key(exec_policy,
                                   combined,
                                   combined + volume,
                                   indices_out,
                                   thrust::less<thrust::tuple<size_t, VAL>>());
      } else {
        thrust::sort_by_key(exec_policy,
                            combined,
                            combined + volume,
                            indices_out,
                            thrust::less<thrust::tuple<size_t, VAL>>());
      }

      sort_id.destroy();
    }
  }
}

}  // namespace detail
}  // namespace cupynumeric
