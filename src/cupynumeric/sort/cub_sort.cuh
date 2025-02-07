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

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {
namespace detail {

using namespace legate;
using namespace Legion;

template <class VAL>
void cub_local_sort(const VAL* values_in,
                    VAL* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream)
{
  Buffer<VAL> keys_in;
  const VAL* values_in_cub = values_in;
  if (values_in == values_out) {
    keys_in       = create_buffer<VAL>(volume, Memory::Kind::GPU_FB_MEM);
    values_in_cub = keys_in.ptr(0);
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      keys_in.ptr(0), values_out, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  }

  auto multiply = [=] __host__ __device__(int x) { return x * sort_dim_size; };

  size_t temp_storage_bytes = 0;
  if (indices_out == nullptr) {
    if (volume == sort_dim_size) {
      // sort (initial call to compute buffer size)
      cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes, values_in_cub, values_out, volume, 0, sizeof(VAL) * 8, stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Memory::Kind::GPU_FB_MEM);
      cub::DeviceRadixSort::SortKeys(temp_storage.ptr(0),
                                     temp_storage_bytes,
                                     values_in_cub,
                                     values_out,
                                     volume,
                                     0,
                                     sizeof(VAL) * 8,
                                     stream);
      temp_storage.destroy();
    } else {
      // segmented sort (initial call to compute buffer size)
      // generate start/end positions for all segments via iterators to avoid allocating buffers
      auto off_start_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply);
      auto off_end_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply);

      cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                              temp_storage_bytes,
                                              values_in_cub,
                                              values_out,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_pos_it,
                                              off_end_pos_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortKeys(temp_storage.ptr(0),
                                              temp_storage_bytes,
                                              values_in_cub,
                                              values_out,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_pos_it,
                                              off_end_pos_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      temp_storage.destroy();
    }
  } else {
    Buffer<int64_t> idx_in;
    const int64_t* indices_in_cub = indices_in;
    if (indices_in == indices_out) {
      idx_in         = create_buffer<int64_t>(volume, Memory::Kind::GPU_FB_MEM);
      indices_in_cub = idx_in.ptr(0);
      CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
        idx_in.ptr(0), indices_out, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));
    }

    if (volume == sort_dim_size) {
      // argsort (initial call to compute buffer size)
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      temp_storage_bytes,
                                      values_in_cub,
                                      values_out,
                                      indices_in_cub,
                                      indices_out,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Memory::Kind::GPU_FB_MEM);

      cub::DeviceRadixSort::SortPairs(temp_storage.ptr(0),
                                      temp_storage_bytes,
                                      values_in_cub,
                                      values_out,
                                      indices_in_cub,
                                      indices_out,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);
      temp_storage.destroy();
    } else {
      // segmented argsort (initial call to compute buffer size)
      // generate start/end positions for all segments via iterators to avoid allocating buffers
      auto off_start_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply);
      auto off_end_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply);

      cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                               temp_storage_bytes,
                                               values_in_cub,
                                               values_out,
                                               indices_in_cub,
                                               indices_out,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_pos_it,
                                               off_end_pos_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.ptr(0),
                                               temp_storage_bytes,
                                               values_in_cub,
                                               values_out,
                                               indices_in_cub,
                                               indices_out,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_pos_it,
                                               off_end_pos_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);
      temp_storage.destroy();
    }
    if (indices_in == indices_out) {
      idx_in.destroy();
    }
  }

  if (values_in == values_out) {
    keys_in.destroy();
  }
}

}  // namespace detail
}  // namespace cupynumeric
