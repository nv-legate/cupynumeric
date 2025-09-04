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

#include "cupynumeric/set/in1d.h"
#include "cupynumeric/set/in1d_template.inl"
#include "cupynumeric/utilities/thrust_util.h"
#include "cupynumeric/cuda_help.h"

#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace cupynumeric {

using namespace legate;

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  check_in1d(bool* result,
             const VAL* in1,
             const VAL* in2,
             const size_t volume1,
             const size_t volume2,
             const bool invert)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume1) {
    return;
  }

  size_t left  = 0;
  size_t right = volume2;
  VAL target   = in1[idx];
  bool found   = false;

  while (left < right) {
    size_t mid  = left + ((right - left) >> 1);
    VAL mid_val = in2[mid];
    if (mid_val < target) {
      left = mid + 1;
    } else if (mid_val > target) {
      right = mid;
    } else {
      found = true;
      break;
    }
  }
  result[idx] = invert ? !found : found;
}

template <Type::Code CODE>
struct In1dImplBody<VariantKind::GPU, CODE, 1> {
  TaskContext context;
  explicit In1dImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorWO<bool, 1>& result,
                  const AccessorRO<VAL, 1>& in1,
                  const AccessorRO<VAL, 1>& in2,
                  const Rect<1>& rect1,
                  const Rect<1>& rect2,
                  const size_t volume1,
                  const size_t volume2,
                  const bool assume_unique,
                  const bool invert,
                  const std::string& kind,
                  int64_t ar2_min,
                  int64_t ar2_max)
  {
    auto stream = context.get_task_stream();

    const VAL* in1_ptr = in1.ptr(rect1);
    const VAL* in2_ptr = in2.ptr(rect2);
    bool* result_ptr   = result.ptr(rect1);

    thrust::device_vector<VAL> temp_in2;
    size_t in2_volume = volume2;

    // Handle duplicates and sorting based on assume_unique and algorithm choice
    if (!assume_unique || kind != "table") {
      temp_in2.assign(in2_ptr, in2_ptr + volume2);  // Copy
      thrust::sort(DEFAULT_POLICY.on(stream), temp_in2.begin(), temp_in2.end());

      // Remove duplicates if not assumed unique
      if (!assume_unique) {
        auto last = thrust::unique(DEFAULT_POLICY.on(stream), temp_in2.begin(), temp_in2.end());
        temp_in2.erase(last, temp_in2.end());
        in2_volume = temp_in2.size();
      }
      in2_ptr = thrust::raw_pointer_cast(temp_in2.data());
    }

    // Handle table
    if (kind == "table") {
      if constexpr (std::is_integral<VAL>::value) {
        if (ar2_max < ar2_min) {
          // If range is invalid, fill result with 'invert' value and return
          thrust::fill(DEFAULT_POLICY.on(stream), result_ptr, result_ptr + volume1, invert);
          CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
          return;
        }

        size_t range = static_cast<size_t>(ar2_max - ar2_min + 1);
        thrust::device_vector<uint8_t> mask(range, 0);
        uint8_t* mask_ptr = thrust::raw_pointer_cast(mask.data());

        // Set mask for ar2 values
        if (in2_volume > 0) {
          thrust::for_each_n(thrust::device,
                             thrust::counting_iterator<size_t>(0),
                             in2_volume,
                             [mask_ptr, ar2_min, ar2_max, in2_ptr] __device__(size_t idx) {
                               auto v        = in2_ptr[idx];
                               size_t offset = static_cast<size_t>(v - ar2_min);
                               if (v >= ar2_min && v <= ar2_max) {
                                 mask_ptr[offset] = 1;
                               }
                             });
        }

        // For each ar1 value, check mask
        if (volume1 > 0) {
          thrust::for_each_n(
            thrust::device,
            thrust::counting_iterator<size_t>(0),
            volume1,
            [mask_ptr, ar2_min, ar2_max, result_ptr, in1_ptr, invert] __device__(size_t idx) {
              auto v          = in1_ptr[idx];
              bool found      = (v >= ar2_min && v <= ar2_max)
                                  ? (mask_ptr[static_cast<size_t>(v - ar2_min)] != 0)
                                  : false;
              result_ptr[idx] = invert ? !found : found;
            });
        }
        CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
        return;
      }
    }

    // Launch kernel to binary search for matches
    if (volume1 > 0) {
      const size_t blocks = (volume1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

      check_in1d<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        result_ptr, in1_ptr, in2_ptr, volume1, in2_volume, invert);
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void In1dTask::gpu_variant(TaskContext context)
{
  in1d_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
