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
#include <set>
#include <vector>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>

namespace cupynumeric {

using namespace legate;

template <Type::Code CODE>
struct In1dImplBody<VariantKind::CPU, CODE, 1> {
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
    const VAL* in1_ptr = in1.ptr(rect1);
    const VAL* in2_ptr = in2.ptr(rect2);
    bool* result_ptr   = result.ptr(rect1);

    thrust::host_vector<VAL> temp_in2(in2_ptr, in2_ptr + volume2);  // Copy
    size_t in2_volume = volume2;

    // Handle duplicates and sorting based on assume_unique and algorithm choice
    if (!assume_unique || kind != "table") {
      thrust::sort(temp_in2.begin(), temp_in2.end());

      // Remove duplicates if not assumed unique
      if (!assume_unique) {
        auto last = thrust::unique(temp_in2.begin(), temp_in2.end());
        temp_in2.erase(last, temp_in2.end());
        in2_volume = temp_in2.size();
      }
      in2_ptr = temp_in2.data();
    }

    // Lookup table logic
    if (kind == "table") {
      if (ar2_max < ar2_min) {
        // If range is invalid, fill result with 'invert' value and return
        thrust::fill(result_ptr, result_ptr + volume1, invert);
        return;
      }

      if constexpr (std::is_integral<VAL>::value) {
        size_t range = ar2_max - ar2_min + 1;
        std::vector<bool> mask(range, false);

        // Create the table
        for (size_t idx = 0; idx < in2_volume; ++idx) {
          VAL v = in2_ptr[idx];
          if (v >= ar2_min && v <= ar2_max) {
            mask[v - ar2_min] = true;
          }
        }

        // Use the table to check for matches
        for (size_t idx = 0; idx < volume1; ++idx) {
          VAL v           = in1_ptr[idx];
          bool found      = (v >= ar2_min && v <= ar2_max) ? mask[v - ar2_min] : false;
          result_ptr[idx] = invert ? !found : found;
        }
        return;
      }
    }

    for (size_t idx = 0; idx < volume1; ++idx) {
      VAL v           = in1_ptr[idx];
      bool found      = thrust::binary_search(in2_ptr, in2_ptr + in2_volume, v);
      result_ptr[idx] = invert ? !found : found;
    }
  }
};

/*static*/ void In1dTask::cpu_variant(TaskContext context)
{
  in1d_template<VariantKind::CPU>(context);
}

// Register the task variants
namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  In1dTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric