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

#include "cupynumeric/binary/binary_op_util.h"

namespace cupynumeric {

std::vector<uint64_t> broadcast_shapes(std::vector<NDArray> arrays)
{
#ifdef DEBUG_CUPYNUMERIC
  assert(!arrays.empty());
#endif
  int32_t dim = 0;
  for (auto& array : arrays) {
    dim = std::max(dim, array.dim());
  }

  std::vector<uint64_t> result(dim, 1);

  for (auto& array : arrays) {
    const auto shape = array.shape();

    auto in_it  = shape.rbegin();
    auto out_it = result.rbegin();
    for (; in_it != shape.rend() && out_it != result.rend(); ++in_it, ++out_it) {
      if (1 == *out_it) {
        *out_it = *in_it;
      } else if (*in_it != 1 && *out_it != *in_it) {
        throw std::exception();
      }
    }
  }
  return result;
}

}  // namespace cupynumeric
