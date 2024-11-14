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
#include "cupynumeric/runtime.h"
namespace cupynumeric {

template <typename T>
NDArray arange(T start, std::optional<T> stop, T step)
{
  if (!stop.has_value()) {
    stop  = start;
    start = 0;
  }

  int64_t _N = static_cast<int64_t>(std::ceil((stop.value() - start) / static_cast<double>(step)));
  size_t N   = _N < 0 ? 0 : _N;

  auto s_start = Scalar(start);
  auto s_stop  = Scalar(stop.value());
  auto s_step  = Scalar(step);
  auto out     = CuPyNumericRuntime::get_runtime()->create_array({N}, s_start.type());
  out.arange(s_start, s_stop, s_step);
  return out;
}

}  // namespace cupynumeric
