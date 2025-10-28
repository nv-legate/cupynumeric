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

#include "legate/utilities/typedefs.h"
#include "legate/type/complex.h"
#include "legate/type/half.h"

#include <complex>

namespace cupynumeric {

template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
constexpr bool is_nan(const T& x)
{
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
__CUDA_HD__ bool is_nan(const T& x)
{
  return std::isnan(x);
}

template <typename T>
__CUDA_HD__ bool is_nan(const legate::Complex<T>& x)
{
  return std::isnan(x.imag()) || std::isnan(x.real());
}

template <typename T>
__CUDA_HD__ bool is_nan(const std::complex<T>& x)
{
  return std::isnan(x.imag()) || std::isnan(x.real());
}

__CUDA_HD__ inline bool is_nan(const legate::Half& x)
{
  using std::isnan;
  return isnan(static_cast<float>(x));
}

}  // namespace cupynumeric
