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

#include "cupynumeric/cupynumeric_task.h"

namespace cupynumeric {

using namespace legate;

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) fft_dispatch(CuPyNumericFFTType type, Functor f, Fnargs&&... args)
{
  switch (type) {
    case CUPYNUMERIC_FFT_R2C:
      return f.template operator()<CUPYNUMERIC_FFT_R2C>(std::forward<Fnargs>(args)...);
    case CUPYNUMERIC_FFT_C2R:
      return f.template operator()<CUPYNUMERIC_FFT_C2R>(std::forward<Fnargs>(args)...);
    case CUPYNUMERIC_FFT_C2C:
      return f.template operator()<CUPYNUMERIC_FFT_C2C>(std::forward<Fnargs>(args)...);
    case CUPYNUMERIC_FFT_D2Z:
      return f.template operator()<CUPYNUMERIC_FFT_D2Z>(std::forward<Fnargs>(args)...);
    case CUPYNUMERIC_FFT_Z2D:
      return f.template operator()<CUPYNUMERIC_FFT_Z2D>(std::forward<Fnargs>(args)...);
    case CUPYNUMERIC_FFT_Z2Z:
      return f.template operator()<CUPYNUMERIC_FFT_Z2Z>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<CUPYNUMERIC_FFT_C2C>(std::forward<Fnargs>(args)...);
}

template <CuPyNumericFFTType TYPE, Type::Code CODE_IN>
struct FFT {
  static constexpr bool valid = false;
};

template <>
struct FFT<CUPYNUMERIC_FFT_R2C, Type::Code::FLOAT32> {
  static constexpr bool valid          = true;
  static constexpr Type::Code CODE_OUT = Type::Code::COMPLEX64;
};

template <>
struct FFT<CUPYNUMERIC_FFT_C2R, Type::Code::COMPLEX64> {
  static constexpr bool valid          = true;
  static constexpr Type::Code CODE_OUT = Type::Code::FLOAT32;
};

template <>
struct FFT<CUPYNUMERIC_FFT_C2C, Type::Code::COMPLEX64> {
  static constexpr bool valid          = true;
  static constexpr Type::Code CODE_OUT = Type::Code::COMPLEX64;
};

template <>
struct FFT<CUPYNUMERIC_FFT_D2Z, Type::Code::FLOAT64> {
  static constexpr bool valid          = true;
  static constexpr Type::Code CODE_OUT = Type::Code::COMPLEX128;
};

template <>
struct FFT<CUPYNUMERIC_FFT_Z2D, Type::Code::COMPLEX128> {
  static constexpr bool valid          = true;
  static constexpr Type::Code CODE_OUT = Type::Code::FLOAT64;
};

template <>
struct FFT<CUPYNUMERIC_FFT_Z2Z, Type::Code::COMPLEX128> {
  static constexpr bool valid          = true;
  static constexpr Type::Code CODE_OUT = Type::Code::COMPLEX128;
};

}  // namespace cupynumeric
