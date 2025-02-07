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

#define _USE_MATH_DEFINES

#include "cupynumeric/cupynumeric_task.h"
#include <math.h>

extern double i0(double);

namespace cupynumeric {

enum class WindowOpCode : int {
  BARLETT  = CUPYNUMERIC_WINDOW_BARLETT,
  BLACKMAN = CUPYNUMERIC_WINDOW_BLACKMAN,
  HAMMING  = CUPYNUMERIC_WINDOW_HAMMING,
  HANNING  = CUPYNUMERIC_WINDOW_HANNING,
  KAISER   = CUPYNUMERIC_WINDOW_KAISER,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(WindowOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case WindowOpCode::BARLETT:
      return f.template operator()<WindowOpCode::BARLETT>(std::forward<Fnargs>(args)...);
    case WindowOpCode::BLACKMAN:
      return f.template operator()<WindowOpCode::BLACKMAN>(std::forward<Fnargs>(args)...);
    case WindowOpCode::HAMMING:
      return f.template operator()<WindowOpCode::HAMMING>(std::forward<Fnargs>(args)...);
    case WindowOpCode::HANNING:
      return f.template operator()<WindowOpCode::HANNING>(std::forward<Fnargs>(args)...);
    case WindowOpCode::KAISER:
      return f.template operator()<WindowOpCode::KAISER>(std::forward<Fnargs>(args)...);
  }
  assert(false);
  return f.template operator()<WindowOpCode::BARLETT>(std::forward<Fnargs>(args)...);
}

template <WindowOpCode OP_CODE>
struct WindowOp;

template <>
struct WindowOp<WindowOpCode::BARLETT> {
  WindowOp(int64_t M, double) : alpha_(static_cast<double>(M - 1) / 2.0) {}
  constexpr double operator()(int64_t idx) const
  {
    return idx < alpha_ ? idx / alpha_ : 2.0 - idx / alpha_;
  }

  double alpha_;
};

template <>
struct WindowOp<WindowOpCode::BLACKMAN> {
  WindowOp(int64_t M, double) : alpha_(M_PI * 2 / (M - 1)) {}
  __CUDA_HD__ double operator()(int64_t idx) const
  {
    using std::cos;
    double val = idx * alpha_;
    return 0.42 - 0.5 * cos(val) + 0.08 * cos(2 * val);
  }

  double alpha_;
};

template <>
struct WindowOp<WindowOpCode::HAMMING> {
  WindowOp(int64_t M, double) : alpha_(M_PI * 2 / (M - 1)) {}
  __CUDA_HD__ double operator()(int64_t idx) const { return 0.54 - 0.46 * std::cos(idx * alpha_); }

  double alpha_;
};

template <>
struct WindowOp<WindowOpCode::HANNING> {
  WindowOp(int64_t M, double) : alpha_(M_PI * 2 / (M - 1)) {}
  __CUDA_HD__ double operator()(int64_t idx) const { return 0.5 - 0.5 * std::cos(idx * alpha_); }

  double alpha_;
};

template <>
struct WindowOp<WindowOpCode::KAISER> {
  WindowOp(int64_t M, double beta) : alpha_((M - 1) / 2.0), beta_(beta) {}

#if defined(__NVCC__) || defined(__CUDACC__)
  __device__ double operator()(int64_t idx) const
  {
    auto val    = (idx - alpha_) / alpha_;
    auto result = cyl_bessel_i0(beta_ * std::sqrt(1 - val * val));
    result /= cyl_bessel_i0(beta_);
    return result;
  }
#else
  double operator()(int64_t idx) const
  {
    auto val    = (idx - alpha_) / alpha_;
    auto result = i0(beta_ * std::sqrt(1 - val * val));
    result /= i0(beta_);
    return result;
  }
#endif

  double alpha_;
  double beta_;
};

}  // namespace cupynumeric
