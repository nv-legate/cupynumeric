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
#include "cupynumeric/arg.h"
#include "cupynumeric/arg.inl"

#ifdef __NVCC__
#include "thrust/complex.h"
#endif

#define _USE_MATH_DEFINES

#include <math.h>
#include <complex>

namespace cupynumeric {

enum class UnaryOpCode : int {
  ABSOLUTE    = CUPYNUMERIC_UOP_ABSOLUTE,
  ANGLE       = CUPYNUMERIC_UOP_ANGLE,
  ARCCOS      = CUPYNUMERIC_UOP_ARCCOS,
  ARCCOSH     = CUPYNUMERIC_UOP_ARCCOSH,
  ARCSIN      = CUPYNUMERIC_UOP_ARCSIN,
  ARCSINH     = CUPYNUMERIC_UOP_ARCSINH,
  ARCTAN      = CUPYNUMERIC_UOP_ARCTAN,
  ARCTANH     = CUPYNUMERIC_UOP_ARCTANH,
  CBRT        = CUPYNUMERIC_UOP_CBRT,
  CEIL        = CUPYNUMERIC_UOP_CEIL,
  CLIP        = CUPYNUMERIC_UOP_CLIP,
  CONJ        = CUPYNUMERIC_UOP_CONJ,
  COPY        = CUPYNUMERIC_UOP_COPY,
  COS         = CUPYNUMERIC_UOP_COS,
  COSH        = CUPYNUMERIC_UOP_COSH,
  DEG2RAD     = CUPYNUMERIC_UOP_DEG2RAD,
  EXP         = CUPYNUMERIC_UOP_EXP,
  EXP2        = CUPYNUMERIC_UOP_EXP2,
  EXPM1       = CUPYNUMERIC_UOP_EXPM1,
  FLOOR       = CUPYNUMERIC_UOP_FLOOR,
  FREXP       = CUPYNUMERIC_UOP_FREXP,
  GETARG      = CUPYNUMERIC_UOP_GETARG,
  IMAG        = CUPYNUMERIC_UOP_IMAG,
  INVERT      = CUPYNUMERIC_UOP_INVERT,
  ISFINITE    = CUPYNUMERIC_UOP_ISFINITE,
  ISINF       = CUPYNUMERIC_UOP_ISINF,
  ISNAN       = CUPYNUMERIC_UOP_ISNAN,
  LOG         = CUPYNUMERIC_UOP_LOG,
  LOG10       = CUPYNUMERIC_UOP_LOG10,
  LOG1P       = CUPYNUMERIC_UOP_LOG1P,
  LOG2        = CUPYNUMERIC_UOP_LOG2,
  LOGICAL_NOT = CUPYNUMERIC_UOP_LOGICAL_NOT,
  MODF        = CUPYNUMERIC_UOP_MODF,
  NEGATIVE    = CUPYNUMERIC_UOP_NEGATIVE,
  POSITIVE    = CUPYNUMERIC_UOP_POSITIVE,
  RAD2DEG     = CUPYNUMERIC_UOP_RAD2DEG,
  REAL        = CUPYNUMERIC_UOP_REAL,
  RECIPROCAL  = CUPYNUMERIC_UOP_RECIPROCAL,
  RINT        = CUPYNUMERIC_UOP_RINT,
  ROUND       = CUPYNUMERIC_UOP_ROUND,
  SIGN        = CUPYNUMERIC_UOP_SIGN,
  SIGNBIT     = CUPYNUMERIC_UOP_SIGNBIT,
  SIN         = CUPYNUMERIC_UOP_SIN,
  SINH        = CUPYNUMERIC_UOP_SINH,
  SQRT        = CUPYNUMERIC_UOP_SQRT,
  SQUARE      = CUPYNUMERIC_UOP_SQUARE,
  TAN         = CUPYNUMERIC_UOP_TAN,
  TANH        = CUPYNUMERIC_UOP_TANH,
  TRUNC       = CUPYNUMERIC_UOP_TRUNC,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(UnaryOpCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case UnaryOpCode::ABSOLUTE:
      return f.template operator()<UnaryOpCode::ABSOLUTE>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ANGLE:
      return f.template operator()<UnaryOpCode::ANGLE>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCCOS:
      return f.template operator()<UnaryOpCode::ARCCOS>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCCOSH:
      return f.template operator()<UnaryOpCode::ARCCOSH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCSIN:
      return f.template operator()<UnaryOpCode::ARCSIN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCSINH:
      return f.template operator()<UnaryOpCode::ARCSINH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCTAN:
      return f.template operator()<UnaryOpCode::ARCTAN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ARCTANH:
      return f.template operator()<UnaryOpCode::ARCTANH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CBRT:
      return f.template operator()<UnaryOpCode::CBRT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CEIL:
      return f.template operator()<UnaryOpCode::CEIL>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CLIP:
      return f.template operator()<UnaryOpCode::CLIP>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::CONJ:
      return f.template operator()<UnaryOpCode::CONJ>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::COPY:
      return f.template operator()<UnaryOpCode::COPY>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::COS:
      return f.template operator()<UnaryOpCode::COS>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::COSH:
      return f.template operator()<UnaryOpCode::COSH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::DEG2RAD:
      return f.template operator()<UnaryOpCode::DEG2RAD>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::EXP:
      return f.template operator()<UnaryOpCode::EXP>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::EXP2:
      return f.template operator()<UnaryOpCode::EXP2>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::EXPM1:
      return f.template operator()<UnaryOpCode::EXPM1>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::FLOOR:
      return f.template operator()<UnaryOpCode::FLOOR>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::GETARG:
      return f.template operator()<UnaryOpCode::GETARG>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::IMAG:
      return f.template operator()<UnaryOpCode::IMAG>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::INVERT:
      return f.template operator()<UnaryOpCode::INVERT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ISFINITE:
      return f.template operator()<UnaryOpCode::ISFINITE>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ISINF:
      return f.template operator()<UnaryOpCode::ISINF>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ISNAN:
      return f.template operator()<UnaryOpCode::ISNAN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOG:
      return f.template operator()<UnaryOpCode::LOG>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOG10:
      return f.template operator()<UnaryOpCode::LOG10>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOG1P:
      return f.template operator()<UnaryOpCode::LOG1P>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOG2:
      return f.template operator()<UnaryOpCode::LOG2>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::LOGICAL_NOT:
      return f.template operator()<UnaryOpCode::LOGICAL_NOT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::NEGATIVE:
      return f.template operator()<UnaryOpCode::NEGATIVE>(std::forward<Fnargs>(args)...);
    // UnaryOpCode::POSITIVE is an alias to UnaryOpCode::COPY
    case UnaryOpCode::POSITIVE:
      return f.template operator()<UnaryOpCode::COPY>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::RAD2DEG:
      return f.template operator()<UnaryOpCode::RAD2DEG>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::REAL:
      return f.template operator()<UnaryOpCode::REAL>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::RECIPROCAL:
      return f.template operator()<UnaryOpCode::RECIPROCAL>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::RINT:
      return f.template operator()<UnaryOpCode::RINT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::ROUND:
      return f.template operator()<UnaryOpCode::ROUND>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SIGN:
      return f.template operator()<UnaryOpCode::SIGN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SIGNBIT:
      return f.template operator()<UnaryOpCode::SIGNBIT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SIN:
      return f.template operator()<UnaryOpCode::SIN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SINH:
      return f.template operator()<UnaryOpCode::SINH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SQRT:
      return f.template operator()<UnaryOpCode::SQRT>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::SQUARE:
      return f.template operator()<UnaryOpCode::SQUARE>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::TAN:
      return f.template operator()<UnaryOpCode::TAN>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::TANH:
      return f.template operator()<UnaryOpCode::TANH>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::TRUNC:
      return f.template operator()<UnaryOpCode::TRUNC>(std::forward<Fnargs>(args)...);
    case UnaryOpCode::FREXP:
    case UnaryOpCode::MODF: {
      // These operations should be handled somewhere else
      assert(false);
      return f.template operator()<UnaryOpCode::ABSOLUTE>(std::forward<Fnargs>(args)...);
    }
  }
  assert(false);
  return f.template operator()<UnaryOpCode::ABSOLUTE>(std::forward<Fnargs>(args)...);
}

template <legate::Type::Code CODE>
static constexpr bool is_floating_point =
  legate::is_floating_point<CODE>::value || CODE == legate::Type::Code::FLOAT16;

template <legate::Type::Code CODE>
static constexpr bool is_floating_or_complex =
  is_floating_point<CODE> || legate::is_complex<CODE>::value;

template <UnaryOpCode OP_CODE, legate::Type::Code CODE>
struct UnaryOp {
  static constexpr bool valid = false;
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ABSOLUTE, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    return abs(x);
  }

  template <
    typename _T                                                                    = T,
    std::enable_if_t<(std::is_integral<_T>::value and std::is_signed<_T>::value)>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return x >= 0 ? x : -x;
  }

  template <
    typename _T                                                                    = T,
    std::enable_if_t<std::is_integral<_T>::value and std::is_unsigned<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    return x;
  }

  template <typename _T                                     = T,
            std::enable_if_t<!legate::is_complex_type<_T>::value and
                             !std::is_integral<_T>::value>* = nullptr>
  constexpr _T operator()(const _T& x) const
  {
    using std::fabs;
    return static_cast<_T>(fabs(x));
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ANGLE, CODE> {
  using T                     = legate::type_of_t<CODE>;
  static constexpr bool valid = true;

  UnaryOp(const std::vector<legate::Scalar>& args) : deg{args.size() == 1 && args[0].value<bool>()}
  {
    assert(args.size() == 1);
  }

  template <typename U = T, std::enable_if_t<legate::is_complex_type<U>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    double res = atan2(x.imag(), x.real());
    return deg ? res * 180.0 / M_PI : res;
  }

  template <typename U = T, std::enable_if_t<!legate::is_complex_type<U>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    double res = atan2(0.0, static_cast<double>(x));
    return res >= 0 ? 0.0 : (deg ? 180.0 : M_PI);
  }

  bool deg;
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ARCCOS, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::acos;
    return acos(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ARCCOSH, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::acosh;
    return acosh(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::ARCCOSH, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::acosh;
    return __half{acosh(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ARCSIN, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::asin;
    return asin(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ARCSINH, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::asinh;
    return asinh(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::ARCSINH, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::asinh;
    return __half{asinh(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ARCTAN, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::atan;
    return atan(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ARCTANH, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::atanh;
    return atanh(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::ARCTANH, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::atanh;
    return __half{atanh(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::CBRT, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::cbrt;
    return cbrt(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::CBRT, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::cbrt;
    return __half{cbrt(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::CEIL, CODE> {
  static constexpr bool valid = is_floating_point<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::ceil;
    return ceil(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::CLIP, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args)
    : min{args[0].value<T>()}, max{args[1].value<T>()}
  {
    assert(args.size() == 2);
  }

  constexpr T operator()(const T& x) const { return (x < min) ? min : (x > max) ? max : x; }

  T min;
  T max;
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::CONJ, CODE> {
  using T                     = legate::type_of<CODE>;
  static constexpr bool valid = true;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr T operator()(const T& x) const
  {
    return T{x.real(), -x.imag()};
  }

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr T operator()(const T& x) const
  {
    return x;
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::COPY, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr T operator()(const T& x) const { return x; }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::COS, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::cos;
    return cos(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::COSH, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::cosh;
    return cosh(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::COSH, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::cosh;
    return __half{cosh(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::DEG2RAD, CODE> {
  static constexpr bool valid = is_floating_point<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x * T{M_PI / 180.0}; }
};

template <>
struct UnaryOp<UnaryOpCode::DEG2RAD, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    return __half{static_cast<float>(x) * static_cast<float>(M_PI / 180.0)};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::EXP, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::exp;
    return exp(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::EXP2, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr T operator()(const T& x) const
  {
    return std::exp2(x);
  }

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr T operator()(const T& x) const
  {
    using std::exp;
    using std::log;
#ifdef __NVCC__
    using thrust::exp;
    using thrust::log;
#endif
    // FIXME this is not the most performant implementation
    return exp(T(log(2), 0) * x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::EXP2, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::exp2;
    return __half{exp2(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::EXPM1, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::expm1;
    return expm1(x);
  }

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::exp;
    return exp(x) - T(1);
  }
};

template <>
struct UnaryOp<UnaryOpCode::EXPM1, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::expm1;
    return __half{expm1(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::FLOOR, CODE> {
  static constexpr bool valid = is_floating_point<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::floor;
    return floor(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::GETARG, CODE> {
  using T                     = Argval<legate::type_of<CODE>>;
  static constexpr bool valid = true;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x.arg; }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::IMAG, CODE> {
  using T                     = legate::type_of<CODE>;
  static constexpr bool valid = legate::is_complex_type<T>::value;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x.imag(); }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::INVERT, CODE> {
  static constexpr bool valid =
    legate::is_integral<CODE>::value && CODE != legate::Type::Code::BOOL;
  using T = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr T operator()(const T& x) const { return ~x; }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ISFINITE, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return true;
  }

  template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value>* = nullptr>
  __CUDA_HD__ bool operator()(const T& x) const
  {
    return std::isfinite(x);
  }

  template <typename _T>
  __CUDA_HD__ bool operator()(const complex<_T>& x) const
  {
    return std::isfinite(x.imag()) && std::isfinite(x.real());
  }

  __CUDA_HD__ bool operator()(const __half& x) const { return isfinite(static_cast<float>(x)); }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ISINF, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return false;
  }

  template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value>* = nullptr>
  __CUDA_HD__ bool operator()(const T& x) const
  {
    return std::isinf(x);
  }

  template <typename _T>
  __CUDA_HD__ bool operator()(const complex<_T>& x) const
  {
    return std::isinf(x.imag()) || std::isinf(x.real());
  }

  __CUDA_HD__ bool operator()(const __half& x) const { return isinf(static_cast<float>(x)); }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ISNAN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<std::is_integral<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return false;
  }

  template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value>* = nullptr>
  __CUDA_HD__ bool operator()(const T& x) const
  {
    using std::isnan;
    return isnan(x);
  }

  template <typename _T>
  __CUDA_HD__ bool operator()(const complex<_T>& x) const
  {
    return std::isnan(x.imag()) || std::isnan(x.real());
  }

  __CUDA_HD__ bool operator()(const __half& x) const { return isnan(x); }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::LOG, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  ;
  using T = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log;
    return log(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::LOG10, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  ;
  using T = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log10;
    return log10(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::LOG10, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::log10;
    return __half{log10(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::LOG1P, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  ;
  using T = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log1p;
    return log1p(x);
  }

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log;
    return log(T(1) + x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::LOG1P, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::log1p;
    return __half{log1p(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::LOG2, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  ;
  using T = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log2;
    return log2(x);
  }

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::log;
    return log(x) / log(T{2});
  }
};

template <>
struct UnaryOp<UnaryOpCode::LOG2, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::sinh;
    return __half{log2(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::LOGICAL_NOT, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return !static_cast<bool>(x);
  }

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr bool operator()(const T& x) const
  {
    return !static_cast<bool>(x.real());
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::NEGATIVE, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr T operator()(const T& x) const { return -x; }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::RAD2DEG, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr T operator()(const T& x) const { return x * 180.0 / M_PI; }
};

template <>
struct UnaryOp<UnaryOpCode::RAD2DEG, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    return __half{static_cast<float>(x) * static_cast<float>(180.0 / M_PI)};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::REAL, CODE> {
  using T                     = legate::type_of<CODE>;
  static constexpr bool valid = legate::is_complex_type<T>::value;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const { return x.real(); }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::RECIPROCAL, CODE> {
  using T                     = legate::type_of<CODE>;
  static constexpr bool valid = true;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr T operator()(const T& x) const
  {
    // TODO: We should raise an exception for any divide-by-zero attempt
    return x != T(0) ? T(1) / x : 0;
  }
};

template <>
struct UnaryOp<UnaryOpCode::RECIPROCAL, legate::Type::Code::FLOAT16> {
  using T                     = __half;
  static constexpr bool valid = true;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    return static_cast<float>(x) != 0 ? __half{1} / x : __half{0};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::RINT, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    return _T(std::rint(x.real()), std::rint(x.imag()));
  }

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    return std::rint(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::RINT, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::rint;
    return __half{rint(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::ROUND, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args)
    // args[0] is original signed decimals, which is needed for sign comparison
    // args[1] is pre-multiplied factor 10**abs(decimals) to avoid calling std::pow here
    : decimals{args[0].value<int64_t>()}, factor{args[1].value<int64_t>()}
  {
    LEGATE_ASSERT(args.size() == 2);
  }

  constexpr T operator()(const T& x) const
  {
    if constexpr (legate::is_complex_type<T>::value) {
      if (decimals < 0) {
        return T{static_cast<typename T::value_type>(std::rint(x.real() / factor) * factor),
                 static_cast<typename T::value_type>(std::rint(x.imag() / factor) * factor)};
      } else {
        return T{static_cast<typename T::value_type>(std::rint(x.real() * factor) / factor),
                 static_cast<typename T::value_type>(std::rint(x.imag() * factor) / factor)};
      }
    } else {
      if (decimals < 0) {
        return static_cast<T>(std::rint(x / factor) * factor);
      } else {
        return static_cast<T>(std::rint(x * factor) / factor);
      }
    }
  }

  int64_t decimals;
  int64_t factor;
};

template <>
struct UnaryOp<UnaryOpCode::ROUND, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args)
    // args[0] is original signed decimals, which is needed for sign comparison
    // args[1] is pre-multiplied factor 10**abs(decimals) to avoid calling std::pow here
    : decimals{args[0].value<int64_t>()}, factor{args[1].value<int64_t>()}
  {
    LEGATE_ASSERT(args.size() == 2);
  }

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    if (decimals < 0) {
      return static_cast<__half>(std::rint(static_cast<float>(x) / factor) * factor);
    } else {
      return static_cast<__half>(std::rint(static_cast<float>(x) * factor) / factor);
    }
  }

  int64_t decimals;
  int64_t factor;
};

namespace detail {

template <typename T, std::enable_if_t<std::is_signed<T>::value>* = nullptr>
constexpr T sign(const T& x)
{
  return x > 0 ? T(1) : (x < 0 ? T(-1) : T(0));
}

template <typename T, std::enable_if_t<!std::is_signed<T>::value>* = nullptr>
constexpr T sign(const T& x)
{
  return x > 0 ? T(1) : T(0);
}

}  // namespace detail

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::SIGN, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  template <typename _T = T, std::enable_if_t<legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    if (x.real() != 0) {
      return _T(detail::sign(x.real()), 0);
    } else {
      return _T(detail::sign(x.imag()), 0);
    }
  }

  template <typename _T = T, std::enable_if_t<!legate::is_complex_type<_T>::value>* = nullptr>
  constexpr decltype(auto) operator()(const _T& x) const
  {
    return detail::sign(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::SIGN, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    return __half{detail::sign(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::SIGNBIT, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr bool operator()(const T& x) const
  {
    using std::signbit;
    return signbit(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::SIGNBIT, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ bool operator()(const __half& x) const
  {
    using std::signbit;
    return std::signbit(static_cast<float>(x));
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::SIN, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::sin;
    return sin(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::SINH, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::sinh;
    return sinh(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::SINH, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::sinh;
    return __half{sinh(static_cast<float>(x))};
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::SQUARE, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr T operator()(const T& x) const { return x * x; }
};

template <>
struct UnaryOp<UnaryOpCode::SQUARE, legate::Type::Code::BOOL> {
  static constexpr bool valid = true;
  using T                     = bool;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr bool operator()(const bool& x) const { return x && x; }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::SQRT, CODE> {
  static constexpr bool valid = true;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::sqrt;
    return sqrt(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::TAN, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::tan;
    return tan(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::TANH, CODE> {
  static constexpr bool valid = is_floating_or_complex<CODE>;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::tanh;
    return tanh(x);
  }
};

template <legate::Type::Code CODE>
struct UnaryOp<UnaryOpCode::TRUNC, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using T                     = legate::type_of<CODE>;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  constexpr decltype(auto) operator()(const T& x) const
  {
    using std::trunc;
    return trunc(x);
  }
};

template <>
struct UnaryOp<UnaryOpCode::TRUNC, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using T                     = __half;

  UnaryOp(const std::vector<legate::Scalar>& args) {}

  __CUDA_HD__ __half operator()(const __half& x) const
  {
    using std::trunc;
    return __half{trunc(static_cast<float>(x))};
  }
};

template <UnaryOpCode OP_CODE, legate::Type::Code CODE>
struct MultiOutUnaryOp {
  static constexpr bool valid = false;
};

template <legate::Type::Code CODE>
struct MultiOutUnaryOp<UnaryOpCode::FREXP, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using RHS1                  = legate::type_of<CODE>;
  using RHS2                  = int32_t;
  using LHS                   = RHS1;

  __CUDA_HD__ LHS operator()(const RHS1& rhs1, RHS2* rhs2) const
  {
    using std::frexp;
    return frexp(rhs1, rhs2);
  }
};

template <>
struct MultiOutUnaryOp<UnaryOpCode::FREXP, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using RHS1                  = __half;
  using RHS2                  = int32_t;
  using LHS                   = __half;

  __CUDA_HD__ LHS operator()(const RHS1& rhs1, RHS2* rhs2) const
  {
    using std::frexp;
    return static_cast<__half>(frexp(static_cast<float>(rhs1), rhs2));
  }
};

template <legate::Type::Code CODE>
struct MultiOutUnaryOp<UnaryOpCode::MODF, CODE> {
  static constexpr bool valid = legate::is_floating_point<CODE>::value;
  using RHS1                  = legate::type_of<CODE>;
  using RHS2                  = RHS1;
  using LHS                   = RHS1;

  __CUDA_HD__ LHS operator()(const RHS1& rhs1, RHS2* rhs2) const
  {
    using std::modf;
    return modf(rhs1, rhs2);
  }
};

template <>
struct MultiOutUnaryOp<UnaryOpCode::MODF, legate::Type::Code::FLOAT16> {
  static constexpr bool valid = true;
  using RHS1                  = __half;
  using RHS2                  = __half;
  using LHS                   = __half;

  __CUDA_HD__ LHS operator()(const RHS1& rhs1, RHS2* rhs2) const
  {
    using std::modf;
    float tmp;
    float result = modf(static_cast<float>(rhs1), &tmp);
    *rhs2        = static_cast<__half>(tmp);
    return static_cast<__half>(result);
  }
};

}  // namespace cupynumeric
