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
#include "cupynumeric/unary/isnan.h"

namespace cupynumeric {

enum class UnaryRedCode : int {
  ALL           = CUPYNUMERIC_RED_ALL,
  ANY           = CUPYNUMERIC_RED_ANY,
  ARGMAX        = CUPYNUMERIC_RED_ARGMAX,
  ARGMIN        = CUPYNUMERIC_RED_ARGMIN,
  CONTAINS      = CUPYNUMERIC_RED_CONTAINS,
  COUNT_NONZERO = CUPYNUMERIC_RED_COUNT_NONZERO,
  MAX           = CUPYNUMERIC_RED_MAX,
  MIN           = CUPYNUMERIC_RED_MIN,
  NANARGMAX     = CUPYNUMERIC_RED_NANARGMAX,
  NANARGMIN     = CUPYNUMERIC_RED_NANARGMIN,
  NANMAX        = CUPYNUMERIC_RED_NANMAX,
  NANMIN        = CUPYNUMERIC_RED_NANMIN,
  NANPROD       = CUPYNUMERIC_RED_NANPROD,
  NANSUM        = CUPYNUMERIC_RED_NANSUM,
  PROD          = CUPYNUMERIC_RED_PROD,
  SUM           = CUPYNUMERIC_RED_SUM,
  SUM_SQUARES   = CUPYNUMERIC_RED_SUM_SQUARES,
  VARIANCE      = CUPYNUMERIC_RED_VARIANCE
};

template <UnaryRedCode OP_CODE>
struct is_arg_reduce : std::false_type {};
template <>
struct is_arg_reduce<UnaryRedCode::ARGMAX> : std::true_type {};
template <>
struct is_arg_reduce<UnaryRedCode::ARGMIN> : std::true_type {};
template <>
struct is_arg_reduce<UnaryRedCode::NANARGMAX> : std::true_type {};
template <>
struct is_arg_reduce<UnaryRedCode::NANARGMIN> : std::true_type {};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(UnaryRedCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case UnaryRedCode::ALL:
      return f.template operator()<UnaryRedCode::ALL>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::ANY:
      return f.template operator()<UnaryRedCode::ANY>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::ARGMAX:
      return f.template operator()<UnaryRedCode::ARGMAX>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::ARGMIN:
      return f.template operator()<UnaryRedCode::ARGMIN>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::CONTAINS:
      return f.template operator()<UnaryRedCode::CONTAINS>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::COUNT_NONZERO:
      return f.template operator()<UnaryRedCode::COUNT_NONZERO>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::MAX:
      return f.template operator()<UnaryRedCode::MAX>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::MIN:
      return f.template operator()<UnaryRedCode::MIN>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::NANARGMAX:
      return f.template operator()<UnaryRedCode::NANARGMAX>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::NANARGMIN:
      return f.template operator()<UnaryRedCode::NANARGMIN>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::NANMAX:
      return f.template operator()<UnaryRedCode::NANMAX>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::NANMIN:
      return f.template operator()<UnaryRedCode::NANMIN>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::NANPROD:
      return f.template operator()<UnaryRedCode::NANPROD>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::NANSUM:
      return f.template operator()<UnaryRedCode::NANSUM>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::PROD:
      return f.template operator()<UnaryRedCode::PROD>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::SUM:
      return f.template operator()<UnaryRedCode::SUM>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::SUM_SQUARES:
      return f.template operator()<UnaryRedCode::SUM_SQUARES>(std::forward<Fnargs>(args)...);
    case UnaryRedCode::VARIANCE:
      return f.template operator()<UnaryRedCode::VARIANCE>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<UnaryRedCode::MAX>(std::forward<Fnargs>(args)...);
}

template <UnaryRedCode OP_CODE, legate::Type::Code TYPE_CODE, typename Tag = void>
struct UnaryRedOp {
  static constexpr bool valid = false;
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ALL, TYPE_CODE> {
  static constexpr bool valid = TYPE_CODE != legate::Type::Code::COMPLEX128;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = bool;
  using OP  = legate::ProdReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs != RHS(0);
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs != RHS(0); }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ANY, TYPE_CODE> {
  static constexpr bool valid = TYPE_CODE != legate::Type::Code::COMPLEX128;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = bool;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs != RHS(0);
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs != RHS(0); }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::COUNT_NONZERO, TYPE_CODE> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = uint64_t;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return static_cast<VAL>(rhs != RHS(0));
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL)
  {
    return static_cast<VAL>(rhs != RHS(0));
  }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::MAX, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MaxReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs; }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::MIN, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MinReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs; }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::PROD, TYPE_CODE> {
  static constexpr bool valid = TYPE_CODE != legate::Type::Code::COMPLEX128;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::ProdReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs; }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::SUM, TYPE_CODE> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs; }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::SUM_SQUARES, TYPE_CODE> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = Legion::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const Legion::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs * rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs * rhs; }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::VARIANCE, TYPE_CODE> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = Legion::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const Legion::Point<DIM>&, int32_t, const VAL, const RHS& rhs)
  {
    return rhs * rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL) { return rhs * rhs; }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ARGMAX, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = Argval<RHS>;
  using OP  = ArgmaxReduction<RHS>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 int32_t collapsed_dim,
                                 const VAL,
                                 const RHS& rhs)
  {
    return VAL(point[collapsed_dim], rhs);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 const legate::Point<DIM>& shape,
                                 const VAL,
                                 const RHS& rhs)
  {
    int64_t idx = 0;
    for (int32_t dim = 0; dim < DIM; ++dim) {
      idx = idx * shape[dim] + point[dim];
    }
    return VAL(idx, rhs);
  }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::ARGMIN, TYPE_CODE> {
  static constexpr bool valid = !legate::is_complex<TYPE_CODE>::value;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = Argval<RHS>;
  using OP  = ArgminReduction<RHS>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 int32_t collapsed_dim,
                                 const VAL,
                                 const RHS& rhs)
  {
    return VAL(point[collapsed_dim], rhs);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 const legate::Point<DIM>& shape,
                                 const VAL,
                                 const RHS& rhs)
  {
    int64_t idx = 0;
    for (int32_t dim = 0; dim < DIM; ++dim) {
      idx = idx * shape[dim] + point[dim];
    }
    return VAL(idx, rhs);
  }
};

template <legate::Type::Code TYPE_CODE>
using enabled_for_floating =
  typename std::enable_if<legate::is_floating_point<TYPE_CODE>::value>::type;

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::NANARGMAX, TYPE_CODE, enabled_for_floating<TYPE_CODE>> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = Argval<RHS>;
  using OP  = ArgmaxReduction<RHS>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 int32_t collapsed_dim,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    return is_nan(rhs) ? identity : VAL(point[collapsed_dim], rhs);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 const legate::Point<DIM>& shape,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    int64_t idx = 0;

    for (int32_t dim = 0; dim < DIM; ++dim) {
      idx = idx * shape[dim] + point[dim];
    }
    return is_nan(rhs) ? identity : VAL(idx, rhs);
  }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::NANARGMIN, TYPE_CODE, enabled_for_floating<TYPE_CODE>> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = Argval<RHS>;
  using OP  = ArgminReduction<RHS>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 int32_t collapsed_dim,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    return is_nan(rhs) ? identity : VAL(point[collapsed_dim], rhs);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>& point,
                                 const legate::Point<DIM>& shape,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    int64_t idx = 0;

    for (int32_t dim = 0; dim < DIM; ++dim) {
      idx = idx * shape[dim] + point[dim];
    }
    return is_nan(rhs) ? identity : VAL(idx, rhs);
  }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::NANMIN, TYPE_CODE, enabled_for_floating<TYPE_CODE>> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MinReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&,
                                 int32_t,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    return is_nan(rhs) ? identity : rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL identity)
  {
    return is_nan(rhs) ? identity : rhs;
  }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::NANMAX, TYPE_CODE, enabled_for_floating<TYPE_CODE>> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::MaxReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&,
                                 int32_t,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    return is_nan(rhs) ? identity : rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS& rhs, const VAL identity)
  {
    return is_nan(rhs) ? identity : rhs;
  }
};

template <legate::Type::Code TYPE_CODE>
using enabled_for_floating_or_complex64 =
  typename std::enable_if<legate::is_floating_point<TYPE_CODE>::value ||
                          TYPE_CODE == legate::Type::Code::COMPLEX64>::type;

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::NANPROD, TYPE_CODE, enabled_for_floating_or_complex64<TYPE_CODE>> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::ProdReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&,
                                 int32_t,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    return is_nan(rhs) ? identity : rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS rhs, const VAL identity)
  {
    return is_nan(rhs) ? identity : rhs;
  }
};

template <legate::Type::Code TYPE_CODE>
using enabled_for_floating_or_complex =
  typename std::enable_if<legate::is_floating_point<TYPE_CODE>::value ||
                          legate::is_complex<TYPE_CODE>::value>::type;

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::NANSUM, TYPE_CODE, enabled_for_floating_or_complex<TYPE_CODE>> {
  static constexpr bool valid = true;

  using RHS = legate::type_of<TYPE_CODE>;
  using VAL = RHS;
  using OP  = legate::SumReduction<VAL>;

  template <bool EXCLUSIVE>
  __CUDA_HD__ static void fold(VAL& a, VAL b)
  {
    OP::template fold<EXCLUSIVE>(a, b);
  }

  template <int32_t DIM>
  __CUDA_HD__ static VAL convert(const legate::Point<DIM>&,
                                 int32_t,
                                 const VAL identity,
                                 const RHS& rhs)
  {
    return is_nan(rhs) ? identity : rhs;
  }

  __CUDA_HD__ static VAL convert(const RHS rhs, const VAL identity)
  {
    return is_nan(rhs) ? identity : rhs;
  }
};

template <legate::Type::Code TYPE_CODE>
struct UnaryRedOp<UnaryRedCode::CONTAINS, TYPE_CODE> {
  // Set to false so that this only gets enabled when expliclty declared valid.
  static constexpr bool valid = false;
  // This class only provides the typedefs necessary to match the other operators.
  // It does not provide fold/convert functions.
  using RHS     = legate::type_of<TYPE_CODE>;
  using VAL     = bool;
  using _RED_OP = UnaryRedOp<UnaryRedCode::SUM, legate::Type::Code::BOOL>;
  using OP      = _RED_OP::OP;
};

}  // namespace cupynumeric
