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

#include "legate.h"

namespace cupynumeric {

template <typename T>
class Argval {
 public:
  // Calling this constructor manually is unsafe, as the members are left uninitialized.
  // This constructor exists only to make nvcc happy when we use a shared memory of Argval<T>.
  __CUDA_HD__
  Argval() {}
  __CUDA_HD__
  Argval(T value);
  __CUDA_HD__
  Argval(int64_t arg, T value);
  __CUDA_HD__
  Argval(const Argval& other);

 public:
  template <typename REDOP, bool EXCLUSIVE>
  __CUDA_HD__ inline void apply(const Argval<T>& rhs);

 public:
  __CUDA_HD__ Argval& operator=(const Argval& other)
  {
    arg       = other.arg;
    arg_value = other.arg_value;
    return *this;
  }
  constexpr bool operator!=(Argval& other) const
  {
    return arg != other.arg || arg_value != other.arg_value;
  }

 public:
  int64_t arg;
  T arg_value;
};

template <typename T>
class ArgmaxReduction {
 public:
  using LHS = Argval<T>;
  using RHS = Argval<T>;

  static const Argval<T> identity;

  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs)
  {
    lhs.template apply<legate::MaxReduction<T>, EXCLUSIVE>(rhs);
  }
  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2)
  {
    rhs1.template apply<legate::MaxReduction<T>, EXCLUSIVE>(rhs2);
  }
};

template <typename T>
class ArgminReduction {
 public:
  using LHS = Argval<T>;
  using RHS = Argval<T>;

  static const Argval<T> identity;

  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs)
  {
    lhs.template apply<legate::MinReduction<T>, EXCLUSIVE>(rhs);
  }
  template <bool EXCLUSIVE>
  __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2)
  {
    rhs1.template apply<legate::MinReduction<T>, EXCLUSIVE>(rhs2);
  }
};

}  // namespace cupynumeric

#include "cupynumeric/arg.inl"
