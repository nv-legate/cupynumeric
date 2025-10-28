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

// Useful for IDEs
#include "arg.h"

namespace cupynumeric {

template <typename T>
__CUDA_HD__ Argval<T>::Argval(T v) : arg(LLONG_MIN), arg_value(v)
{
}

template <typename T>
__CUDA_HD__ Argval<T>::Argval(int64_t a, T v) : arg(a), arg_value(v)
{
}

template <typename T>
__CUDA_HD__ Argval<T>::Argval(const Argval& other) : arg(other.arg), arg_value(other.arg_value)
{
}

template <typename T>
template <typename REDOP, bool EXCLUSIVE>
__CUDA_HD__ inline void Argval<T>::apply(const Argval<T>& rhs)
{
  if (EXCLUSIVE) {
    // This is the easy case
    T copy = arg_value;
    REDOP::template fold<true>(copy, rhs.arg_value);
    if (copy != arg_value) {
      arg_value = copy;
      arg       = rhs.arg;
    }
  } else {
    // Handle conflicts here
#ifdef __CUDA_ARCH__
    const unsigned long long guard = (unsigned long long)-1LL;
    unsigned long long* ptr        = (unsigned long long*)&arg;
    union {
      long long as_signed;
      unsigned long long as_unsigned;
    } next, current;
    next.as_signed = *ptr;
    do {
      current.as_signed = next.as_signed;
      next.as_unsigned  = atomicCAS(ptr, current.as_unsigned, guard);
    } while ((next.as_signed != current.as_signed) || (next.as_signed == -1LL));
    // Memory fence to prevent the compiler from hoisting the load
    __threadfence();
    // Once we get here then we can do our comparison
    T copy = arg_value;
    REDOP::template fold<true>(copy, rhs.arg_value);
    if (copy != arg_value) {
      arg_value = copy;
      // Memory fence to make sure that things are ordered
      __threadfence();
      // Write back the rhs args since the value changed
      // We know the value is minus 1 so this is guaranteed to succeed
      next.as_signed = rhs.arg;
      atomicCAS(ptr, guard, next.as_unsigned);
    } else {
      // Write back our arg since the value is the same
      // We know the value is minus 1 so this is guaranteed to succeed
      atomicCAS(ptr, guard, next.as_unsigned);
    }
#else
    // Spin until no one else is doing their comparison
    // We use -1 as a guard to indicate we're doing our
    // comparison since we know all indexes should be >= 0
    volatile long long* ptr = reinterpret_cast<volatile long long*>(&arg);
    long long next          = *ptr;
    long long current;
    do {
      current = next;
      next    = __sync_val_compare_and_swap(ptr, current, -1);
    } while ((next != current) || (next == -1));
    // Memory fence to prevent the compiler from hoisting the load
    __sync_synchronize();
    // Once we get here then we can do our comparison
    T copy = arg_value;
    REDOP::template fold<true>(copy, rhs.arg_value);
    if (copy != arg_value) {
      arg_value = copy;
      // Memory fence to make sure things are ordered
      __sync_synchronize();
      // Write back the rhs args since the value changed
      // We know the value is minus 1 so this is guaranteed to succeed
      __sync_val_compare_and_swap(ptr, -1, rhs.arg);
    } else {
      // Write back our arg since the value is the same
      // We know the value is minus 1 so this is guaranteed to succeed
      __sync_val_compare_and_swap(ptr, -1, next);
    }
#endif
  }
}

// Declare these here, to work around undefined-var-template warnings

#define DECLARE_ARGMAX_IDENTITY(TYPE) \
  template <>                         \
  const Argval<TYPE> ArgmaxReduction<TYPE>::identity;

#define DECLARE_ARGMIN_IDENTITY(TYPE) \
  template <>                         \
  const Argval<TYPE> ArgminReduction<TYPE>::identity;

#define DECLARE_IDENTITIES(TYPE) \
  DECLARE_ARGMAX_IDENTITY(TYPE)  \
  DECLARE_ARGMIN_IDENTITY(TYPE)

DECLARE_IDENTITIES(legate::Half)
DECLARE_IDENTITIES(float)
DECLARE_IDENTITIES(double)
DECLARE_IDENTITIES(bool)
DECLARE_IDENTITIES(int8_t)
DECLARE_IDENTITIES(int16_t)
DECLARE_IDENTITIES(int32_t)
DECLARE_IDENTITIES(int64_t)
DECLARE_IDENTITIES(uint8_t)
DECLARE_IDENTITIES(uint16_t)
DECLARE_IDENTITIES(uint32_t)
DECLARE_IDENTITIES(uint64_t)

#undef DECLARE_IDENTITIES
#undef DECLARE_ARGMIN_IDENTITY
#undef DECLARE_ARGMAX_IDENTITY

}  // namespace cupynumeric
