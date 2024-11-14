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

// An implementation of DE Shaw's Philox 2x32 PRNG

#ifndef __CUDAPREFIX__
#ifdef __NVCC__
#define __CUDAPREFIX__ __device__ __forceinline__
#else
#define __CUDAPREFIX__
#endif
#endif

namespace cupynumeric {

template <int ROUNDS>
class Philox_2x32 {
 public:
  typedef unsigned u32;
  typedef unsigned long long u64;

  static const u32 PHILOX_M2x32_0x = 0xD256D193U;
  static const u32 PHILOX_W32_0x   = 0x9E3779B9U;

  __CUDAPREFIX__
  static u64 rand_raw(u32 key, u32 ctr_hi, u32 ctr_lo)
  {
#ifdef __NVCC__
#pragma unroll
#endif
    for (int i = 0; i < ROUNDS; i++) {
      u32 prod_hi, prod_lo;
#ifdef __NVCC__
      prod_hi = __umulhi(ctr_lo, PHILOX_M2x32_0x);
      prod_lo = ctr_lo * PHILOX_M2x32_0x;
#else
      u64 prod = u64{ctr_lo} * PHILOX_M2x32_0x;
      prod_hi  = prod >> 32;
      prod_lo  = prod;
#endif
      ctr_lo = ctr_hi ^ key ^ prod_hi;
      ctr_hi = prod_lo;
      key += PHILOX_W32_0x;
    }
    return (u64{ctr_hi} << 32) + ctr_lo;
  }

  // Helper function for CPU hi 64-bit multiplication
  static inline u64 mul64hi(u64 op1, u64 op2)
  {
    u64 u1 = (op1 & 0xffffffff);
    u64 v1 = (op2 & 0xffffffff);
    u64 t  = u1 * v1;
    u64 k  = (t >> 32);

    op1 >>= 32;
    t      = (op1 * v1) + k;
    k      = (t & 0xffffffff);
    u64 w1 = (t >> 32);

    op2 >>= 32;
    t = (u1 * op2) + k;
    k = (t >> 32);

    return (op1 * op2) + w1 + k;
  }

  // returns an unsigned 32-bit integer in the range [0, n)
  __CUDAPREFIX__
  static u32 rand_int(u32 key, u32 ctr_hi, u32 ctr_lo, u32 n)
  {
    // need 32 random bits
    u32 bits = rand_raw(key, ctr_hi, ctr_lo);
    // now treat them as a 0.32 fixed-point value, multiply by n and truncate
#ifdef __NVCC__
    return __umulhi(bits, n);
#else
    return (u64{bits} * u64{bits}) >> 32;
#endif
  }

  // returns an unsigned 64-bit integer in the range [0, n)
  __CUDAPREFIX__
  static u64 rand_long(u32 key, u32 ctr_hi, u32 ctr_lo, u64 n)
  {
    // need 64 random bits
    u64 bits = rand_raw(key, ctr_hi, ctr_lo);
    // now treat them as a 0.64 fixed-point value, multiply by n and truncate
#ifdef __NVCC__
    return __umul64hi(bits, n);
#else
    return mul64hi(bits, n);
#endif
  }

  // returns a float in the range [0.0, 1.0)
  __CUDAPREFIX__
  static double rand_float(u32 key, u32 ctr_hi, u32 ctr_lo)
  {
    // need 32 random bits (we probably lose a bunch when this gets converted to float)
    u32 bits = rand_raw(key, ctr_hi, ctr_lo);
#if __cplusplus > 201402L
    // This syntax is only supported on >= c++17
    const float scale = 0x1.p-32;  // 2^-32
#else
    const float scale  = 0.00000000023283064365386962890625;
#endif
    return (bits * scale);
  }

  // returns a double in the range [0.0, 1.0)
  __CUDAPREFIX__
  static double rand_double(u32 key, u32 ctr_hi, u32 ctr_lo)
  {
    // need 64 random bits (we probably lose a bunch when this gets converted to float)
    u64 bits = rand_raw(key, ctr_hi, ctr_lo);
#if __cplusplus > 201402L
    // This syntax is only supported on >= c++17
    const double scale = 0x1.p-64;  // 2^-64
#else
    const double scale = 0.0000000000000000000542101086242752217003726400434970855712890625;
#endif
    return (bits * scale);
  }
};

}  // namespace cupynumeric
