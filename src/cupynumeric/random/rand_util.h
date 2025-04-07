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
#include "cupynumeric/random/philox.h"

#define HI_BITS(x) (static_cast<unsigned>((x) >> 32))
#define LO_BITS(x) (static_cast<unsigned>((x) & 0x00000000FFFFFFFF))

namespace cupynumeric {

// Match these to RandGenCode in config.py
enum class RandGenCode : int32_t {
  UNIFORM = 1,
  NORMAL  = 2,
  INTEGER = 3,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(RandGenCode gen_code, Functor f, Fnargs&&... args)
{
  switch (gen_code) {
    case RandGenCode::UNIFORM:
      return f.template operator()<RandGenCode::UNIFORM>(std::forward<Fnargs>(args)...);
    case RandGenCode::NORMAL:
      return f.template operator()<RandGenCode::NORMAL>(std::forward<Fnargs>(args)...);
    case RandGenCode::INTEGER:
      return f.template operator()<RandGenCode::INTEGER>(std::forward<Fnargs>(args)...);
    default: LEGATE_ABORT("Unknown RNG generator code");
  }
  return f.template operator()<RandGenCode::UNIFORM>(std::forward<Fnargs>(args)...);
}

template <RandGenCode GEN_CODE, legate::Type::Code CODE>
struct RandomGenerator {
  static constexpr bool valid = false;
};

template <legate::Type::Code CODE>
struct RandomGenerator<RandGenCode::UNIFORM, CODE> {
  using RNG                   = Philox_2x32<10>;
  static constexpr bool valid = CODE == legate::Type::Code::FLOAT64;

  RandomGenerator(uint32_t ep, const std::vector<legate::Scalar>& args) : epoch(ep) {}

  __CUDAPREFIX__ double operator()(uint32_t hi, uint32_t lo) const
  {
    return RNG::rand_double(epoch, hi, lo);
  };

  uint32_t epoch;
};

template <legate::Type::Code CODE>
struct RandomGenerator<RandGenCode::NORMAL, CODE> {
  using RNG                   = Philox_2x32<10>;
  static constexpr bool valid = CODE == legate::Type::Code::FLOAT64;

  RandomGenerator(uint32_t ep, const std::vector<legate::Scalar>& args) : epoch(ep) {}

#ifndef __NVCC__
  static inline double erfinv(double a)
  {
    double p, q, t, fa;
    unsigned long long int l;

    fa = fabs(a);
    if (fa >= 1.0) {
      l = 0xfff8000000000000ull;
      memcpy(&t, &l, sizeof(double)); /* INDEFINITE */
      if (fa == 1.0) {
        t = a * exp(1000.0); /* Infinity */
      }
    } else if (fa >= 0.9375) {
      /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
         Approximations for the Inverse of the Error Function. Mathematics of
         Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 59
       */
      t = log1p(-fa);
      t = 1.0 / sqrt(-t);
      p = 2.7834010353747001060e-3;
      p = p * t + 8.6030097526280260580e-1;
      p = p * t + 2.1371214997265515515e+0;
      p = p * t + 3.1598519601132090206e+0;
      p = p * t + 3.5780402569085996758e+0;
      p = p * t + 1.5335297523989890804e+0;
      p = p * t + 3.4839207139657522572e-1;
      p = p * t + 5.3644861147153648366e-2;
      p = p * t + 4.3836709877126095665e-3;
      p = p * t + 1.3858518113496718808e-4;
      p = p * t + 1.1738352509991666680e-6;
      q = t + 2.2859981272422905412e+0;
      q = q * t + 4.3859045256449554654e+0;
      q = q * t + 4.6632960348736635331e+0;
      q = q * t + 3.9846608184671757296e+0;
      q = q * t + 1.6068377709719017609e+0;
      q = q * t + 3.5609087305900265560e-1;
      q = q * t + 5.3963550303200816744e-2;
      q = q * t + 4.3873424022706935023e-3;
      q = q * t + 1.3858762165532246059e-4;
      q = q * t + 1.1738313872397777529e-6;
      t = p / (q * t);
      if (a < 0.0) {
        t = -t;
      }
    } else if (fa >= 0.75) {
      /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
         Approximations for the Inverse of the Error Function. Mathematics of
         Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 39
      */
      t = a * a - .87890625;
      p = .21489185007307062000e+0;
      p = p * t - .64200071507209448655e+1;
      p = p * t + .29631331505876308123e+2;
      p = p * t - .47644367129787181803e+2;
      p = p * t + .34810057749357500873e+2;
      p = p * t - .12954198980646771502e+2;
      p = p * t + .25349389220714893917e+1;
      p = p * t - .24758242362823355486e+0;
      p = p * t + .94897362808681080020e-2;
      q = t - .12831383833953226499e+2;
      q = q * t + .41409991778428888716e+2;
      q = q * t - .53715373448862143349e+2;
      q = q * t + .33880176779595142685e+2;
      q = q * t - .11315360624238054876e+2;
      q = q * t + .20369295047216351160e+1;
      q = q * t - .18611650627372178511e+0;
      q = q * t + .67544512778850945940e-2;
      p = p / q;
      t = a * p;
    } else {
      /* Based on: J.M. Blair, C.A. Edwards, J.H. Johnson: Rational Chebyshev
         Approximations for the Inverse of the Error Function. Mathematics of
         Computation, Vol. 30, No. 136 (Oct. 1976), pp. 827-830. Table 18
      */
      t = a * a - .5625;
      p = -.23886240104308755900e+2;
      p = p * t + .45560204272689128170e+3;
      p = p * t - .22977467176607144887e+4;
      p = p * t + .46631433533434331287e+4;
      p = p * t - .43799652308386926161e+4;
      p = p * t + .19007153590528134753e+4;
      p = p * t - .30786872642313695280e+3;
      q = t - .83288327901936570000e+2;
      q = q * t + .92741319160935318800e+3;
      q = q * t - .35088976383877264098e+4;
      q = q * t + .59039348134843665626e+4;
      q = q * t - .48481635430048872102e+4;
      q = q * t + .18997769186453057810e+4;
      q = q * t - .28386514725366621129e+3;
      p = p / q;
      t = a * p;
    }
    return t;
  }
#endif

  __CUDAPREFIX__ double operator()(uint32_t hi, uint32_t lo) const
  {
    return erfinv(2.0 * RNG::rand_double(epoch, hi, lo) - 1.0);
  };

  uint32_t epoch;
};

template <legate::Type::Code CODE>
struct RandomGenerator<RandGenCode::INTEGER, CODE> {
  using RNG = Philox_2x32<10>;
  using VAL = legate::type_of<CODE>;

  static constexpr bool valid = legate::is_integral<CODE>::value;

  RandomGenerator(uint32_t ep, const std::vector<legate::Scalar>& args) : epoch(ep)
  {
    assert(args.size() == 2);
    lo   = args[0].value<VAL>();
    diff = args[1].value<VAL>() - lo;
  }

  __CUDAPREFIX__ double operator()(uint32_t hi_bits, uint32_t lo_bits) const
  {
    return static_cast<VAL>(lo + RNG::rand_long(epoch, hi_bits, lo_bits, diff));
  };

  uint32_t epoch;
  VAL lo;
  uint64_t diff;
};

}  // namespace cupynumeric
