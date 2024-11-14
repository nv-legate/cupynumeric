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

namespace cupynumeric {

// This is a small helper class that will also work if we have zero-sized arrays
// We also need to have this instead of std::array so that it works on devices
template <int DIM, bool C_ORDER = true>
class Pitches {
 public:
  __CUDA_HD__
  inline size_t flatten(const legate::Rect<DIM + 1>& rect)
  {
    size_t pitch  = 1;
    size_t volume = 1;
    for (int d = DIM; d >= 0; --d) {
      // Quick exit for empty rectangle dimensions
      if (rect.lo[d] > rect.hi[d]) {
        return 0;
      }
      const size_t diff = rect.hi[d] - rect.lo[d] + 1;
      volume *= diff;
      if (d > 0) {
        pitch *= diff;
        pitches[d - 1] = pitch;
      }
    }
    return volume;
  }
  __CUDA_HD__
  inline legate::Point<DIM + 1> unflatten(size_t index, const legate::Point<DIM + 1>& lo) const
  {
    legate::Point<DIM + 1> point = lo;
    for (int d = 0; d < DIM; d++) {
      point[d] += index / pitches[d];
      index = index % pitches[d];
    }
    point[DIM] += index;
    return point;
  }

 private:
  size_t pitches[DIM];
};

template <int DIM>
class Pitches<DIM, false /*C_ORDER*/> {
 public:
  __CUDA_HD__
  inline size_t flatten(const legate::Rect<DIM + 1>& rect)
  {
    size_t pitch  = 1;
    size_t volume = 1;
    for (int d = 0; d <= DIM; ++d) {
      // Quick exit for empty rectangle dimensions
      if (rect.lo[d] > rect.hi[d]) {
        return 0;
      }
      const size_t diff = rect.hi[d] - rect.lo[d] + 1;
      volume *= diff;
      if (d < DIM) {
        pitch *= diff;
        pitches[d] = pitch;
      }
    }
    return volume;
  }
  __CUDA_HD__
  inline legate::Point<DIM + 1> unflatten(size_t index, const legate::Point<DIM + 1>& lo) const
  {
    legate::Point<DIM + 1> point = lo;
    for (int d = DIM - 1; d >= 0; --d) {
      point[d + 1] += index / pitches[d];
      index = index % pitches[d];
    }
    point[0] += index;
    return point;
  }

 private:
  size_t pitches[DIM];
};

// Specialization for the zero-sized case
template <bool C_ORDER>
class Pitches<0, C_ORDER> {
 public:
  __CUDA_HD__
  inline size_t flatten(const legate::Rect<1>& rect)
  {
    if (rect.lo[0] > rect.hi[0]) {
      return 0;
    } else {
      return (rect.hi[0] - rect.lo[0] + 1);
    }
  }
  __CUDA_HD__
  inline legate::Point<1> unflatten(size_t index, const legate::Point<1>& lo) const
  {
    legate::Point<1> point = lo;
    point[0] += index;
    return point;
  }
};

}  // namespace cupynumeric
