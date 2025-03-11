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
#include "cupynumeric/bits/bits_util.h"

namespace cupynumeric {

template <Bitorder BITORDER, bool ALIGNED>
struct Pack;

template <>
struct Pack<Bitorder::BIG, true /*ALIGNED*/> {
  template <typename VAL, int32_t DIM>
  __CUDA_HD__ inline uint8_t operator()(legate::AccessorRO<VAL, DIM> in,
                                        legate::Point<DIM> p,
                                        int64_t in_hi_axis,
                                        uint32_t axis)
  {
    int64_t in_lo = p[axis] * 8;
    int64_t in_hi = in_lo + 8;
    uint8_t acc   = 0;
    for (int64_t c = in_lo; c < in_hi; ++c) {
      p[axis] = c;
      acc     = (acc << 1) | static_cast<uint8_t>(in[p] != 0);
    }
    return acc;
  }
};

template <>
struct Pack<Bitorder::BIG, false /*ALIGNED*/> {
  template <typename VAL, int32_t DIM>
  __CUDA_HD__ inline uint8_t operator()(legate::AccessorRO<VAL, DIM> in,
                                        legate::Point<DIM> p,
                                        int64_t in_hi_axis,
                                        uint32_t axis)
  {
    int64_t in_lo = p[axis] * 8;
    int64_t in_hi = std::min<int64_t>(in_lo + 8, in_hi_axis + 1);
    uint8_t acc   = 0;
    for (int64_t c = in_lo; c < in_hi; ++c) {
      p[axis] = c;
      acc     = (acc << 1) | static_cast<uint8_t>(in[p] != 0);
    }
    acc <<= 8 - (in_hi - in_lo);
    return acc;
  }
};

template <>
struct Pack<Bitorder::LITTLE, true /*ALIGNED*/> {
  template <typename VAL, int32_t DIM>
  __CUDA_HD__ inline uint8_t operator()(legate::AccessorRO<VAL, DIM> in,
                                        legate::Point<DIM> p,
                                        int64_t in_hi_axis,
                                        uint32_t axis)
  {
    int64_t in_lo = p[axis] * 8;
    int64_t in_hi = in_lo + 7;
    uint8_t acc   = 0;
    for (int64_t c = in_hi; c >= in_lo; --c) {
      p[axis] = c;
      acc     = (acc << 1) | static_cast<uint8_t>(in[p] != 0);
    }
    return acc;
  }
};

template <>
struct Pack<Bitorder::LITTLE, false /*ALIGNED*/> {
  template <typename VAL, int32_t DIM>
  __CUDA_HD__ inline uint8_t operator()(legate::AccessorRO<VAL, DIM> in,
                                        legate::Point<DIM> p,
                                        int64_t in_hi_axis,
                                        uint32_t axis)
  {
    int64_t in_lo = p[axis] * 8;
    int64_t in_hi = std::min<int64_t>(in_lo + 7, in_hi_axis);
    uint8_t acc   = 0;
    for (int64_t c = in_hi; c >= in_lo; --c) {
      p[axis] = c;
      acc     = (acc << 1) | static_cast<uint8_t>(in[p] != 0);
    }
    return acc;
  }
};

class PackbitsTask : public CuPyNumericTask<PackbitsTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_PACKBITS}};

 public:
  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace cupynumeric
