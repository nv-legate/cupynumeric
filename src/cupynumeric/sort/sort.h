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

#include <thrust/sort.h>

namespace cupynumeric {

struct SortArgs {
  legate::PhysicalStore input;
  legate::PhysicalStore output;
  bool argsort;
  bool stable;
  size_t segment_size_g;
  bool is_index_space;  // !single_task
  size_t local_rank;
  size_t num_ranks;
  size_t num_sort_ranks;
};

template <typename VAL>
struct SegmentSample {
  VAL value;
  size_t segment;
  int32_t rank;
  size_t position;
};

template <typename VAL>
struct SortPiece {
  legate::Buffer<VAL> values;
  legate::Buffer<int64_t> indices;
  size_t size;
};

template <typename VAL>
struct SegmentMergePiece {
  legate::Buffer<size_t> segments;
  legate::Buffer<VAL> values;
  legate::Buffer<int64_t> indices;
  size_t size;
};

template <typename VAL>
struct SegmentSampleComparator
  : public thrust::binary_function<SegmentSample<VAL>, SegmentSample<VAL>, bool> {
  __CUDA_HD__ bool operator()(const SegmentSample<VAL>& lhs, const SegmentSample<VAL>& rhs) const
  {
    if (lhs.segment != rhs.segment) {
      return lhs.segment < rhs.segment;
    } else {
      // special case for unused samples
      if (lhs.rank < 0 || rhs.rank < 0) {
        return rhs.rank < 0 && lhs.rank >= 0;
      }

      if (lhs.value != rhs.value) {
        return lhs.value < rhs.value;
      } else if (lhs.rank != rhs.rank) {
        return lhs.rank < rhs.rank;
      } else {
        return lhs.position < rhs.position;
      }
    }
  }
};

struct modulusWithOffset : public thrust::binary_function<int64_t, int64_t, int64_t> {
  const size_t constant;

  modulusWithOffset(size_t _constant) : constant(_constant) {}

  __CUDA_HD__ int64_t operator()(const int64_t& lhs, const int64_t& rhs) const
  {
    return lhs % rhs + constant;
  }
};

class SortTask : public CuPyNumericTask<SortTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{CUPYNUMERIC_SORT};

  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);
  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);
  static constexpr auto OMP_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);

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
