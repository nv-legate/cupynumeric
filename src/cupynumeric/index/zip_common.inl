/* Copyright 2026 NVIDIA Corporation
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

// Shared helpers used by both zipgather and zipscatter tasks. Gather and
// scatter compute the same source-side Point<N> from a base Point<DIM> plus
// per-dim index arrays — only the read/write direction differs — so these
// pieces live in a neutral header rather than under either side.

#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/index/zip.h"

namespace cupynumeric {

using namespace legate;

template <int32_t DIM>
inline bool is_dense_row_major(const size_t strides[DIM], const Rect<DIM>& rect, size_t elem_size)
{
  size_t expected = elem_size;
  for (int d = DIM - 1; d >= 0; --d) {
    if (strides[d] != expected) {
      return false;
    }
    expected *= (rect.hi[d] - rect.lo[d] + 1);
  }
  return true;
}

// Loads index values from a per-dim accessor at a Point<DIM>. Storage is
// templated so the same struct works for std::vector<AccessorRO<...>> on the
// CPU/OMP path and Buffer<AccessorRO<...>, 1> on the GPU path.
template <int DIM, typename Storage>
struct SparseIndexLoader {
  Storage index_arrays;

  LEGATE_HOST_DEVICE int64_t load(size_t dim, const Point<DIM>& p, size_t /*idx*/) const
  {
    return index_arrays[dim][p];
  }
};

// Loads index values from per-dim raw int64_t pointers using a flat index.
// Storage may be std::vector<const int64_t*> or Buffer<const int64_t*, 1>.
template <int DIM, typename Storage>
struct DenseIndexLoader {
  Storage index_ptrs;

  LEGATE_HOST_DEVICE int64_t load(size_t dim, const Point<DIM>& /*p*/, size_t idx) const
  {
    return index_ptrs[dim][idx];
  }
};

// Functor wrapping ``compute_idx_cuda`` so it can be passed as the
// ComputeIndexFn template parameter of ``build_source_point``.  Shared by the
// single-GPU zipgather/zipscatter kernels and the fused all2all path.
struct ComputeIdxCudaFn {
  LEGATE_HOST_DEVICE legate::coord_t operator()(legate::coord_t index, legate::coord_t extent) const
  {
    return compute_idx_cuda(index, extent);
  }
};

// Builds the source-side Point<N> consumed by zip / zipgather / zipscatter
// kernels.
//
// When narrays == N every output dimension is provided by an index array.
// Otherwise the leading [0, start_index) dimensions are copied from p, the
// next narrays come from the index arrays, and the trailing dimensions come
// from broadcast positions in p (offset by key_dim - narrays).
//
// compute_index normalizes a raw index against an extent. CPU/OMP variants
// pass a lambda that may signal out-of-bounds; the GPU variant passes a
// device functor wrapping compute_idx_cuda.
template <int DIM, int N, typename Loader, typename ComputeIndexFn>
LEGATE_HOST_DEVICE inline Point<N> build_source_point(const Point<DIM>& p,
                                                      const Loader& loader,
                                                      size_t flat_idx,
                                                      int64_t narrays,
                                                      int64_t key_dim,
                                                      int64_t start_index,
                                                      const DomainPoint& shape,
                                                      const ComputeIndexFn& compute_index)
{
  Point<N> new_point;

  if (narrays == N) {
    for (size_t i = 0; i < N; ++i) {
      new_point[i] = compute_index(loader.load(i, p, flat_idx), shape[i]);
    }
  } else {
    for (int64_t i = 0; i < start_index; ++i) {
      new_point[i] = p[i];
    }
    for (size_t i = 0; i < static_cast<size_t>(narrays); ++i) {
      const auto dim = start_index + i;
      new_point[dim] = compute_index(loader.load(i, p, flat_idx), shape[dim]);
    }
    for (size_t i = start_index + narrays; i < N; ++i) {
      const int64_t j = key_dim + i - narrays;
      new_point[i]    = p[j];
    }
  }

  return new_point;
}

}  // namespace cupynumeric
