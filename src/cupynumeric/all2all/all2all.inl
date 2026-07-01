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

// Shared by all2all_gather.cu and all2all_scatter.cu. Include after
// all2all.cuh and zip_template.inl (for check_index_arrays_out_of_bounds).

namespace cupynumeric::detail {

struct IndexScalarArgs {
  int64_t key_dim     = 0;
  int64_t start_index = 0;
  legate::DomainPoint shape{};
  bool check_bounds = false;
};

// Fused all2all tasks always publish scalar(2) = narrays.  The remaining
// metadata is only present (and read) when narrays > 0: key_dim (3),
// start_index (4), shape_ndim (5), shape extents (6 .. 6+shape_ndim-1),
// and check_bounds (6+shape_ndim).  Shape extents are plain int64 scalars
// rather than a DomainPoint future so Legion's dynamic type check succeeds
// in index-space launches.
inline IndexScalarArgs read_index_scalar_args(legate::TaskContext& context, int64_t narrays)
{
  IndexScalarArgs result{};
  if (narrays <= 0) {
    return result;
  }

  result.key_dim     = context.scalar(3).value<int64_t>();
  result.start_index = context.scalar(4).value<int64_t>();

  const auto shape_ndim = context.scalar(5).value<int64_t>();

  result.shape.dim = static_cast<int>(shape_ndim);
  for (int64_t i = 0; i < shape_ndim; ++i) {
    result.shape[static_cast<int>(i)] = context.scalar(6 + i).value<int64_t>();
  }
  result.check_bounds = context.scalar(6 + shape_ndim).value<bool>();
  return result;
}

// Builds the FusedIndexLoaderProvider in FB memory from `narrays` per-dim
// int64 input stores. Mirrors the ZIPGATHER GPU body (see index/zipgather.cu):
// the accessor table lives in FB (sparse loader + bounds-check pre-pass); the
// dense pointer table is populated only when every index array is dense.
//
// When `check_bounds` is set this helper additionally performs a collective
// OOB check across ranks (single uint8 ncclAllReduce via `nccl_any_bool`)
// and throws a legate::TaskException symmetrically on every rank if any
// rank observed an out-of-bounds index.  `nccl_comm` is therefore required
// even when the bounds check itself ends up being skipped — the caller has
// already entered the index-space launch and the helper must be free to
// synchronize with peers.
template <int DIM_input, int DIM_output>
[[nodiscard]] FusedIndexLoaderProvider<DIM_input, DIM_output> build_fused_index_provider(
  TaskContext& context,
  int64_t narrays,
  int64_t key_dim,
  int64_t start_index,
  const legate::DomainPoint& shape,
  bool check_bounds,
  const legate::Rect<DIM_output>& index_rect,
  ncclComm_t* nccl_comm,
  cudaStream_t stream)
{
  std::vector<AccessorRO<int64_t, DIM_output>> host_accs;
  host_accs.reserve(static_cast<size_t>(narrays));
  bool all_dense = true;
  std::vector<const int64_t*> host_ptrs(static_cast<size_t>(narrays), nullptr);
  for (int64_t i = 0; i < narrays; ++i) {
    legate::PhysicalStore index_store{context.input(static_cast<uint32_t>(1 + i))};
    auto acc = index_store.read_accessor<int64_t, DIM_output>(index_rect);
    host_accs.push_back(acc);
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    if (!index_rect.empty() && acc.accessor.is_dense_row_major(index_rect)) {
      host_ptrs[static_cast<size_t>(i)] = acc.ptr(index_rect);
    } else {
      all_dense = false;
    }
#else
    all_dense = false;
#endif
  }

  auto index_acc_buf =
    create_buffer<AccessorRO<int64_t, DIM_output>, 1>(narrays, Memory::Kind::GPU_FB_MEM);
  if (narrays > 0) {
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(index_acc_buf.ptr(0),
                                           host_accs.data(),
                                           sizeof(AccessorRO<int64_t, DIM_output>) * narrays,
                                           cudaMemcpyHostToDevice,
                                           stream));
  }

  // Dense pointer table must live in FB and be populated via a stream-ordered
  // cudaMemcpyAsync. Filling a Z_COPY buffer with raw host stores and then
  // having a kernel on `stream` dereference those entries is a data race:
  // there is no memory barrier ordering the CPU store relative to the GPU
  // load, so on weakly-ordered systems (aarch64) the kernel can pick up the
  // initial zero (or stale bytes), turning index_ptrs[k][...] into an
  // illegal-address dereference. The accessor table above uses the same
  // pattern for the same reason.
  Buffer<const int64_t*, 1> index_ptr_buf;
  if (all_dense) {
    index_ptr_buf = create_buffer<const int64_t*, 1>(narrays, Memory::Kind::GPU_FB_MEM);
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(index_ptr_buf.ptr(0),
                                           host_ptrs.data(),
                                           sizeof(const int64_t*) * narrays,
                                           cudaMemcpyHostToDevice,
                                           stream));
  }

  // Symmetric bounds check.  Each rank evaluates its local shard of the
  // index arrays into a single uint8 OOB flag (or 0 when the local shard
  // is empty), then we collectively OR-reduce that flag across ranks via
  // `nccl_any_bool` and throw a legate::TaskException on every rank if any
  // rank observed an out-of-bounds index.
  if (check_bounds) {
    bool local_oob = false;

    Pitches<DIM_output - 1> bounds_pitches;
    const size_t bounds_volume = bounds_pitches.flatten(index_rect);
    if (bounds_volume > 0) {
      local_oob = check_index_arrays_out_of_bounds<DIM_output>(index_acc_buf,
                                                               static_cast<int64_t>(bounds_volume),
                                                               index_rect,
                                                               bounds_pitches,
                                                               narrays,
                                                               start_index,
                                                               shape,
                                                               stream);
    }

    if (nccl_any_bool(context, local_oob, nccl_comm, stream)) {
      throw legate::TaskException("index is out of bounds in index array");
    }
  }

  FusedIndexLoaderProvider<DIM_input, DIM_output> provider{};
  provider.index_accs = index_acc_buf;
  // index_ptrs is only consumed when provider.dense is true. Leave the default-
  // constructed buffer in place otherwise so we never store an unallocated buf.
  if (all_dense) {
    provider.index_ptrs = index_ptr_buf;
  }
  provider.dense       = all_dense;
  provider.narrays     = narrays;
  provider.key_dim     = key_dim;
  provider.start_index = start_index;
  provider.shape       = shape;
  provider.pitches.flatten(index_rect);
  provider.lo = index_rect.lo;
  return provider;
}

}  // namespace cupynumeric::detail
