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

// Shared helpers for the NCCL all-to-all shuffle pipeline. Layout mirrors
// cupynumeric/sort/cub_sort.cuh: everything lives in cupynumeric::detail.
// See all2all_gather.cu and all2all_scatter.cu for the orchestrators that
// drive these helpers.
//
// Distributed gather and scatter via all-to-all shuffle (NCCL)
//
// Two operations share the same 5-step pipeline. Only Step 4 (NCCL direction)
// and Step 5 (which side performs the indirect write) differ.
//
//   All2AllTask        (gather):  output[p]        = source[index[p]]
//   All2AllScatterTask (scatter): output[index[p]] = source[p]
//
// In both cases one of the arrays is partitioned across N ranks. For gather
// it is `source` (the read-from array); for scatter it is `output` (the
// write-into array). The "partitioned array" below refers to whichever of
// the two is sharded.
//
// Step 1: Discover partition layout (AllGather)
//   Each rank AllGathers its local partition Rect<DIM> so every rank knows
//   which rank owns which region of the partitioned array.
//
// Step 2: Classify requests & build ShuffleDescriptor
//   a) assign_target_ranks_kernel — For each local index point, determine
//      which rank owns the corresponding cell of the partitioned array by
//      testing containment in the gathered rects.
//   b) cub::DeviceHistogram::HistogramEven — Count how many requests go
//      to each rank (send_counts_per_rank) directly from the unsorted target ranks.
//   c) Group request indices into per-rank buckets (pack_indices_by_rank_warp)
//   d) Exchange histograms via NCCL (pairwise send/recv of one uint32 per
//      rank pair).  After this every rank knows:
//        send_counts_per_rank[r] = how many requests I send to rank r
//        receive_counts_per_rank[r] = how many requests rank r sends to me
//   e) Prefix-sum the histograms to get send_offsets_per_rank and receive_offsets_per_rank.
//
// Step 3: Linearize & exchange offsets
//   a) linearize_offsets_kernel — Convert each Point<DIM> into a uint64
//      flat offset relative to the target rank's rect.  This reduces the
//      per-element payload from sizeof(Point<DIM>) (e.g. 24 bytes for 3D)
//      to 8 bytes.
//   b) Exchange the flat offsets via NCCL pairwise send/recv (requester →
//      owner). After this, each owning rank knows which elements its local
//      partition is referenced from.
//
// Step 4: Pack local values & exchange them
//   gather:  Owner reads its local source at recv_flat_offsets, then
//            ncclSend's the values back to the requester (owner → requester).
//   scatter: Requester reads its local source at request_positions, then
//            ncclSend's the values to the owner (requester → owner).
//   Both paths share gather_data_by_offset_kernel (a "permuted gather" that
//   turns a local store + permutation array into a packed staging buffer)
//   and the same NCCL group send/recv code; only the direction and which
//   counts/offsets gate each side differ.
//
// Step 5: Indirect-write received values into the local output
//   gather:  Requester writes recv_staging into its local output at
//            request_positions.
//   scatter: Owner writes recv_data into its local output at recv_flat_offsets.
//   Both share unpack_recv_data_kernel.

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/utilities/thrust_util.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <vector>

#include <cub/device/device_histogram.cuh>
#include <thrust/execution_policy.h>

namespace cupynumeric {
namespace detail {

using namespace legate;

// ============================================================================
// Utilities
// ============================================================================

// Bundles a Legate accessor with its Pitches, origin point, and an optional
// dense pointer.  When dense_ptr is non-null the store is known to be dense
// row-major and kernels can bypass the accessor/unflatten overhead.
enum class AccessMode { Read, Write };

template <AccessMode MODE, typename ElemType, int DIM>
struct StoreView {
  using Acc_t = std::
    conditional_t<MODE == AccessMode::Read, AccessorRO<ElemType, DIM>, AccessorRW<ElemType, DIM>>;
  using Ptr_t = std::conditional_t<MODE == AccessMode::Read, const ElemType*, ElemType*>;

  Acc_t acc;
  Pitches<DIM - 1> pitches;
  legate::Point<DIM> lo;
  Ptr_t dense_ptr = nullptr;

  StoreView(Acc_t acc, const legate::Rect<DIM>& rect) : acc(acc)
  {
    pitches.flatten(rect);
    lo = rect.lo;
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    dense_ptr = !rect.empty() && acc.accessor.is_dense_row_major(rect) ? acc.ptr(rect) : nullptr;
#endif
  }

  bool is_dense() const { return dense_ptr != nullptr; }
};

// Device functors that abstract the accessor-vs-dense point loading.
// Used by assign_target_ranks and linearize_offsets kernels.
template <int DIM_input, int DIM_output>
struct AccessorPointLoader {
  AccessorRO<legate::Point<DIM_input>, DIM_output> acc;
  Pitches<DIM_output - 1> pitches;
  legate::Point<DIM_output> lo;

  __device__ legate::Point<DIM_input> operator()(size_t flat_idx) const
  {
    return acc[pitches.unflatten(flat_idx, lo)];
  }
};

template <int DIM_input>
struct DensePointLoader {
  const legate::Point<DIM_input>* __restrict__ ptr;

  __device__ legate::Point<DIM_input> operator()(size_t flat_idx) const { return ptr[flat_idx]; }
};

// Stores the rect's low corner and pre-computed pitches for the linearization.
// Only `lo` is needed at linearization time; `hi` is used on the host side to
// compute the pitches and is then thrown away. Pitches<DIM-1> holds the
// row-major strides for a Rect<DIM>; the Pitches<0> specialization makes the
// DIM == 1 case a no-storage zero-pitch struct automatically.
template <int DIM>
struct LinearizedRectInfo {
  legate::Point<DIM> lo;
  Pitches<DIM - 1> pitches;

  __device__ uint64_t linearize(const legate::Point<DIM>& point) const
  {
    return static_cast<uint64_t>(pitches.flatten_point(point, lo));
  }
};

// Builds pre-computed strides for linearizing points into offsets.
template <int DIM_input>
[[nodiscard]] Buffer<LinearizedRectInfo<DIM_input>> build_linearized_rect_infos(
  const Buffer<int8_t>& source_rects, size_t num_ranks, cudaStream_t stream)
{
  std::vector<legate::Rect<DIM_input>> h_rects(num_ranks);
  std::vector<LinearizedRectInfo<DIM_input>> h_rect_infos(num_ranks);
  const size_t rect_bytes = num_ranks * sizeof(legate::Rect<DIM_input>);
  const size_t info_bytes = num_ranks * sizeof(LinearizedRectInfo<DIM_input>);

  CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
    h_rects.data(), source_rects.ptr(0), rect_bytes, cudaMemcpyDeviceToHost, stream));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (size_t idx = 0; idx < num_ranks; ++idx) {
    h_rect_infos[idx].lo = h_rects[idx].lo;
    h_rect_infos[idx].pitches.flatten(h_rects[idx]);
  }

  auto d_rect_infos =
    create_buffer<LinearizedRectInfo<DIM_input>>(num_ranks, Memory::Kind::GPU_FB_MEM);

  CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
    d_rect_infos.ptr(0), h_rect_infos.data(), info_bytes, cudaMemcpyHostToDevice, stream));

  return d_rect_infos;
}

inline void compute_histogram(const int* samples,
                              size_t num_samples,
                              unsigned long long* histogram,
                              int num_bins,
                              cudaStream_t stream)
{
  size_t temp_bytes = 0;

  cub::DeviceHistogram::HistogramEven(
    nullptr, temp_bytes, samples, histogram, num_bins + 1, 0, num_bins, num_samples, stream);

  Buffer<int8_t> temp_storage;

  if (temp_bytes > 0) {
    temp_storage = create_buffer<int8_t>(temp_bytes, Memory::Kind::GPU_FB_MEM);
  }

  cub::DeviceHistogram::HistogramEven(temp_bytes > 0 ? temp_storage.ptr(0) : nullptr,
                                      temp_bytes,
                                      samples,
                                      histogram,
                                      num_bins + 1,
                                      0,
                                      num_bins,
                                      num_samples,
                                      stream);
}

// Grouped pairwise send/recv all-to-all. Each rank sends `count` elements of
// `dtype` to every peer (data destined for rank j at sendbuf + j*count) and
// receives `count` elements from every peer (data from rank i at recvbuf +
// i*count). We use the pairwise send/recv form rather than ncclAlltoAll so we
// don't take a hard dependency on NCCL >= 2.28 - the user's NCCL version is
// not guaranteed. Caller is responsible for any surrounding Legate
// concurrent_task_barrier coordination.
template <typename T>
void nccl_alltoall(const T* sendbuf,
                   T* recvbuf,
                   size_t count,
                   ncclDataType_t dtype,
                   int num_ranks,
                   ncclComm_t* nccl_comm,
                   cudaStream_t stream)
{
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    CHECK_NCCL(ncclRecv(recvbuf + i * count, count, dtype, i, *nccl_comm, stream));
    CHECK_NCCL(ncclSend(sendbuf + i * count, count, dtype, i, *nccl_comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
}

// ShuffleDescriptor is a helper class that stores the shuffle descriptor for the
// all2all operation. The three count/offset arrays live in Z_COPY_MEM (pinned host
// memory mapped into the GPU address space).
struct ShuffleDescriptor {
  // All length num_ranks; backed by Z_COPY_MEM, owned by the caller.
  // h_send_counts_per_rank[r]    : number of local requests sent to rank r
  // h_send_offsets_per_rank[r]   : prefix sum of h_send_counts_per_rank
  // h_receive_counts_per_rank[r] : number of requests this rank receives from rank r
  const unsigned long long* h_send_counts_per_rank;
  const unsigned long long* h_send_offsets_per_rank;
  const unsigned long long* h_receive_counts_per_rank;
  // Prefix sum of h_receive_counts_per_rank, computed on the host.
  std::vector<unsigned long long> h_receive_offsets_per_rank;
  // Total number of requests this rank will receive from all other ranks.
  size_t total_incoming;

  // Caller must ensure the GPU writes that populate the Z_COPY count buffers
  // (CUB histogram, NCCL recv) have completed on the host before invoking
  // this constructor - typically via cudaStreamSynchronize.
  ShuffleDescriptor(const unsigned long long* z_send_counts_per_rank,
                    const unsigned long long* z_send_offsets_per_rank,
                    const unsigned long long* z_receive_counts_per_rank,
                    int num_ranks)
    : h_send_counts_per_rank(z_send_counts_per_rank),
      h_send_offsets_per_rank(z_send_offsets_per_rank),
      h_receive_counts_per_rank(z_receive_counts_per_rank),
      h_receive_offsets_per_rank(num_ranks),
      total_incoming(0)
  {
    std::exclusive_scan(h_receive_counts_per_rank,
                        h_receive_counts_per_rank + num_ranks,
                        h_receive_offsets_per_rank.begin(),
                        0ULL);
    // After the exclusive scan, h_receive_offsets_per_rank[num_ranks - 1]
    // already equals sum of counts[0..num_ranks-2]; adding the last count gives
    // the total without re-walking the array.
    if (num_ranks > 0) {
      total_incoming = static_cast<size_t>(h_receive_offsets_per_rank[num_ranks - 1] +
                                           h_receive_counts_per_rank[num_ranks - 1]);
    }
  }
};

// ============================================================================
// Step 1 - AllGather partition rects
// ============================================================================

// Used by both gather (partition_rect = local source rect) and scatter
// (partition_rect = local output rect). The returned Z_COPY buffer holds an
// array of `num_ranks` Rect<DIM_partition> values, one per rank, in rank order.
template <int DIM_partition>
Buffer<int8_t> allgather_partition_rects(TaskContext& context,
                                         const legate::Rect<DIM_partition>& partition_rect,
                                         ncclComm_t* nccl_comm,
                                         cudaStream_t stream)
{
  const size_t num_ranks     = context.get_launch_domain().get_volume();
  constexpr size_t rect_size = sizeof(legate::Rect<DIM_partition>);
  // Both buffers are tiny (num_ranks rects total) and each is touched only by a single
  // NCCL operation, so put them in pinned host memory rather than burning FB-resident
  // allocations and a H<->D copy on each.
  auto partition_rects   = create_buffer<int8_t>(num_ranks * rect_size, Memory::Kind::Z_COPY_MEM);
  auto local_rect_device = create_buffer<int8_t>(rect_size, Memory::Kind::Z_COPY_MEM);

  std::memcpy(local_rect_device.ptr(0), &partition_rect, rect_size);
  std::memset(partition_rects.ptr(0), 0, num_ranks * rect_size);

  context.concurrent_task_barrier();
  CHECK_NCCL(ncclAllGather(
    local_rect_device.ptr(0), partition_rects.ptr(0), rect_size, ncclInt8, *nccl_comm, stream));
  context.concurrent_task_barrier();
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  return partition_rects;
}

// ============================================================================
// Step 2 - Classify requests & build ShuffleDescriptor
// ============================================================================

template <typename PointLoader, int DIM_input>
__global__ void assign_target_ranks_kernel(PointLoader loader,
                                           size_t count,
                                           int* target_ranks,
                                           const legate::Rect<DIM_input>* source_rects,
                                           int num_ranks,
                                           int tile_size)
{
  extern __shared__ char smem_raw[];
  auto* smem_source_rects = reinterpret_cast<legate::Rect<DIM_input>*>(smem_raw);

  const size_t idx  = global_tid_1d();
  const bool active = idx < count;
  bool found        = false;

  const legate::Point<DIM_input> point = active ? loader(idx) : legate::Point<DIM_input>{};

  for (int tile_start = 0; tile_start < num_ranks; tile_start += tile_size) {
    const int this_tile = min(tile_size, num_ranks - tile_start);

    for (int i = threadIdx.x; i < this_tile; i += blockDim.x) {
      smem_source_rects[i] = source_rects[tile_start + i];
    }
    __syncthreads();

    if (active && !found) {
      for (int r = 0; r < this_tile; r++) {
        bool inside = true;
        for (int d = 0; d < DIM_input; d++) {
          if (smem_source_rects[r].lo[d] > point[d] || smem_source_rects[r].hi[d] < point[d]) {
            inside = false;
            break;
          }
        }
        if (inside) {
          target_ranks[idx] = tile_start + r;
          found             = true;
          break;
        }
      }
    }
    __syncthreads();
  }
}

template <int DIM_input, int DIM_output>
void classify_index_points(
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t local_index_count,
  int* target_ranks,
  const void* source_rects,
  int num_ranks,
  cudaStream_t stream)
{
  using AccLoader = AccessorPointLoader<DIM_input, DIM_output>;
  using DnsLoader = DensePointLoader<DIM_input>;

  const auto& properties = get_device_properties();
  const size_t grid      = (local_index_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const size_t max_smem = [&]() -> size_t {
    if (properties.sharedMemPerBlock < properties.sharedMemPerBlockOptin) {
      const void* kernel_func = index.is_dense()
                                  ? (const void*)assign_target_ranks_kernel<DnsLoader, DIM_input>
                                  : (const void*)assign_target_ranks_kernel<AccLoader, DIM_input>;
      CUPYNUMERIC_CHECK_CUDA(cudaFuncSetAttribute(kernel_func,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  properties.sharedMemPerBlockOptin));
      return properties.sharedMemPerBlockOptin;
    }
    return properties.sharedMemPerBlock;
  }();

  const size_t tile_size = max_smem / sizeof(legate::Rect<DIM_input>);

  assert(tile_size > 0);

  const size_t smem_size =
    std::min(static_cast<size_t>(num_ranks), static_cast<size_t>(tile_size)) *
    sizeof(legate::Rect<DIM_input>);

  const auto* source_rects_ptr = static_cast<const legate::Rect<DIM_input>*>(source_rects);

  if (index.is_dense()) {
    assign_target_ranks_kernel<DnsLoader, DIM_input>
      <<<grid, THREADS_PER_BLOCK, smem_size, stream>>>(DnsLoader{index.dense_ptr},
                                                       local_index_count,
                                                       target_ranks,
                                                       source_rects_ptr,
                                                       num_ranks,
                                                       tile_size);
  } else {
    assign_target_ranks_kernel<AccLoader, DIM_input>
      <<<grid, THREADS_PER_BLOCK, smem_size, stream>>>(
        AccLoader{index.acc, index.pitches, index.lo},
        local_index_count,
        target_ranks,
        source_rects_ptr,
        num_ranks,
        tile_size);
  }
}

// Group request indices into per-rank buckets.
// `static` so each TU that includes this header gets its own copy (this is a
// non-template __global__ kernel, so we cannot share a single symbol across
// TUs without risking ODR violations). Launches only from within this header.
static __global__ void pack_indices_by_rank_warp(const int* target_ranks,
                                                 size_t count,
                                                 unsigned long long* next_slot,
                                                 uint64_t* packed_positions)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  const int r          = target_ranks[idx];
  const unsigned amask = __activemask();
  const unsigned peers = __match_any_sync(amask, r);
  const int lane       = threadIdx.x & (warpSize - 1);
  const int leader     = __ffs(peers) - 1;
  const int group_size = __popc(peers);

  const unsigned long long base = __shfl_sync(
    peers,
    (lane == leader) ? atomicAdd(&next_slot[r], static_cast<unsigned long long>(group_size)) : 0ULL,
    leader);

  const int intra_offset = __popc(peers & ((1u << lane) - 1));

  packed_positions[base + intra_offset] = static_cast<uint64_t>(idx);
}

// Classify requests & build ShuffleDescriptor
template <int DIM_input, int DIM_output>
[[nodiscard]] ShuffleDescriptor create_shuffle_information(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t local_index_count,
  const void* source_rects,
  uint64_t* request_positions,
  int* target_ranks,
  unsigned long long* send_offsets_per_rank,
  ncclComm_t* nccl_comm,
  cudaStream_t stream)
{
  const int num_ranks = static_cast<int>(context.get_launch_domain().get_volume());

  auto send_counts_per_rank =
    create_buffer<unsigned long long>(num_ranks, Memory::Kind::Z_COPY_MEM);
  auto receive_counts_per_rank =
    create_buffer<unsigned long long>(num_ranks, Memory::Kind::Z_COPY_MEM);
  const auto send_counts_per_rank_ptr    = send_counts_per_rank.ptr(0);
  const auto receive_counts_per_rank_ptr = receive_counts_per_rank.ptr(0);

  std::memset(send_counts_per_rank_ptr, 0, num_ranks * sizeof(unsigned long long));
  std::memset(receive_counts_per_rank_ptr, 0, num_ranks * sizeof(unsigned long long));

  // classify local index points by destination rank, then histogram
  // them into the Z_COPY-resident send_counts_per_rank.
  if (local_index_count > 0) {
    classify_index_points<DIM_input, DIM_output>(
      index, local_index_count, target_ranks, source_rects, num_ranks, stream);

    compute_histogram(target_ranks, local_index_count, send_counts_per_rank_ptr, num_ranks, stream);
  }

  // exchange counts so each rank knows how many requests it receives from every other rank.
  context.concurrent_task_barrier();
  nccl_alltoall(send_counts_per_rank_ptr,
                receive_counts_per_rank_ptr,
                /*count=*/1,
                ncclUint64,
                num_ranks,
                nccl_comm,
                stream);
  context.concurrent_task_barrier();

  // Single sync: the CUB histogram and NCCL recv writes into the Z_COPY count
  // buffers are now visible on the host.
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  // Prefix sum of send_counts to get send_offsets.
  std::exclusive_scan(
    send_counts_per_rank_ptr, send_counts_per_rank_ptr + num_ranks, send_offsets_per_rank, 0ULL);

  // pack request indices into per-rank buckets.
  if (local_index_count > 0) {
    auto next_slot = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);

    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(next_slot.ptr(0),
                                           send_offsets_per_rank,
                                           num_ranks * sizeof(unsigned long long),
                                           cudaMemcpyDefault,
                                           stream));

    const size_t grid = (local_index_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    pack_indices_by_rank_warp<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      target_ranks, local_index_count, next_slot.ptr(0), request_positions);
  }

  return ShuffleDescriptor(
    send_counts_per_rank_ptr, send_offsets_per_rank, receive_counts_per_rank_ptr, num_ranks);
}

// ============================================================================
// Step 3 - Linearize & exchange offsets
// ============================================================================

template <typename PointLoader, int DIM_input>
__global__ void linearize_offsets_kernel(PointLoader loader,
                                         const uint64_t* request_positions,
                                         const unsigned long long* send_offsets_per_rank,
                                         int num_ranks,
                                         const LinearizedRectInfo<DIM_input>* rect_infos,
                                         uint64_t* offsets_out,
                                         size_t count)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  int lo = 0;
  int hi = num_ranks;
  // binary search to find the rank that owns the request
  while (lo < hi) {
    const int mid = (lo + hi) / 2;

    if (send_offsets_per_rank[mid] <= idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const int target_rank = lo - 1;

  assert(target_rank >= 0);

  const legate::Point<DIM_input> point           = loader(request_positions[idx]);
  const LinearizedRectInfo<DIM_input>& rect_info = rect_infos[target_rank];

  offsets_out[idx] = rect_info.linearize(point);
}

template <int DIM_input, int DIM_output>
[[nodiscard]] Buffer<uint64_t> linearize_and_exchange_offsets(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t local_index_count,
  const uint64_t* request_positions,
  const unsigned long long* send_offsets_per_rank,
  const LinearizedRectInfo<DIM_input>* rect_infos,
  const ShuffleDescriptor& plan,
  ncclComm_t* nccl_comm,
  cudaStream_t stream)
{
  const int num_ranks   = static_cast<int>(context.get_launch_domain().get_volume());
  auto send_offsets_buf = create_buffer<uint64_t>(local_index_count, Memory::Kind::GPU_FB_MEM);

  if (local_index_count > 0) {
    const size_t grid = (local_index_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (index.is_dense()) {
      const DensePointLoader<DIM_input> loader{index.dense_ptr};

      linearize_offsets_kernel<decltype(loader), DIM_input>
        <<<grid, THREADS_PER_BLOCK, 0, stream>>>(loader,
                                                 request_positions,
                                                 send_offsets_per_rank,
                                                 num_ranks,
                                                 rect_infos,
                                                 send_offsets_buf.ptr(0),
                                                 local_index_count);
    } else {
      const AccessorPointLoader<DIM_input, DIM_output> loader{index.acc, index.pitches, index.lo};

      linearize_offsets_kernel<decltype(loader), DIM_input>
        <<<grid, THREADS_PER_BLOCK, 0, stream>>>(loader,
                                                 request_positions,
                                                 send_offsets_per_rank,
                                                 num_ranks,
                                                 rect_infos,
                                                 send_offsets_buf.ptr(0),
                                                 local_index_count);
    }
  }

  auto recv_offsets_buf = create_buffer<uint64_t>(plan.total_incoming, Memory::Kind::GPU_FB_MEM);

  // do the NCCL exchanges of the offsets, after this, each rank knows which
  // elements other ranks need from its local data
  context.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    if (plan.h_send_counts_per_rank[i] > 0) {
      CHECK_NCCL(ncclSend(send_offsets_buf.ptr(0) + plan.h_send_offsets_per_rank[i],
                          plan.h_send_counts_per_rank[i],
                          ncclUint64,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (plan.h_receive_counts_per_rank[i] > 0) {
      CHECK_NCCL(ncclRecv(recv_offsets_buf.ptr(0) + plan.h_receive_offsets_per_rank[i],
                          plan.h_receive_counts_per_rank[i],
                          ncclUint64,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  context.concurrent_task_barrier();

  return recv_offsets_buf;
}

// ============================================================================
// Step 4 - Pack local values & exchange them
// ============================================================================

// Permuted gather kernel: out[idx] = src[permutation[idx]] where the
// permutation entries are flat offsets into the source store. Reused by both
// gather (owner side, permutation = recv_flat_offsets) and scatter (requester
// side, permutation = request_positions).
template <typename DataType, int DIM>
__global__ void gather_data_by_offset_kernel(AccessorRO<DataType, DIM> src_acc,
                                             Pitches<DIM - 1> src_pitches,
                                             legate::Point<DIM> src_lo,
                                             const uint64_t* permutation,
                                             size_t count,
                                             DataType* out)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  const auto p = src_pitches.unflatten(permutation[idx], src_lo);

  out[idx] = src_acc[p];
}

// `static` for the same ODR reason as pack_indices_by_rank_warp.
static __global__ void gather_data_by_offset_kernel_dense(
  const char* src_ptr, const uint64_t* permutation, size_t count, size_t elem_size, char* out)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  memcpy(out + idx * elem_size, src_ptr + permutation[idx] * elem_size, elem_size);
}

// Pack `count` values from `source` (a local read-only store) into a packed
// staging buffer using `permutation` as flat offsets into `source`. The
// staging buffer lives in caller-provided GPU_FB memory.
template <Type::Code CODE, int DIM>
void pack_values_into_buffer(const StoreView<AccessMode::Read, type_of<CODE>, DIM>& source,
                             const uint64_t* permutation,
                             size_t count,
                             size_t elem_size,
                             int8_t* out_buf,
                             cudaStream_t stream)
{
  using VAL = type_of<CODE>;

  if (count == 0) {
    return;
  }

  const size_t grid = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (source.is_dense()) {
    gather_data_by_offset_kernel_dense<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const char*>(source.dense_ptr),
      permutation,
      count,
      elem_size,
      reinterpret_cast<char*>(out_buf));
  } else {
    gather_data_by_offset_kernel<VAL, DIM><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      source.acc, source.pitches, source.lo, permutation, count, reinterpret_cast<VAL*>(out_buf));
  }
}

// Step 4 value exchange - gather direction (owner -> requester).
//
// Each owner has previously gathered values from its local source at the
// flat offsets requested by every other rank, contiguously in `sendbuf` in
// peer order driven by receive_*. Every rank sends those slices back to the
// peers that asked for them and receives, in send_* order, the values it
// originally requested from each peer.
//
//   sendbuf size = plan.total_incoming               (owner side)
//   recvbuf size = sum(plan.h_send_counts_per_rank)  (requester side)
inline void exchange_values_owner_to_requester(TaskContext& context,
                                               const int8_t* sendbuf,
                                               int8_t* recvbuf,
                                               size_t elem_size,
                                               const ShuffleDescriptor& plan,
                                               int num_ranks,
                                               ncclComm_t* nccl_comm,
                                               cudaStream_t stream)
{
  context.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    if (plan.h_receive_counts_per_rank[i] > 0) {
      CHECK_NCCL(ncclSend(sendbuf + plan.h_receive_offsets_per_rank[i] * elem_size,
                          plan.h_receive_counts_per_rank[i] * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (plan.h_send_counts_per_rank[i] > 0) {
      CHECK_NCCL(ncclRecv(recvbuf + plan.h_send_offsets_per_rank[i] * elem_size,
                          plan.h_send_counts_per_rank[i] * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  context.concurrent_task_barrier();
}

// Step 4 value exchange - scatter direction (requester -> owner).
//
// Each requester has packed its local source values into `sendbuf` in peer
// order driven by send_*. Every rank ships those slices to the peers that
// own the targets and receives, in receive_* order, the values that other
// ranks intend to write into its local output.
//
//   sendbuf size = sum(plan.h_send_counts_per_rank)  (requester side)
//   recvbuf size = plan.total_incoming               (owner side)
inline void exchange_values_requester_to_owner(TaskContext& context,
                                               const int8_t* sendbuf,
                                               int8_t* recvbuf,
                                               size_t elem_size,
                                               const ShuffleDescriptor& plan,
                                               int num_ranks,
                                               ncclComm_t* nccl_comm,
                                               cudaStream_t stream)
{
  context.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    if (plan.h_send_counts_per_rank[i] > 0) {
      CHECK_NCCL(ncclSend(sendbuf + plan.h_send_offsets_per_rank[i] * elem_size,
                          plan.h_send_counts_per_rank[i] * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (plan.h_receive_counts_per_rank[i] > 0) {
      CHECK_NCCL(ncclRecv(recvbuf + plan.h_receive_offsets_per_rank[i] * elem_size,
                          plan.h_receive_counts_per_rank[i] * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  context.concurrent_task_barrier();
}

// ============================================================================
// Step 5 - Unpack received values into output
// ============================================================================

template <typename DataType, int DIM_output>
__global__ void unpack_recv_data_kernel(AccessorRW<DataType, DIM_output> output_acc,
                                        Pitches<DIM_output - 1> output_pitches,
                                        legate::Point<DIM_output> output_lo,
                                        const uint64_t* request_indices,
                                        size_t count,
                                        const DataType* recv_data)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  const auto p = output_pitches.unflatten(request_indices[idx], output_lo);

  output_acc[p] = recv_data[idx];
}

// `static` for the same ODR reason as pack_indices_by_rank_warp.
static __global__ void unpack_recv_data_kernel_dense(char* __restrict__ output_ptr,
                                                     const uint64_t* request_indices,
                                                     size_t count,
                                                     size_t elem_size,
                                                     const char* recv_data)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= count) {
    return;
  }

  memcpy(output_ptr + request_indices[idx] * elem_size, recv_data + idx * elem_size, elem_size);
}

template <Type::Code CODE, int DIM_output>
void unpack_recv_into_output(const StoreView<AccessMode::Write, type_of<CODE>, DIM_output>& output,
                             const uint64_t* request_positions,
                             size_t local_index_count,
                             const void* recv_data,
                             cudaStream_t stream)
{
  using VAL = type_of<CODE>;

  if (local_index_count > 0) {
    const size_t grid = (local_index_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (output.is_dense()) {
      unpack_recv_data_kernel_dense<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<char*>(output.dense_ptr),
        request_positions,
        local_index_count,
        sizeof(VAL),
        static_cast<const char*>(recv_data));
    } else {
      unpack_recv_data_kernel<VAL, DIM_output>
        <<<grid, THREADS_PER_BLOCK, 0, stream>>>(output.acc,
                                                 output.pitches,
                                                 output.lo,
                                                 request_positions,
                                                 local_index_count,
                                                 static_cast<const VAL*>(recv_data));
    }
  }
}

}  // namespace detail
}  // namespace cupynumeric
