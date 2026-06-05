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
// Step 3+4+5: Linearize / exchange / pack / exchange / unpack
//             (`local_gather_and_exchange`)
//   To bound peak FB memory, all per-element scratch (offset buffers AND
//   data staging buffers) lives in round-local FB allocations sized
//   `num_ranks * max_elems_per_peer * sizeof(...)`. The exchange runs in K rounds,
//   where K comes from a single ncclAllReduce(max, uint64) over per-pair
//   element counts. Per-round per-peer chunk sizes are derived locally
//   and are symmetric on both sides of every pair (because by construction
//   plan_A.h_recv_counts[B] == plan_B.h_send_counts[A]), so no extra
//   coordination traffic is needed.
//
//   Each round runs:
//     1) linearize_offsets_kernel — convert this round's chunk
//        of Point<DIM> requests (per peer) into uint64 flat offsets,
//        written into round_send_offsets at strided peer slots.
//     2) ncclGroupStart/End — pairwise send/recv of the offsets,
//        requester → owner. Result lands in round_recv_offsets.
//     3) pack_data_by_offset_kernel — read source data into
//        send_staging.
//     4) ncclGroupStart/End — pairwise send/recv of the data, in the
//        direction dictated by gather (owner → requester) or scatter
//        (requester → owner).
//     5) unpack_recv_data_kernel — write recv_staging into the
//        local output.
//
//   gather:  Owner reads its local source at the offsets it just received
//            in step 2, ncclSend's the values back to the requester, which
//            writes them into output at request_positions.
//   scatter: Requester reads its local source at request_positions,
//            ncclSend's the values to the owner, which writes them into
//            output at the offsets it just received in step 2.

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/utilities/thrust_util.h"

#include "legate/utilities/abort.h"

#include <cuda_runtime.h>

#include <algorithm>
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

  [[nodiscard]] bool is_dense() const { return dense_ptr != nullptr; }
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
  // Scalars populated by the caller after the post-counts NCCL allreduce
  // and the per-buffer staging budget have been resolved. Bundling them
  // here lets `local_gather_and_exchange` accept a single plan object.
  int num_ranks             = 0;
  size_t max_elems_per_peer = 0;
  size_t num_rounds         = 0;

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
      total_incoming(0),
      num_ranks(num_ranks)
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
[[nodiscard]] Buffer<int8_t> allgather_partition_rects(
  TaskContext& context,
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
  const size_t blocks    = (local_index_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

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
      <<<blocks, THREADS_PER_BLOCK, smem_size, stream>>>(DnsLoader{index.dense_ptr},
                                                         local_index_count,
                                                         target_ranks,
                                                         source_rects_ptr,
                                                         num_ranks,
                                                         tile_size);
  } else {
    assign_target_ranks_kernel<AccLoader, DIM_input>
      <<<blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
        AccLoader{index.acc, index.pitches, index.lo},
        local_index_count,
        target_ranks,
        source_rects_ptr,
        num_ranks,
        tile_size);
  }
}

// Group request indices into peer-major layout slots.
//
// `next_slot[r]` starts at 0 and is atomically advanced by each warp-grouped
// run of same-rank threads; the returned `j` is this thread's intra-peer
// counter (0-based). The slot is the closed-form expression
//   slot = send_offsets_per_rank[r] + j
// so peer p's requests land contiguously at
// `[send_offsets_per_rank[p], send_offsets_per_rank[p] + h_send_counts[p])`.
// Round chunks of this list are derived on demand each round (see
// `fill_round_schedule` and `gather_round_request_positions`).
//
// `static` so each TU that includes this header gets its own copy (this is a
// non-template __global__ kernel, so we cannot share a single symbol across
// TUs without risking ODR violations). Launches only from within this header.
static __global__ void pack_indices_by_rank_warp(const int* target_ranks,
                                                 size_t count,
                                                 unsigned long long* next_slot,
                                                 const unsigned long long* send_offsets_per_rank,
                                                 uint64_t* packed_positions)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  // Skip points that matched no source rect (target_ranks left at the -1
  // memset sentinel). `compute_histogram` already drops them, so offsets
  // exclude these slots.
  const int r = target_ranks[idx];
  if (r < 0) {
    return;
  }
  const unsigned amask = __activemask();
  const unsigned peers = __match_any_sync(amask, r);
  const int lane       = threadIdx.x & (warpSize - 1);
  const int leader     = __ffs(peers) - 1;
  const int group_size = __popc(peers);

  const unsigned long long base_j = __shfl_sync(
    peers,
    (lane == leader) ? atomicAdd(&next_slot[r], static_cast<unsigned long long>(group_size)) : 0ULL,
    leader);

  const int intra_offset     = __popc(peers & ((1u << lane) - 1));
  const unsigned long long j = base_j + static_cast<unsigned long long>(intra_offset);

  packed_positions[send_offsets_per_rank[r] + j] = static_cast<uint64_t>(idx);
}

// Classify requests, exchange counts, build ShuffleDescriptor.
//
// Step 2a..2d only: classify each local index point's target rank,
// histogram into send_counts_per_rank, NCCL-exchange into
// receive_counts_per_rank, and host-prefix-sum into send_offsets_per_rank.
// `request_positions` is filled in a separate pass
// (`pack_request_positions`) once `send_offsets_per_rank`
// is known, so the warp pack can write directly into peer-major slots.
template <int DIM_input, int DIM_output>
[[nodiscard]] ShuffleDescriptor create_shuffle_information(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t local_index_count,
  const void* source_rects,
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

  return ShuffleDescriptor(
    send_counts_per_rank_ptr, send_offsets_per_rank, receive_counts_per_rank_ptr, num_ranks);
}

// Step 2e: pack target_ranks into `request_positions` arranged by peer.
//
// Each peer-grouped warp atomically advances `next_slot[r]` (zero-based
// peer-local counter) and writes into the closed-form slot
// `send_offsets_per_rank[r] + j`. Result: peer p's full request list lives
// contiguously at
// `request_positions[send_offsets_per_rank[p]
//                  : send_offsets_per_rank[p] + h_send_counts[p])`.
inline void pack_request_positions(const int* target_ranks,
                                   size_t local_index_count,
                                   const unsigned long long* send_offsets_per_rank,
                                   int num_ranks,
                                   uint64_t* request_positions,
                                   cudaStream_t stream)
{
  if (local_index_count == 0) {
    return;
  }

  auto next_slot = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);
  CUPYNUMERIC_CHECK_CUDA(
    cudaMemsetAsync(next_slot.ptr(0), 0, num_ranks * sizeof(unsigned long long), stream));

  const size_t blocks = (local_index_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  pack_indices_by_rank_warp<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    target_ranks, local_index_count, next_slot.ptr(0), send_offsets_per_rank, request_positions);
}

// ============================================================================
// Step 3 - Linearize & exchange offsets (chunked, fused into the K-round loop)
// ============================================================================

// Returns the peer index p such that
//   round_send_prefix[p] <= idx < round_send_prefix[p+1].
// Preconditions: `idx < total_round_send` (caller has already early-returned
// otherwise) and `round_send_prefix[0] == 0` (it's a prefix scan that starts
// at zero). Together these guarantee the first comparison advances `lo` to at
// least 1, so the returned peer is always >= 0.
__device__ inline int find_peer_for_offset(const unsigned long long* round_send_prefix,
                                           int num_ranks,
                                           size_t idx)
{
  int lo = 0;
  int hi = num_ranks;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (round_send_prefix[mid] <= idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo - 1;
}

// Materialize the round-contiguous slice of `request_positions` from the
// persistent peer-major buffer.
//
// `request_positions_peer_major[send_offsets_per_rank[p] + j]` holds peer
// p's j-th request (j in [0, h_send_counts[p])). Round r's chunk for peer p
// lives at `[peer_src_offsets[p], peer_src_offsets[p] + chunk_send_counts[p])`
// (with `peer_src_offsets[p] = send_offsets_per_rank[p] + r*max_elems_per_peer`).
//
// Each thread `idx` finds its peer via a binary search over
// `round_send_prefix` (same layout as the round_offset_buf consumers) and
// copies one slot.
static __global__ void gather_round_request_positions_kernel(
  const uint64_t* request_positions_peer_major,
  const unsigned long long* peer_src_offsets,
  const unsigned long long* round_send_prefix,
  int num_ranks,
  uint64_t* round_request_positions,
  size_t total_round_send)
{
  const size_t idx = global_tid_1d();

  if (idx >= total_round_send) {
    return;
  }

  const int peer = find_peer_for_offset(round_send_prefix, num_ranks, idx);

  const unsigned long long within_peer = idx - round_send_prefix[peer];
  round_request_positions[idx] = request_positions_peer_major[peer_src_offsets[peer] + within_peer];
}

inline void gather_round_request_positions(const uint64_t* request_positions_peer_major,
                                           const unsigned long long* peer_src_offsets,
                                           const unsigned long long* round_send_prefix,
                                           int num_ranks,
                                           uint64_t* round_request_positions,
                                           size_t total_round_send,
                                           cudaStream_t stream)
{
  if (total_round_send == 0) {
    return;
  }
  const size_t blocks = (total_round_send + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  gather_round_request_positions_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    request_positions_peer_major,
    peer_src_offsets,
    round_send_prefix,
    num_ranks,
    round_request_positions,
    total_round_send);
}

// Round-local linearize: `request_positions` here is the per-round slice
// materialized by `gather_round_request_positions` (peer i lives at
// `[round_send_prefix[i], round_send_prefix[i+1])` within this slice).
// Thread `idx` reads `request_positions[idx]` directly; the binary search
// over `round_send_prefix` recovers the peer only to pick `rect_infos[peer]`
// for the Point<DIM> -> flat-offset conversion.
template <typename PointLoader, int DIM_input>
__global__ void linearize_offsets_kernel(PointLoader loader,
                                         const uint64_t* request_positions,
                                         const unsigned long long* round_send_prefix,
                                         int num_ranks,
                                         const LinearizedRectInfo<DIM_input>* rect_infos,
                                         uint64_t* round_send_offsets,
                                         size_t total_round_send)
{
  const size_t idx = global_tid_1d();

  if (idx >= total_round_send) {
    return;
  }

  const int peer = find_peer_for_offset(round_send_prefix, num_ranks, idx);

  const legate::Point<DIM_input> point           = loader(request_positions[idx]);
  const LinearizedRectInfo<DIM_input>& rect_info = rect_infos[peer];

  round_send_offsets[idx] = rect_info.linearize(point);
}

template <int DIM_input, int DIM_index_outer>
inline void linearize_and_exchange_offsets(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_index_outer>& index,
  const uint64_t* round_request_positions,
  const unsigned long long* round_send_counts,
  const unsigned long long* round_recv_counts,
  const unsigned long long* round_send_prefix,
  const unsigned long long* round_recv_prefix,
  const LinearizedRectInfo<DIM_input>* rect_infos,
  uint64_t* round_send_offsets,
  uint64_t* round_recv_offsets,
  unsigned long long total_round_send,
  int num_ranks,
  ncclComm_t* nccl_comm,
  cudaStream_t stream)
{
  if (total_round_send > 0) {
    const size_t blocks =
      (static_cast<size_t>(total_round_send) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (index.is_dense()) {
      const DensePointLoader<DIM_input> loader{index.dense_ptr};
      linearize_offsets_kernel<decltype(loader), DIM_input>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(loader,
                                                   round_request_positions,
                                                   round_send_prefix,
                                                   num_ranks,
                                                   rect_infos,
                                                   round_send_offsets,
                                                   static_cast<size_t>(total_round_send));
    } else {
      const AccessorPointLoader<DIM_input, DIM_index_outer> loader{
        index.acc, index.pitches, index.lo};
      linearize_offsets_kernel<decltype(loader), DIM_input>
        <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(loader,
                                                   round_request_positions,
                                                   round_send_prefix,
                                                   num_ranks,
                                                   rect_infos,
                                                   round_send_offsets,
                                                   static_cast<size_t>(total_round_send));
    }
  }

  context.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    const size_t send_count = round_send_counts[i];
    const size_t recv_count = round_recv_counts[i];
    if (send_count > 0) {
      CHECK_NCCL(ncclSend(
        round_send_offsets + round_send_prefix[i], send_count, ncclUint64, i, *nccl_comm, stream));
    }
    if (recv_count > 0) {
      CHECK_NCCL(ncclRecv(
        round_recv_offsets + round_recv_prefix[i], recv_count, ncclUint64, i, *nccl_comm, stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  context.concurrent_task_barrier();
}

// ============================================================================
// Step 3+4+5 - Chunked linearize, exchange, pack, exchange, unpack
//
// To bound peak FB memory the offset-exchange (Step 3) AND the data-
// exchange (Step 4+5) both run in K chunked rounds and reuse round-local
// FB buffers. Each round issues two `ncclGroupStart/End` collectives
// (one for offsets requester→owner, one for data in the direction
// dictated by gather vs scatter) and folds the unpack into the same loop
// so all staging slots can be reused before the next round overwrites
// them.
//
// One implementation drives both the gather direction (data: owner ->
// requester) and the scatter direction (data: requester -> owner). The
// the pack / unpack permutations and per-peer count arrays are
// chosen accordingly. The Step 3 offset exchange is always
// requester→owner, so its per-peer counts are always
// `chunk_send_counts` / `chunk_recv_counts` (which are populated from
// step 2's send_counts / receive_counts for both directions).
//
// All ranks compute the same K from a single ncclAllReduce(max, uint64) of
// `max(local send_count_max, local recv_count_max)`. Per-round chunk sizes
// are derived locally via `min(remaining, max_elems_per_peer)`, which is symmetric
// on both sides of every pair (because by construction
// `plan_A.h_recv_counts[B] == plan_B.h_send_counts[A]`). No extra
// coordination traffic is needed.
// ============================================================================

// Permuted gather: out[idx] = src[permutation[idx]] where the permutation
// entries are flat offsets into the source store. Reused by both gather
// (owner side, permutation = round_recv_offsets) and scatter (requester
// side, permutation = round_request_positions). Identical in shape to
// main's `gather_data_by_offset_kernel` because the chunked algorithm now
// hands each round a permutation array sized 1:1 with the staging buffer.
template <typename DataType, int DIM>
__global__ void pack_data_by_offset_kernel(AccessorRO<DataType, DIM> src_acc,
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

// `static` to avoid ODR collisions across TUs that include this header.
static __global__ void pack_data_by_offset_kernel_dense(
  const char* src_ptr, const uint64_t* permutation, size_t count, size_t elem_size, char* out)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  copy_elements(out + idx * elem_size, src_ptr + permutation[idx] * elem_size, elem_size);
}

template <Type::Code CODE, int DIM>
inline void pack_values_into_buffer(const StoreView<AccessMode::Read, type_of<CODE>, DIM>& source,
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

  const size_t blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (source.is_dense()) {
    pack_data_by_offset_kernel_dense<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const char*>(source.dense_ptr),
      permutation,
      count,
      elem_size,
      reinterpret_cast<char*>(out_buf));
  } else {
    pack_data_by_offset_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      source.acc, source.pitches, source.lo, permutation, count, reinterpret_cast<VAL*>(out_buf));
  }
}

// Send/recv staging buffers are laid out contiguously per peer for the
// round, with peer i occupying `[round_*_prefix[i], round_*_prefix[i+1])`.
// Gather flips the role of `send`/`recv` counts (owner→requester data
// follows the offsets that were sent requester→owner in step 3).
inline void exchange_values_owner_to_requester(TaskContext& context,
                                               const int8_t* send_staging,
                                               int8_t* recv_staging,
                                               const unsigned long long* round_send_counts,
                                               const unsigned long long* round_recv_counts,
                                               const unsigned long long* round_send_prefix,
                                               const unsigned long long* round_recv_prefix,
                                               size_t elem_size,
                                               int num_ranks,
                                               ncclComm_t* nccl_comm,
                                               cudaStream_t stream)
{
  context.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    const size_t send_count = round_recv_counts[i];
    const size_t recv_count = round_send_counts[i];
    if (send_count > 0) {
      CHECK_NCCL(ncclSend(send_staging + round_recv_prefix[i] * elem_size,
                          send_count * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (recv_count > 0) {
      CHECK_NCCL(ncclRecv(recv_staging + round_send_prefix[i] * elem_size,
                          recv_count * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  context.concurrent_task_barrier();
}

inline void exchange_values_requester_to_owner(TaskContext& context,
                                               const int8_t* send_staging,
                                               int8_t* recv_staging,
                                               const unsigned long long* round_send_counts,
                                               const unsigned long long* round_recv_counts,
                                               const unsigned long long* round_send_prefix,
                                               const unsigned long long* round_recv_prefix,
                                               size_t elem_size,
                                               int num_ranks,
                                               ncclComm_t* nccl_comm,
                                               cudaStream_t stream)
{
  context.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    const size_t send_count = round_send_counts[i];
    const size_t recv_count = round_recv_counts[i];
    if (send_count > 0) {
      CHECK_NCCL(ncclSend(send_staging + round_send_prefix[i] * elem_size,
                          send_count * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (recv_count > 0) {
      CHECK_NCCL(ncclRecv(recv_staging + round_recv_prefix[i] * elem_size,
                          recv_count * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  context.concurrent_task_barrier();
}

// Indirect write: output[request_indices[idx]] = recv_data[idx]. Reused by
// both gather (requester side, request_indices = round_request_positions)
// and scatter (owner side, request_indices = round_recv_offsets). Identical
// in shape to main's `unpack_recv_data_kernel` thanks to the round-layout
// permutations being 1:1 with the receive buffer.
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

// `static` to avoid ODR collisions across TUs that include this header.
static __global__ void unpack_recv_data_kernel_dense(char* __restrict__ output_ptr,
                                                     const uint64_t* request_indices,
                                                     size_t count,
                                                     size_t elem_size,
                                                     const char* recv_data)
{
  const size_t idx = global_tid_1d();

  if (idx >= count) {
    return;
  }

  copy_elements(
    output_ptr + request_indices[idx] * elem_size, recv_data + idx * elem_size, elem_size);
}

template <Type::Code CODE, int DIM_output>
inline void unpack_recv_into_output(
  const StoreView<AccessMode::Write, type_of<CODE>, DIM_output>& output,
  const uint64_t* request_indices,
  size_t count,
  size_t elem_size,
  const int8_t* recv_data,
  cudaStream_t stream)
{
  using VAL = type_of<CODE>;

  if (count == 0) {
    return;
  }

  const size_t blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (output.is_dense()) {
    unpack_recv_data_kernel_dense<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<char*>(output.dense_ptr),
      request_indices,
      count,
      elem_size,
      reinterpret_cast<const char*>(recv_data));
  } else {
    unpack_recv_data_kernel<VAL, DIM_output>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(output.acc,
                                                 output.pitches,
                                                 output.lo,
                                                 request_indices,
                                                 count,
                                                 reinterpret_cast<const VAL*>(recv_data));
  }
}

// Per-round schedule. N-sized; rebuilt in place each round by
// `fill_round_schedule` so peak ZCMEM is O(N) regardless of K.
//
// `within_round_send_prefix` and `peer_src_offsets` are read by GPU
// kernels and live in ZCMEM; the other N-sized tables are host-only
// (NCCL count/offset args) and live in plain std::vector.
//
// Because the ZCMEM tables are rewritten in place each round and host
// stores into ZCMEM are not ordered against in-flight GPU work, the
// `fill_round_schedule` call that prepares round `R+1` must
// `cudaStreamSynchronize` first to ensure the previous round's kernels
// have finished consuming the tables before they are overwritten.
//
//   chunk_send_counts[p]        - NCCL send count for the current round,
//                                 peer p (offset exchange and, after the
//                                 direction swap, data exchange).
//   chunk_recv_counts[p]        - NCCL recv count for the current round.
//   within_round_send_prefix[p] - exclusive prefix sum of chunk_send_counts
//                                 over peers. Gives peer p's starting slot
//                                 in the round-local `round_send_offsets` /
//                                 `send_staging` buffers, used both by the
//                                 kernels' binary search (linearize) and by
//                                 NCCL slice arithmetic.
//   within_round_recv_prefix[p] - same for recv side.
//   peer_src_offsets[p]         - absolute offset into the peer-major
//                                 `request_positions` for the current
//                                 round, peer p. Used by the per-round
//                                 `gather_round_request_positions_kernel`
//                                 to materialize the round-contiguous
//                                 permutation; closed form
//                                 `send_offsets_per_rank[p] + round * max_elems_per_peer`.
//
// Per-round scalars (updated by `fill_round_schedule`):
//   total_round_send - sum of chunk_send_counts.
//   total_round_recv - sum of chunk_recv_counts.
struct RoundSchedule {
  Buffer<unsigned long long> within_round_send_prefix;
  Buffer<unsigned long long> peer_src_offsets;

  std::vector<unsigned long long> chunk_send_counts;
  std::vector<unsigned long long> chunk_recv_counts;
  std::vector<unsigned long long> within_round_recv_prefix;

  unsigned long long total_round_send = 0;
  unsigned long long total_round_recv = 0;

  size_t num_rounds         = 0;
  size_t num_ranks          = 0;
  size_t max_elems_per_peer = 0;
};

// Allocates the N-sized RoundSchedule buffers once per shuffle. Contents
// are filled in by `fill_round_schedule` before each round runs.
[[nodiscard]] inline RoundSchedule build_round_schedule(int num_ranks,
                                                        size_t max_elems_per_peer,
                                                        size_t num_rounds)
{
  RoundSchedule schedule;
  schedule.num_rounds         = num_rounds;
  schedule.num_ranks          = static_cast<size_t>(num_ranks);
  schedule.max_elems_per_peer = max_elems_per_peer;

  const size_t n = static_cast<size_t>(num_ranks);
  schedule.within_round_send_prefix =
    create_buffer<unsigned long long>(n, Memory::Kind::Z_COPY_MEM);
  schedule.peer_src_offsets = create_buffer<unsigned long long>(n, Memory::Kind::Z_COPY_MEM);
  schedule.chunk_send_counts.assign(n, 0);
  schedule.chunk_recv_counts.assign(n, 0);
  schedule.within_round_recv_prefix.assign(n, 0);

  return schedule;
}

// Populate `schedule` in place for round `round`. `h_send_counts` /
// `h_recv_counts` and `send_offsets_per_rank` come from
// `create_shuffle_information` and are reused across rounds. Pure host work,
// O(N) per call.
inline void fill_round_schedule(RoundSchedule& schedule,
                                const unsigned long long* h_send_counts,
                                const unsigned long long* h_recv_counts,
                                const unsigned long long* send_offsets_per_rank,
                                size_t round,
                                cudaStream_t stream)
{
  // Drain the prior round's GPU work before overwriting the reused ZCMEM
  // tables (`within_round_send_prefix`, `peer_src_offsets`). Host stores to
  // ZCMEM are not stream-ordered, so without this sync the next round's
  // host writes could race with the previous round's kernel reads.
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  const int num_ranks                = static_cast<int>(schedule.num_ranks);
  const size_t max_elems_per_peer    = schedule.max_elems_per_peer;
  const unsigned long long round_off = round * max_elems_per_peer;

  unsigned long long* send_counts = schedule.chunk_send_counts.data();
  unsigned long long* recv_counts = schedule.chunk_recv_counts.data();
  unsigned long long* send_prefix = schedule.within_round_send_prefix.ptr(0);
  unsigned long long* recv_prefix = schedule.within_round_recv_prefix.data();
  unsigned long long* src_offsets = schedule.peer_src_offsets.ptr(0);

  unsigned long long send_running = 0;
  unsigned long long recv_running = 0;
  for (int peer = 0; peer < num_ranks; ++peer) {
    const unsigned long long send_chunk_count =
      h_send_counts[peer] > round_off
        ? std::min<unsigned long long>(h_send_counts[peer] - round_off, max_elems_per_peer)
        : 0ULL;
    const unsigned long long recv_chunk_count =
      h_recv_counts[peer] > round_off
        ? std::min<unsigned long long>(h_recv_counts[peer] - round_off, max_elems_per_peer)
        : 0ULL;

    send_counts[peer] = send_chunk_count;
    recv_counts[peer] = recv_chunk_count;
    send_prefix[peer] = send_running;
    recv_prefix[peer] = recv_running;
    src_offsets[peer] = send_offsets_per_rank[peer] + round_off;

    send_running += send_chunk_count;
    recv_running += recv_chunk_count;
  }
  schedule.total_round_send = send_running;
  schedule.total_round_recv = recv_running;
}

// Globally-consistent K via one ncclAllReduce(max, uint64) of this rank's
// `max(send_count_max, recv_count_max)` across all peers. Required so every
// rank issues the same number of grouped send/recv collectives.
[[nodiscard]] inline unsigned long long allreduce_global_max_pair_count(
  TaskContext& context,
  const unsigned long long* h_send_counts,
  const unsigned long long* h_recv_counts,
  int num_ranks,
  ncclComm_t* nccl_comm,
  cudaStream_t stream)
{
  unsigned long long local_max = 0;
  for (int i = 0; i < num_ranks; ++i) {
    local_max = std::max(local_max, h_send_counts[i]);
    local_max = std::max(local_max, h_recv_counts[i]);
  }

  // Use separate send/recv slots in one tiny pinned-host buffer so the
  // collective is not in-place; the host can then read the result after the
  // mandatory stream sync.
  auto buf      = create_buffer<unsigned long long>(2, Memory::Kind::Z_COPY_MEM);
  buf.ptr(0)[0] = local_max;
  buf.ptr(0)[1] = 0;

  context.concurrent_task_barrier();
  CHECK_NCCL(ncclAllReduce(buf.ptr(0), buf.ptr(0) + 1, 1, ncclUint64, ncclMax, *nccl_comm, stream));
  context.concurrent_task_barrier();
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  return buf.ptr(0)[1];
}

// Step 3+4+5 orchestration. The `RoundSchedule` is rebuilt in place each
// round (O(N) host + a small GPU gather) and then each iteration is:
//   1. linearize_offsets_kernel - fill round_send_offsets (round-local FB,
//      contiguous per-peer layout via round_send_prefix) from this round's
//      slice of `request_positions` + rect_infos.
//   2. ncclGroupStart/End: pairwise send/recv of offsets, requester->owner.
//      Each peer i's slice lives at `+ round_send_prefix[i]` (send) and
//      `+ round_recv_prefix[i]` (recv).
//   3. pack kernel - per-round permutation:
//        gather:  permutation = round_recv_offsets,         count = total_round_recv
//        scatter: permutation = round_request_positions,    count = total_round_send
//   4. ncclGroupStart/End: pairwise send/recv of data. Direction follows
//      gather (owner->requester) or scatter (requester->owner) by swapping
//      which count/prefix pair is the "send" vs "recv" side.
//   5. unpack kernel - per-round permutation:
//        gather:  permutation = round_request_positions,    count = total_round_send
//        scatter: permutation = round_recv_offsets,         count = total_round_recv
//
// `request_positions` is in peer-major layout (set by
// `pack_request_positions`): peer p's full request list lives at
// `[send_offsets_per_rank[p], send_offsets_per_rank[p] + h_send_counts[p])`.
// Each round we materialize the round-contiguous
// `round_request_positions` slice via `gather_round_request_positions` so
// the round-loop kernels still see a flat per-round permutation.
//
// All FB scratch buffers (round_request_positions, round_send_offsets,
// round_recv_offsets, send_staging, recv_staging) are sized
// `num_ranks * max_elems_per_peer * <bytes>` and reused every round.
template <Type::Code CODE,
          int DIM_input,
          int DIM_index_outer,
          int PACK_DIM,
          int UNPACK_DIM,
          bool IS_GATHER>
void local_gather_and_exchange(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_index_outer>& index,
  const uint64_t* request_positions,
  const ShuffleDescriptor& plan,
  const LinearizedRectInfo<DIM_input>* rect_infos,
  const StoreView<AccessMode::Read, type_of<CODE>, PACK_DIM>& pack_source,
  const StoreView<AccessMode::Write, type_of<CODE>, UNPACK_DIM>& unpack_dest,
  size_t elem_size,
  ncclComm_t* nccl_comm,
  cudaStream_t stream)
{
  using VAL = type_of<CODE>;

  const int num_ranks             = plan.num_ranks;
  const size_t max_elems_per_peer = plan.max_elems_per_peer;
  const size_t num_rounds         = plan.num_rounds;

  if (num_rounds == 0) {
    return;
  }

  RoundSchedule schedule = build_round_schedule(num_ranks, max_elems_per_peer, num_rounds);

  // Round-local FB scratch buffers, all sized num_ranks * max_elems_per_peer * <elt>
  // and reused every round. The contents are laid out contiguously per
  // peer for each round (peer i at [round_*_prefix[i], round_*_prefix[i+1])),
  // so the worst case (all peers max out their per-peer budget) is the
  // same `num_ranks * max_elems_per_peer` total.
  const size_t total_slots         = static_cast<size_t>(num_ranks) * max_elems_per_peer;
  auto round_request_positions_buf = create_buffer<uint64_t>(total_slots, Memory::Kind::GPU_FB_MEM);
  auto round_send_offsets          = create_buffer<uint64_t>(total_slots, Memory::Kind::GPU_FB_MEM);
  auto round_recv_offsets          = create_buffer<uint64_t>(total_slots, Memory::Kind::GPU_FB_MEM);
  auto send_staging = create_buffer<int8_t>(total_slots * elem_size, Memory::Kind::GPU_FB_MEM);
  auto recv_staging = create_buffer<int8_t>(total_slots * elem_size, Memory::Kind::GPU_FB_MEM);

  for (size_t round = 0; round < schedule.num_rounds; ++round) {
    fill_round_schedule(schedule,
                        plan.h_send_counts_per_rank,
                        plan.h_receive_counts_per_rank,
                        plan.h_send_offsets_per_rank,
                        round,
                        stream);

    const auto* round_send_counts = schedule.chunk_send_counts.data();
    const auto* round_recv_counts = schedule.chunk_recv_counts.data();
    const auto* round_send_prefix = schedule.within_round_send_prefix.ptr(0);
    const auto* round_recv_prefix = schedule.within_round_recv_prefix.data();
    const auto total_round_send   = schedule.total_round_send;
    const auto total_round_recv   = schedule.total_round_recv;

    gather_round_request_positions(request_positions,
                                   schedule.peer_src_offsets.ptr(0),
                                   round_send_prefix,
                                   num_ranks,
                                   round_request_positions_buf.ptr(0),
                                   static_cast<size_t>(total_round_send),
                                   stream);
    const uint64_t* round_request_positions = round_request_positions_buf.ptr(0);

    linearize_and_exchange_offsets<DIM_input, DIM_index_outer>(context,
                                                               index,
                                                               round_request_positions,
                                                               round_send_counts,
                                                               round_recv_counts,
                                                               round_send_prefix,
                                                               round_recv_prefix,
                                                               rect_infos,
                                                               round_send_offsets.ptr(0),
                                                               round_recv_offsets.ptr(0),
                                                               total_round_send,
                                                               num_ranks,
                                                               nccl_comm,
                                                               stream);

    // Pack permutation is 1:1 with the staging buffer thanks to the
    // round-contiguous request_positions and round-local round_recv_offsets.
    if constexpr (IS_GATHER) {
      pack_values_into_buffer<CODE, PACK_DIM>(pack_source,
                                              round_recv_offsets.ptr(0),
                                              static_cast<size_t>(total_round_recv),
                                              elem_size,
                                              send_staging.ptr(0),
                                              stream);
    } else {
      pack_values_into_buffer<CODE, PACK_DIM>(pack_source,
                                              round_request_positions,
                                              static_cast<size_t>(total_round_send),
                                              elem_size,
                                              send_staging.ptr(0),
                                              stream);
    }

    // Data exchange: gather flips send/recv counts vs scatter.
    //   gather:  send = chunk_recv_counts (owner ships back what it
    //                                     received offsets for in step 3),
    //            recv = chunk_send_counts (requester gets back what it
    //                                     asked for).
    //   scatter: send = chunk_send_counts (requester ships data to owner),
    //            recv = chunk_recv_counts (owner ingests data).
    if constexpr (IS_GATHER) {
      exchange_values_owner_to_requester(context,
                                         send_staging.ptr(0),
                                         recv_staging.ptr(0),
                                         round_send_counts,
                                         round_recv_counts,
                                         round_send_prefix,
                                         round_recv_prefix,
                                         elem_size,
                                         num_ranks,
                                         nccl_comm,
                                         stream);
    } else {
      exchange_values_requester_to_owner(context,
                                         send_staging.ptr(0),
                                         recv_staging.ptr(0),
                                         round_send_counts,
                                         round_recv_counts,
                                         round_send_prefix,
                                         round_recv_prefix,
                                         elem_size,
                                         num_ranks,
                                         nccl_comm,
                                         stream);
    }

    if constexpr (IS_GATHER) {
      unpack_recv_into_output<CODE, UNPACK_DIM>(unpack_dest,
                                                round_request_positions,
                                                static_cast<size_t>(total_round_send),
                                                elem_size,
                                                recv_staging.ptr(0),
                                                stream);
    } else {
      unpack_recv_into_output<CODE, UNPACK_DIM>(unpack_dest,
                                                round_recv_offsets.ptr(0),
                                                static_cast<size_t>(total_round_recv),
                                                elem_size,
                                                recv_staging.ptr(0),
                                                stream);
    }
  }
}

// Picks the per-peer element budget given a total per-buffer byte budget
// and the global maximum per-pair count (from `allreduce_global_max_pair_count`).
// The result is at least 1, never larger than `global_max` (so small workloads
// don't over-allocate), and at most `max_staging_bytes / num_ranks / elem_size`.
[[nodiscard]] inline size_t compute_max_elems_per_peer(size_t max_staging_bytes,
                                                       size_t elem_size,
                                                       int num_ranks,
                                                       unsigned long long global_max)
{
  const size_t per_peer_byte_budget =
    std::max<size_t>(elem_size, max_staging_bytes / static_cast<size_t>(num_ranks));
  // Callers (global_all2all / global_nccl_scatter) early-return when
  // global_max == 0, so global_max >= 1 here and `std::clamp`'s
  // lo <= hi precondition holds.
  return std::clamp<size_t>(per_peer_byte_budget / elem_size, 1, static_cast<size_t>(global_max));
}

// Derives the per-buffer staging byte budget from the user-facing
// `staging_factor` setting using the global index volume passed by the
// Python launcher. Sized as
// `staging_factor * (global_index_volume / num_ranks) * elem_size`, i.e.
// `staging_factor` times the average per-rank request count. Using the
// global volume (rather than each task's local slice) gives every rank
// an identical budget regardless of partition skew.
[[nodiscard]] inline size_t compute_max_staging_bytes(double staging_factor,
                                                      unsigned long long global_index_volume,
                                                      size_t elem_size,
                                                      int num_ranks)
{
  if (num_ranks <= 0) {
    return elem_size;
  }
  if (!(staging_factor > 0.0)) {
    LEGATE_ABORT("CUPYNUMERIC_ALL2ALL_STAGING_FACTOR must be a positive finite value, got ",
                 staging_factor);
  }
  const double avg_local_count =
    static_cast<double>(global_index_volume) / static_cast<double>(num_ranks);
  const double bytes = staging_factor * avg_local_count * static_cast<double>(elem_size);
  return std::max<size_t>(elem_size, static_cast<size_t>(bytes));
}

}  // namespace detail
}  // namespace cupynumeric
