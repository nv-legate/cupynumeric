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

#include "cupynumeric/all2all/all2all.h"
#include "cupynumeric/index/gather_template.inl"
#include "cupynumeric/pitches.h"
#include "cupynumeric/utilities/thrust_allocator.h"
#include "cupynumeric/utilities/thrust_util.h"
#include "cupynumeric/cuda_help.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <numeric>
#include <type_traits>
#include <cub/device/device_histogram.cuh>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <vector>

namespace cupynumeric {

// Distributed gather via all-to-all shuffle (NCCL)
//
// Implements:  output[p] = source[index[p]] for all points in the index array
//
// The source array is partitioned across N ranks.  Each rank holds a local
// slice of the index array whose points may refer to data owned by *any*
// rank.  The algorithm moves requests to the owning rank, gathers the data,
// and returns results to the requester in several steps:
//
// Step 1: Discover partition layout (AllGather)
//   Each rank AllGathers its local partition Rect<DIM> so every rank knows
//   which rank owns which region of the source array.
//
// Step 2: Classify requests & build ShuffleDescriptor
//   a) assign_target_ranks_kernel  — For each local index point, determine
//      which rank owns it by testing containment in the global rects.
//   b) cub::DeviceHistogram::HistogramEven — Count how many requests go
//      to each rank (send_histo) directly from the unsorted target ranks.
//   c) Group request indices into per-rank buckets (pack_indices_by_rank_warp)
//   d) Exchange histograms via NCCL (pairwise send/recv of one uint32 per
//      rank pair).  After this every rank knows:
//        send_histo[r] = how many requests I send to rank r
//        recv_histo[r] = how many requests rank r sends to me
//   e) Prefix-sum the histograms to get send_offsets and recv_offsets.
//
// Step 3: Linearize & exchange offsets
//   a) linearize_offsets_kernel — Convert each Point<DIM> into a uint64
//      flat offset relative to the target rank's rect.  This reduces the
//      per-element payload from sizeof(Point<DIM>) (e.g. 24 bytes for 3D)
//      to 8 bytes.
//   b) Exchange the flat offsets via NCCL pairwise send/recv.  After this,
//      each rank knows which elements other ranks need from its local data.
//
// Step 4: Gather local data & exchange values
//   a) gather_data_by_offset_kernel — Each rank gathers from its own local
//      source data at the flat offsets received in Phase 3.
//   b) Exchange gathered values via NCCL (direction reversed from Phase 3:
//      data flows from owner back to requester).
//
// Step 5: Unpack recv into output
//   unpack_recv_data_kernel — Place the received values into the output
//   array at the original request positions (tracked since Phase 2b).

using namespace legate;

static constexpr size_t BLOCK_SIZE = 256;

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

[[nodiscard]] static Buffer<int8_t> compute_histogram(const int* samples,
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

  return temp_storage;
}

// If the number of GPUs is not equal to the number of processes, we need to use a barrier
// to ensure that all the processes are synchronized.
[[nodiscard]] static bool needs_nccl_barrier()
{
  const auto machine   = legate::get_machine();
  const auto num_gpus  = machine.count(legate::mapping::TaskTarget::GPU);
  const auto num_procs = machine.count();

  return num_gpus != num_procs;
}

// Apply a barrier if needed before NCCL operations.
static void barrier_if_needed(TaskContext& context, bool needs_barrier)
{
  if (needs_barrier) {
    context.concurrent_task_barrier();
  }
}

// ShuffleDescriptor is a helper class that stores the shuffle descriptor for the
// all2all operation.
struct ShuffleDescriptor {
  // Number of local requests that will be sent to each destination rank.
  std::vector<unsigned long long> h_send_histo;
  // Prefix sums of h_send_histo; start offset of each rank's bucket in the send order.
  std::vector<unsigned long long> h_send_offsets;
  // Number of requests this rank will receive from each source rank.
  std::vector<unsigned long long> h_recv_histo;
  // Prefix sums of h_recv_histo; start offset of each source rank's bucket in
  // the receive order.
  std::vector<unsigned long long> h_recv_offsets;
  // Total number of requests this rank will receive from all other ranks.
  size_t total_incoming;

  ShuffleDescriptor(const unsigned long long* d_send_histo,
                    const unsigned long long* d_send_offsets,
                    const unsigned long long* d_recv_histo,
                    const unsigned long long* d_recv_offsets,
                    int num_ranks,
                    cudaStream_t stream)
    : h_send_histo(num_ranks),
      h_send_offsets(num_ranks),
      h_recv_histo(num_ranks),
      h_recv_offsets(num_ranks),
      total_incoming(0)
  {
    const size_t bytes = num_ranks * sizeof(unsigned long long);

    CUPYNUMERIC_CHECK_CUDA(
      cudaMemcpyAsync(h_send_histo.data(), d_send_histo, bytes, cudaMemcpyDeviceToHost, stream));
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      h_send_offsets.data(), d_send_offsets, bytes, cudaMemcpyDeviceToHost, stream));
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemcpyAsync(h_recv_histo.data(), d_recv_histo, bytes, cudaMemcpyDeviceToHost, stream));
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      h_recv_offsets.data(), d_recv_offsets, bytes, cudaMemcpyDeviceToHost, stream));

    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

    total_incoming = std::accumulate(h_recv_histo.begin(), h_recv_histo.end(), size_t{0});
  }
};

// ============================================================================
// Step 1 — AllGather partition rects
// ============================================================================

template <int DIM_input>
static Buffer<int8_t> allgather_partition_rects(TaskContext& context,
                                                const legate::Rect<DIM_input>& input_rect,
                                                ncclComm_t* nccl_comm,
                                                bool use_nccl_barrier,
                                                cudaStream_t stream)
{
  const size_t num_ranks     = context.get_launch_domain().get_volume();
  constexpr size_t rect_size = sizeof(legate::Rect<DIM_input>);
  auto global_rects      = create_buffer<int8_t>(num_ranks * rect_size, Memory::Kind::GPU_FB_MEM);
  auto input_rect_device = create_buffer<int8_t>(rect_size, Memory::Kind::GPU_FB_MEM);

  CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
    input_rect_device.ptr(0), &input_rect, rect_size, cudaMemcpyHostToDevice, stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(global_rects.ptr(0), 0, num_ranks * rect_size, stream));

  barrier_if_needed(context, use_nccl_barrier);
  CHECK_NCCL(ncclAllGather(
    input_rect_device.ptr(0), global_rects.ptr(0), rect_size, ncclInt8, *nccl_comm, stream));
  barrier_if_needed(context, use_nccl_barrier);
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  return global_rects;
}

// ============================================================================
// Step 2 — Classify requests & build ShuffleDescriptor
// ============================================================================

template <typename PointLoader, int DIM_input>
__global__ void assign_target_ranks_kernel(PointLoader loader,
                                           size_t count,
                                           int* target_ranks,
                                           const legate::Rect<DIM_input>* rects,
                                           int num_ranks,
                                           int tile_size)
{
  extern __shared__ char smem_raw[];
  auto* s_rects = reinterpret_cast<legate::Rect<DIM_input>*>(smem_raw);

  const size_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = idx < count;
  bool found        = false;

  const legate::Point<DIM_input> point = active ? loader(idx) : legate::Point<DIM_input>{};

  for (int tile_start = 0; tile_start < num_ranks; tile_start += tile_size) {
    const int this_tile = min(tile_size, num_ranks - tile_start);

    for (int i = threadIdx.x; i < this_tile; i += blockDim.x) {
      s_rects[i] = rects[tile_start + i];
    }
    __syncthreads();

    if (active && !found) {
      for (int r = 0; r < this_tile; r++) {
        bool inside = true;
        for (int d = 0; d < DIM_input; d++) {
          if (s_rects[r].lo[d] > point[d] || s_rects[r].hi[d] < point[d]) {
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
static void classify_index_points(
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t num_requests,
  int* target_ranks,
  const void* global_rects,
  int num_ranks,
  cudaStream_t stream)
{
  const size_t grid = (num_requests + BLOCK_SIZE - 1) / BLOCK_SIZE;

  using AccLoader = AccessorPointLoader<DIM_input, DIM_output>;
  using DnsLoader = DensePointLoader<DIM_input>;

  const auto& properties = get_device_properties();

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

  const auto* rects_ptr = static_cast<const legate::Rect<DIM_input>*>(global_rects);

  if (index.is_dense()) {
    const DnsLoader loader{index.dense_ptr};
    assign_target_ranks_kernel<DnsLoader, DIM_input><<<grid, BLOCK_SIZE, smem_size, stream>>>(
      loader, num_requests, target_ranks, rects_ptr, num_ranks, tile_size);
  } else {
    const AccLoader loader{index.acc, index.pitches, index.lo};
    assign_target_ranks_kernel<AccLoader, DIM_input><<<grid, BLOCK_SIZE, smem_size, stream>>>(
      loader, num_requests, target_ranks, rects_ptr, num_ranks, tile_size);
  }
}

// Group request indices into per-rank buckets.
__global__ void pack_indices_by_rank_warp(const int* __restrict__ target_ranks,
                                          size_t count,
                                          unsigned long long* __restrict__ next_slot,
                                          uint64_t* __restrict__ packed_positions)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= count) {
    return;
  }

  const int r          = target_ranks[idx];
  const unsigned amask = __activemask();
  const unsigned peers = __match_any_sync(amask, r);
  const int lane       = threadIdx.x & 31;
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
[[nodiscard]] static ShuffleDescriptor create_shuffle_information(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t num_requests,
  const void* global_rects,
  uint64_t* request_positions,
  int* target_ranks,
  unsigned long long* d_send_offsets,
  ncclComm_t* nccl_comm,
  bool use_nccl_barrier,
  cudaStream_t stream)
{
  const int num_ranks = static_cast<int>(context.get_launch_domain().get_volume());
  auto alloc          = ThrustAllocator(Memory::GPU_FB_MEM);
  auto exec_policy    = DEFAULT_POLICY(alloc).on(stream);

  auto send_histo   = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);
  auto recv_histo   = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);
  auto recv_offsets = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);

  CUPYNUMERIC_CHECK_CUDA(
    cudaMemsetAsync(send_histo.ptr(0), 0, num_ranks * sizeof(unsigned long long), stream));
  CUPYNUMERIC_CHECK_CUDA(
    cudaMemsetAsync(recv_histo.ptr(0), 0, num_ranks * sizeof(unsigned long long), stream));

  Buffer<int8_t> histogram_temp_storage;

  if (num_requests > 0) {
    classify_index_points<DIM_input, DIM_output>(
      index, num_requests, target_ranks, global_rects, num_ranks, stream);

    // compute the histogram and keep the temporary storage until stream synchronization
    histogram_temp_storage =
      compute_histogram(target_ranks, num_requests, send_histo.ptr(0), num_ranks, stream);
  }

  // prefix-sum the send histograms to get send_offsets
  thrust::exclusive_scan(
    exec_policy, send_histo.ptr(0), send_histo.ptr(0) + num_ranks, d_send_offsets);

  if (num_requests > 0) {
    auto next_slot = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);

    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(next_slot.ptr(0),
                                           d_send_offsets,
                                           num_ranks * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToDevice,
                                           stream));

    const size_t grid = (num_requests + BLOCK_SIZE - 1) / BLOCK_SIZE;

    pack_indices_by_rank_warp<<<grid, BLOCK_SIZE, 0, stream>>>(
      target_ranks, num_requests, next_slot.ptr(0), request_positions);
  }

  // do the NCCL exchanges of the histograms, after this, each rank knows how many requests
  // it sends to each other rank and how many requests it receives from each other rank
  barrier_if_needed(context, use_nccl_barrier);
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    CHECK_NCCL(ncclRecv(recv_histo.ptr(0) + i, 1, ncclUint64, i, *nccl_comm, stream));
    CHECK_NCCL(ncclSend(send_histo.ptr(0) + i, 1, ncclUint64, i, *nccl_comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  barrier_if_needed(context, use_nccl_barrier);

  // prefix-sum the receive histograms to get recv_offsets
  thrust::exclusive_scan(
    exec_policy, recv_histo.ptr(0), recv_histo.ptr(0) + num_ranks, recv_offsets.ptr(0));

  return ShuffleDescriptor(
    send_histo.ptr(0), d_send_offsets, recv_histo.ptr(0), recv_offsets.ptr(0), num_ranks, stream);
}

// ============================================================================
// Step 3 — Linearize & exchange offsets
// ============================================================================

template <typename PointLoader, int DIM_input>
__global__ void linearize_offsets_kernel(PointLoader loader,
                                         const uint64_t* perm,
                                         const unsigned long long* send_offsets,
                                         int num_ranks,
                                         const legate::Rect<DIM_input>* global_rects,
                                         uint64_t* offsets_out,
                                         size_t count)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= count) {
    return;
  }

  int lo = 0;
  int hi = num_ranks;
  // binary search to find the rank that owns the request
  while (lo < hi) {
    const int mid = (lo + hi) / 2;

    if (send_offsets[mid] <= idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const int r = lo - 1;

  assert(r >= 0);

  const legate::Point<DIM_input> point = loader(perm[idx]);
  const legate::Rect<DIM_input> rect   = global_rects[r];
  uint64_t offset                      = 0;

  for (int d = 0; d < DIM_input - 1; d++) {
    offset += (point[d] - rect.lo[d]);
    offset *= (rect.hi[d + 1] - rect.lo[d + 1] + 1);
  }
  offset += (point[DIM_input - 1] - rect.lo[DIM_input - 1]);
  offsets_out[idx] = offset;
}

template <int DIM_input, int DIM_output>
[[nodiscard]] static Buffer<uint64_t> linearize_and_exchange_offsets(
  TaskContext& context,
  const StoreView<AccessMode::Read, legate::Point<DIM_input>, DIM_output>& index,
  size_t num_requests,
  const uint64_t* request_positions,
  const unsigned long long* d_send_offsets,
  const void* global_rects,
  const ShuffleDescriptor& plan,
  ncclComm_t* nccl_comm,
  bool use_nccl_barrier,
  cudaStream_t stream)
{
  const int num_ranks   = static_cast<int>(context.get_launch_domain().get_volume());
  auto send_offsets_buf = create_buffer<uint64_t>(num_requests, Memory::Kind::GPU_FB_MEM);

  if (num_requests > 0) {
    const size_t grid     = (num_requests + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto* rects_ptr = static_cast<const legate::Rect<DIM_input>*>(global_rects);

    if (index.is_dense()) {
      const DensePointLoader<DIM_input> loader{index.dense_ptr};

      linearize_offsets_kernel<decltype(loader), DIM_input>
        <<<grid, BLOCK_SIZE, 0, stream>>>(loader,
                                          request_positions,
                                          d_send_offsets,
                                          num_ranks,
                                          rects_ptr,
                                          send_offsets_buf.ptr(0),
                                          num_requests);
    } else {
      const AccessorPointLoader<DIM_input, DIM_output> loader{index.acc, index.pitches, index.lo};

      linearize_offsets_kernel<decltype(loader), DIM_input>
        <<<grid, BLOCK_SIZE, 0, stream>>>(loader,
                                          request_positions,
                                          d_send_offsets,
                                          num_ranks,
                                          rects_ptr,
                                          send_offsets_buf.ptr(0),
                                          num_requests);
    }
  }

  auto recv_offsets_buf = create_buffer<uint64_t>(plan.total_incoming, Memory::Kind::GPU_FB_MEM);

  // do the NCCL exchanges of the offsets, after this, each rank knows which
  // elements other ranks need from its local data
  barrier_if_needed(context, use_nccl_barrier);
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    if (plan.h_send_histo[i] > 0) {
      CHECK_NCCL(ncclSend(send_offsets_buf.ptr(0) + plan.h_send_offsets[i],
                          plan.h_send_histo[i],
                          ncclUint64,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (plan.h_recv_histo[i] > 0) {
      CHECK_NCCL(ncclRecv(recv_offsets_buf.ptr(0) + plan.h_recv_offsets[i],
                          plan.h_recv_histo[i],
                          ncclUint64,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  barrier_if_needed(context, use_nccl_barrier);
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  return recv_offsets_buf;
}

// ============================================================================
// Step 4 — Gather local data & exchange values
// ============================================================================

template <typename DataType, int DIM_input>
__global__ void gather_data_by_offset_kernel(AccessorRO<DataType, DIM_input> input_acc,
                                             Pitches<DIM_input - 1> input_pitches,
                                             legate::Point<DIM_input> input_lo,
                                             const uint64_t* offsets,
                                             size_t count,
                                             DataType* out)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= count) {
    return;
  }

  const auto p = input_pitches.unflatten(offsets[idx], input_lo);

  out[idx] = input_acc[p];
}

__global__ void gather_data_by_offset_kernel_dense(const char* __restrict__ input_ptr,
                                                   const uint64_t* offsets,
                                                   size_t count,
                                                   size_t elem_size,
                                                   char* out)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= count) {
    return;
  }

  memcpy(out + idx * elem_size, input_ptr + offsets[idx] * elem_size, elem_size);
}

template <Type::Code CODE, int DIM_input>
static void local_gather_and_exchange(
  TaskContext& context,
  const StoreView<AccessMode::Read, type_of<CODE>, DIM_input>& input,
  const uint64_t* recv_flat_offsets,
  size_t elem_size,
  const ShuffleDescriptor& plan,
  int8_t* recv_data_buf,
  ncclComm_t* nccl_comm,
  bool use_nccl_barrier,
  cudaStream_t stream)
{
  using VAL           = type_of<CODE>;
  const int num_ranks = static_cast<int>(context.get_launch_domain().get_volume());
  auto send_data = create_buffer<int8_t>(plan.total_incoming * elem_size, Memory::Kind::GPU_FB_MEM);

  if (plan.total_incoming > 0) {
    const size_t grid = (plan.total_incoming + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (input.is_dense()) {
      gather_data_by_offset_kernel_dense<<<grid, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const char*>(input.dense_ptr),
        recv_flat_offsets,
        plan.total_incoming,
        elem_size,
        reinterpret_cast<char*>(send_data.ptr(0)));
    } else {
      gather_data_by_offset_kernel<VAL, DIM_input>
        <<<grid, BLOCK_SIZE, 0, stream>>>(input.acc,
                                          input.pitches,
                                          input.lo,
                                          recv_flat_offsets,
                                          plan.total_incoming,
                                          reinterpret_cast<VAL*>(send_data.ptr(0)));
    }
  }

  // do the NCCL exchages of the gathered data, after this, each rank has the data
  // it needs from other ranks
  barrier_if_needed(context, use_nccl_barrier);
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < num_ranks; ++i) {
    if (plan.h_recv_histo[i] > 0) {
      CHECK_NCCL(ncclSend(send_data.ptr(0) + plan.h_recv_offsets[i] * elem_size,
                          plan.h_recv_histo[i] * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
    if (plan.h_send_histo[i] > 0) {
      CHECK_NCCL(ncclRecv(recv_data_buf + plan.h_send_offsets[i] * elem_size,
                          plan.h_send_histo[i] * elem_size,
                          ncclInt8,
                          i,
                          *nccl_comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  barrier_if_needed(context, use_nccl_barrier);
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
}

// ============================================================================
// Step 5 — Unpack received values into output
// ============================================================================

template <typename DataType, int DIM_output>
__global__ void unpack_recv_data_kernel(AccessorRW<DataType, DIM_output> output_acc,
                                        Pitches<DIM_output - 1> output_pitches,
                                        legate::Point<DIM_output> output_lo,
                                        const uint64_t* request_indices,
                                        size_t count,
                                        const DataType* recv_data)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= count) {
    return;
  }

  const auto p = output_pitches.unflatten(request_indices[idx], output_lo);

  output_acc[p] = recv_data[idx];
}

__global__ void unpack_recv_data_kernel_dense(char* __restrict__ output_ptr,
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
static void unpack_recv_into_output(
  const StoreView<AccessMode::Write, type_of<CODE>, DIM_output>& output,
  const uint64_t* request_positions,
  size_t num_requests,
  const void* recv_data,
  cudaStream_t stream)
{
  using VAL = type_of<CODE>;

  if (num_requests > 0) {
    const size_t grid = (num_requests + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (output.is_dense()) {
      unpack_recv_data_kernel_dense<<<grid, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<char*>(output.dense_ptr),
        request_positions,
        num_requests,
        sizeof(VAL),
        static_cast<const char*>(recv_data));
    } else {
      unpack_recv_data_kernel<VAL, DIM_output>
        <<<grid, BLOCK_SIZE, 0, stream>>>(output.acc,
                                          output.pitches,
                                          output.lo,
                                          request_positions,
                                          num_requests,
                                          static_cast<const VAL*>(recv_data));
    }
  }
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
}

// ============================================================================
// Runs steps 1-5
// ============================================================================

template <Type::Code CODE, int DIM_input, int DIM_output>
static void global_all2all(TaskContext& context,
                           AccessorRO<type_of<CODE>, DIM_input> input_acc,
                           AccessorRO<legate::Point<DIM_input>, DIM_output> index_acc,
                           AccessorRW<type_of<CODE>, DIM_output> output_acc,
                           const legate::Rect<DIM_input>& input_rect,
                           const legate::Rect<DIM_output>& index_rect,
                           const legate::Rect<DIM_output>& output_rect,
                           size_t local_index_count,
                           bool use_nccl_barrier,
                           ncclComm_t* nccl_comm,
                           cudaStream_t stream)
{
  using VAL                 = type_of<CODE>;
  using INDEX_VAL           = legate::Point<DIM_input>;
  const size_t num_requests = local_index_count;
  const int num_ranks       = static_cast<int>(context.get_launch_domain().get_volume());

  StoreView<AccessMode::Read, INDEX_VAL, DIM_output> index_view(index_acc, index_rect);
  StoreView<AccessMode::Read, VAL, DIM_input> input_view(input_acc, input_rect);
  StoreView<AccessMode::Write, VAL, DIM_output> output_view(output_acc, output_rect);

  auto global_rects =
    allgather_partition_rects<DIM_input>(context, input_rect, nccl_comm, use_nccl_barrier, stream);

  auto request_positions = create_buffer<uint64_t>(num_requests, Memory::Kind::GPU_FB_MEM);
  auto target_ranks      = create_buffer<int>(num_requests, Memory::Kind::GPU_FB_MEM);
  auto d_send_offsets    = create_buffer<unsigned long long>(num_ranks, Memory::Kind::GPU_FB_MEM);
  auto recv_data_buf = create_buffer<int8_t>(num_requests * sizeof(VAL), Memory::Kind::GPU_FB_MEM);

  if (num_requests > 0) {
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(target_ranks.ptr(0), 0xff, num_requests * sizeof(int), stream));
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(recv_data_buf.ptr(0), 0, num_requests * sizeof(VAL), stream));
  }

  auto plan = create_shuffle_information<DIM_input, DIM_output>(context,
                                                                index_view,
                                                                num_requests,
                                                                global_rects.ptr(0),
                                                                request_positions.ptr(0),
                                                                target_ranks.ptr(0),
                                                                d_send_offsets.ptr(0),
                                                                nccl_comm,
                                                                use_nccl_barrier,
                                                                stream);

  auto recv_flat_offsets =
    linearize_and_exchange_offsets<DIM_input, DIM_output>(context,
                                                          index_view,
                                                          num_requests,
                                                          request_positions.ptr(0),
                                                          d_send_offsets.ptr(0),
                                                          global_rects.ptr(0),
                                                          plan,
                                                          nccl_comm,
                                                          use_nccl_barrier,
                                                          stream);

  local_gather_and_exchange<CODE, DIM_input>(context,
                                             input_view,
                                             recv_flat_offsets.ptr(0),
                                             sizeof(VAL),
                                             plan,
                                             recv_data_buf.ptr(0),
                                             nccl_comm,
                                             use_nccl_barrier,
                                             stream);

  unpack_recv_into_output<CODE, DIM_output>(
    output_view, request_positions.ptr(0), num_requests, recv_data_buf.ptr(0), stream);
}

// ============================================================================
// Legate task dispatch
// ============================================================================

template <Type::Code CODE, int32_t DIM_input, int32_t DIM_output>
struct All2AllGPUBody {
  using VAL       = type_of<CODE>;
  using INDEX_VAL = legate::Point<DIM_input>;

  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input_array,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output_array)
  {
    const auto stream           = context.get_task_stream();
    const auto input_rect       = input_array.shape<DIM_input>();
    const auto index_rect       = index_array.shape<DIM_output>();
    const auto output_rect      = output_array.shape<DIM_output>();
    const bool use_nccl_barrier = needs_nccl_barrier();

    const auto input = [&]() -> AccessorRO<VAL, DIM_input> {
      if (!input_rect.empty()) {
        return input_array.read_accessor<VAL, DIM_input>(input_rect);
      }
      return {};
    }();

    const auto index = [&]() -> AccessorRO<INDEX_VAL, DIM_output> {
      if (!index_rect.empty()) {
        return index_array.read_accessor<INDEX_VAL, DIM_output>(index_rect);
      }
      return {};
    }();

    const auto output = [&]() -> AccessorRW<VAL, DIM_output> {
      if (!index_rect.empty()) {
        return output_array.read_write_accessor<VAL, DIM_output>(output_rect);
      }
      return {};
    }();

    const size_t local_index_count = [&]() -> size_t {
      if (index_rect.empty()) {
        return 0;
      }
      size_t count = 1;
      for (int d = 0; d < DIM_output; d++) {
        const auto extent = index_rect.hi[d] - index_rect.lo[d] + 1;
        count *= (extent > 0) ? static_cast<size_t>(extent) : 0;
      }
      return count;
    }();

    global_all2all<CODE, DIM_input, DIM_output>(context,
                                                input,
                                                index,
                                                output,
                                                input_rect,
                                                index_rect,
                                                output_rect,
                                                local_index_count,
                                                use_nccl_barrier,
                                                context.communicators()[0].get<ncclComm_t*>(),
                                                stream);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <int DIM_input, int DIM_output>
struct All2AllImpl_type {
  template <Type::Code CODE>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output) const
  {
    All2AllGPUBody<CODE, DIM_input, DIM_output>()(context, input, index_array, output);
  }
};

struct All2AllImpl {
  template <int DIM_input, int DIM_output>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output) const
  {
    type_dispatch(
      input.code(), All2AllImpl_type<DIM_input, DIM_output>{}, context, input, index_array, output);
  }
};

static void all2all_gpu(TaskContext& context)
{
  const auto input       = context.input(0);
  const auto index_array = context.input(1);
  auto output            = context.output(0);

  const auto domain         = context.get_launch_domain();
  const size_t num_ranks    = domain.get_volume();
  const bool is_index_space = !context.is_single_task() && context.communicators().size() > 0;

  assert(is_index_space || num_ranks == 1);

  if (is_index_space) {
    const auto dim_input  = std::max(input.dim(), 1);
    const auto dim_output = std::max(output.dim(), 1);

    legate::double_dispatch(
      dim_input, dim_output, All2AllImpl{}, context, input, index_array, output);
  } else {
    const auto stream = context.get_task_stream();

    type_dispatch(
      input.type().code(),
      GatherTypeDispatch<decltype(DEFAULT_POLICY.on(stream))>{DEFAULT_POLICY.on(stream)},
      output,
      input,
      index_array);
  }
}

/*static*/ void All2AllTask::gpu_variant(TaskContext context) { all2all_gpu(context); }

}  // namespace cupynumeric
