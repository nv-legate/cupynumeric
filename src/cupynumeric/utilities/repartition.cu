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

#include "repartition.h"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace Legion;
using namespace legate;

namespace {
// auto align to multiples of 16 bytes
constexpr auto get_16b_aligned = [](auto bytes) {
  return std::max<size_t>(16, (bytes + 15) / 16 * 16);
};
constexpr auto get_16b_aligned_count = [](auto count, auto element_bytes) {
  return (get_16b_aligned(count * element_bytes) + element_bytes - 1) / element_bytes;
};

const auto is_device_only_ptr = [](const void* ptr) {
  cudaPointerAttributes attrs;
  auto res = cudaPointerGetAttributes(&attrs, ptr);
  if (res == cudaSuccess) {
    return attrs.type == cudaMemoryTypeDevice;
  } else {
    cudaGetLastError();
    return false;
  }
};
}  // namespace

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  split_data_to_send_buffers(const VAL* input_2dbc,
                             size_t input_volume,
                             size_t input_lld,
                             Buffer<size_t> send_info,
                             size_t stored_size_per_rank,
                             Buffer<VAL*> send_buffers_ptr,
                             size_t p_r,
                             size_t p_c,
                             size_t tile_r,
                             size_t tile_c)
{
  size_t thread_offset    = blockIdx.x * blockDim.x + threadIdx.x;
  size_t threadgroup_size = blockDim.x * gridDim.x;
  size_t rank_id          = blockIdx.y * blockDim.y + threadIdx.y;

  if (rank_id >= p_r * p_c) {
    return;
  }

  size_t source_size       = send_info[rank_id * stored_size_per_rank + BlockInfo::TOTAL_SIZE];
  size_t source_lld        = send_info[rank_id * stored_size_per_rank + BlockInfo::LLD];
  size_t source_offset_row = send_info[rank_id * stored_size_per_rank + BlockInfo::OFFSET_ROW];
  size_t source_offset_col = send_info[rank_id * stored_size_per_rank + BlockInfo::OFFSET_COL];

  // copy large block from input with all elements for target rank_id
  for (size_t pos = thread_offset; pos < source_size; pos += threadgroup_size) {
    size_t source_row_id = source_offset_row + pos % source_lld;
    size_t source_col_id = source_offset_col + pos / source_lld;
    size_t index_in      = source_col_id * input_lld + source_row_id;

    assert(index_in < input_volume);
    send_buffers_ptr[rank_id][pos] = input_2dbc[index_in];
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  merge_data_to_result(Buffer<VAL> result_2dbc,
                       size_t volume,
                       Buffer<size_t> recv_info,
                       size_t stored_size_per_rank,
                       Buffer<VAL*> merge_buffers,
                       size_t target_lld,
                       size_t tile_r,
                       size_t tile_c,
                       size_t my_rank,
                       size_t num_ranks)
{
  size_t thread_offset    = blockIdx.x * blockDim.x + threadIdx.x;
  size_t threadgroup_size = blockDim.x * gridDim.x;
  size_t rank_id          = blockIdx.y * blockDim.y + threadIdx.y;

  if (rank_id >= num_ranks) {
    return;
  }

  size_t source_size       = recv_info[rank_id * stored_size_per_rank + BlockInfo::TOTAL_SIZE];
  size_t source_lld        = recv_info[rank_id * stored_size_per_rank + BlockInfo::LLD];
  size_t source_offset_row = recv_info[rank_id * stored_size_per_rank + BlockInfo::OFFSET_ROW];
  size_t source_offset_col = recv_info[rank_id * stored_size_per_rank + BlockInfo::OFFSET_COL];

  for (size_t pos = thread_offset; pos < source_size; pos += threadgroup_size) {
    size_t target_col_id = source_offset_col + pos / source_lld;
    size_t target_row_id = source_offset_row + pos % source_lld;

    // store elementwise
    size_t index_out = target_col_id * target_lld + target_row_id;

    assert(index_out < volume);
    result_2dbc[index_out] = merge_buffers[rank_id][pos];
  }
}

__device__ __inline__ std::tuple<size_t, size_t, size_t> compute_tile_info(size_t num_rows,
                                                                           size_t num_cols,
                                                                           size_t row_major,
                                                                           size_t lld,
                                                                           size_t offset_r,
                                                                           size_t offset_c,
                                                                           size_t tile_r,
                                                                           size_t tile_c)
{
  // position info
  // get local tile size and start position
  size_t tile_r_size = tile_r;
  size_t tile_c_size = tile_c;
  size_t start_pos;
  // special cases for first/last tile
  {
    size_t start_r_offset = offset_r % tile_r;
    size_t start_c_offset = offset_c % tile_c;
    size_t start_pos_r    = blockIdx.x * tile_r;
    size_t start_pos_c    = blockIdx.y * tile_c;

    // rows
    if (start_r_offset > 0) {
      if (blockIdx.x == 0) {
        tile_r_size -= start_r_offset;
      } else {
        start_pos_r -= start_r_offset;
      }
    }
    if (blockIdx.x == gridDim.x - 1) {
      size_t last_element_offset = (num_rows + start_r_offset) % tile_r;
      if (last_element_offset > 0) {
        tile_r_size -= (tile_r - last_element_offset);
      }
    }
    // cols
    if (start_c_offset > 0) {
      if (blockIdx.y == 0) {
        tile_c_size -= start_c_offset;
      } else {
        start_pos_c -= start_c_offset;
      }
    }
    if (blockIdx.y == gridDim.y - 1) {
      size_t last_element_offset = (num_cols + start_c_offset) % tile_c;
      if (last_element_offset > 0) {
        tile_c_size -= (tile_c - last_element_offset);
      }
    }

    start_pos = row_major ? start_pos_r * lld + start_pos_c : start_pos_c * lld + start_pos_r;
  }

  return {tile_r_size, tile_c_size, start_pos};
}

__device__ __inline__ std::tuple<size_t, size_t, size_t, size_t> compute_2dbc_info(
  Buffer<size_t> info,
  size_t stored_size_per_rank,
  size_t p_r,
  size_t p_c,
  size_t tile_idx_row,
  size_t tile_idx_col,
  size_t tile_r,
  size_t tile_c)
{
  size_t rank_r    = tile_idx_row % p_r;
  size_t rank_c    = tile_idx_col % p_c;
  size_t rank_id   = rank_r + rank_c * p_r;  // tile ranks are col major
  size_t size      = info[rank_id * stored_size_per_rank + BlockInfo::TOTAL_SIZE];
  size_t lld       = info[rank_id * stored_size_per_rank + BlockInfo::LLD];
  size_t start_pos = 0;
  // compute start position of tile (tile_idx_row/tile_idx_col) within source
  {
    // this is where the OUR part of the whole 2dbc dist of the target rank resides
    size_t offset_row = info[rank_id * stored_size_per_rank + BlockInfo::OFFSET_ROW];
    size_t offset_col = info[rank_id * stored_size_per_rank + BlockInfo::OFFSET_COL];

    // this is where the tile starts / it does not have to be where our PART of the tile starts
    size_t tile_pos_row = (tile_idx_row - rank_r) / p_r * tile_r;
    size_t tile_pos_col = (tile_idx_col - rank_c) / p_c * tile_c;

    // shift to the positions where our PART of the tile starts
    if (tile_pos_row > offset_row) {
      tile_pos_row -= offset_row;
    } else {
      tile_pos_row = 0;
    }
    if (tile_pos_col > offset_col) {
      tile_pos_col -= offset_col;
    } else {
      tile_pos_col = 0;
    }

    start_pos = tile_pos_col * lld + tile_pos_row;  // always col major
  }
  return {rank_id, size, lld, start_pos};
}

#define BLOCK_DIM 16
template <typename VAL>
__device__ __inline__ void transfer_data_src_tgt(const VAL* source,
                                                 size_t source_size,
                                                 size_t source_lld,
                                                 bool source_row_major,
                                                 size_t source_start_pos,
                                                 VAL* target,
                                                 size_t target_size,
                                                 size_t target_lld,
                                                 size_t target_row_major,
                                                 size_t target_start_pos,
                                                 size_t tile_r_size,
                                                 size_t tile_c_size,
                                                 VAL block[BLOCK_DIM][BLOCK_DIM + 1])
{
  if (source_row_major != target_row_major) {
    for (size_t tile_r_pos = 0; tile_r_pos < tile_r_size; tile_r_pos += BLOCK_DIM) {
      for (size_t tile_c_pos = 0; tile_c_pos < tile_c_size; tile_c_pos += BLOCK_DIM) {
        // we are at offset tile_r_pos/tile_c_pos within our tile (start of block)
        // blocks are square, tiles don't need to be!
        size_t tile_r_pos_t = tile_r_pos + threadIdx.y;
        size_t tile_c_pos_t = tile_c_pos + threadIdx.x;
        if (tile_r_pos_t < tile_r_size && tile_c_pos_t < tile_c_size) {
          size_t index_in =
            source_start_pos + (source_row_major ? tile_c_pos_t + tile_r_pos_t * source_lld
                                                 : tile_c_pos_t * source_lld + tile_r_pos_t);
          assert(index_in < source_size);
          block[threadIdx.y][threadIdx.x] = source[index_in];
        }

        __syncthreads();

        // write back data to target (row major OR column major)
        if (tile_r_pos + threadIdx.x < tile_r_size && tile_c_pos + threadIdx.y < tile_c_size) {
          size_t index_out =
            target_start_pos +
            (target_row_major ? (tile_r_pos + threadIdx.x) * target_lld + tile_c_pos + threadIdx.y
                              : tile_r_pos + threadIdx.x + (tile_c_pos + threadIdx.y) * target_lld);
          assert(index_out < target_size);
          target[index_out] = block[threadIdx.x][threadIdx.y];
        }
      }
    }
  } else {
    for (size_t tile_r_pos = threadIdx.x; tile_r_pos < tile_r_size; tile_r_pos += BLOCK_DIM) {
      for (size_t tile_c_pos = threadIdx.y; tile_c_pos < tile_c_size; tile_c_pos += BLOCK_DIM) {
        size_t index_in  = source_start_pos + tile_r_pos + tile_c_pos * source_lld;
        size_t index_out = target_start_pos + tile_r_pos + tile_c_pos * target_lld;
        assert(index_in < source_size);
        assert(index_out < target_size);
        target[index_out] = source[index_in];
      }
    }
  }
}

template <typename VAL>
__global__ void assemble_tiles_to_block_result(VAL* target,
                                               size_t target_volume,
                                               size_t target_lld,
                                               size_t target_offset_r,
                                               size_t target_offset_c,
                                               bool target_row_major,
                                               Buffer<size_t> recv_info,
                                               size_t stored_size_per_rank,
                                               Buffer<VAL*> recv_buffers_ptr,
                                               size_t p_r,
                                               size_t p_c,
                                               size_t tile_r,
                                               size_t tile_c)
{
  __shared__ VAL block[BLOCK_DIM][BLOCK_DIM + 1];

  size_t num_target_cols = target_row_major ? target_lld : target_volume / target_lld;
  size_t num_target_rows = target_row_major ? target_volume / target_lld : target_lld;

  size_t tile_idx_row = blockIdx.x + target_offset_r / tile_r;
  size_t tile_idx_col = blockIdx.y + target_offset_c / tile_c;

  auto [tile_r_size, tile_c_size, target_start_pos] = compute_tile_info(num_target_rows,
                                                                        num_target_cols,
                                                                        target_row_major,
                                                                        target_lld,
                                                                        target_offset_r,
                                                                        target_offset_c,
                                                                        tile_r,
                                                                        tile_c);

  auto [source_rank_id, source_size, source_lld, source_start_pos] = compute_2dbc_info(
    recv_info, stored_size_per_rank, p_r, p_c, tile_idx_row, tile_idx_col, tile_r, tile_c);

  transfer_data_src_tgt(recv_buffers_ptr[source_rank_id],
                        source_size,
                        source_lld,
                        false,
                        source_start_pos,
                        target,
                        target_volume,
                        target_lld,
                        target_row_major,
                        target_start_pos,
                        tile_r_size,
                        tile_c_size,
                        block);
}

template <typename VAL>
__global__ void copy_to_send_buffer(const VAL* input,
                                    size_t volume,
                                    Buffer<size_t> send_info,
                                    size_t stored_size_per_rank,
                                    Buffer<VAL*> send_buffers_ptr,
                                    bool row_major,
                                    size_t offset_r,
                                    size_t offset_c,
                                    size_t lld,
                                    size_t p_r,
                                    size_t p_c,
                                    size_t tile_r,
                                    size_t tile_c)
{
  __shared__ VAL block[BLOCK_DIM][BLOCK_DIM + 1];

  size_t num_input_cols = row_major ? lld : volume / lld;
  size_t num_input_rows = row_major ? volume / lld : lld;

  size_t tile_idx_row = blockIdx.x + offset_r / tile_r;
  size_t tile_idx_col = blockIdx.y + offset_c / tile_c;

  auto [tile_r_size, tile_c_size, source_start_pos] = compute_tile_info(
    num_input_rows, num_input_cols, row_major, lld, offset_r, offset_c, tile_r, tile_c);

  auto [target_rank_id, target_size, target_lld, target_start_pos] = compute_2dbc_info(
    send_info, stored_size_per_rank, p_r, p_c, tile_idx_row, tile_idx_col, tile_r, tile_c);

  transfer_data_src_tgt(input,
                        volume,
                        lld,
                        row_major,
                        source_start_pos,
                        send_buffers_ptr[target_rank_id],
                        target_size,
                        target_lld,
                        false,
                        target_start_pos,
                        tile_r_size,
                        tile_c_size,
                        block);
}

template <typename VAL>
std::tuple<Buffer<VAL>, size_t, size_t> repartition_matrix_2dbc(const VAL* input,
                                                                size_t volume,
                                                                bool row_major,
                                                                size_t offset_r,
                                                                size_t offset_c,
                                                                size_t lld,
                                                                size_t p_r,
                                                                size_t p_c,
                                                                size_t tile_r,
                                                                size_t tile_c,
                                                                comm::Communicator comm_wrapper)
{
  assert(volume == 0 || is_device_only_ptr(input));

  auto num_ranks  = p_r * p_c;
  size_t num_cols = row_major ? lld : volume / lld;
  size_t num_rows = row_major ? volume / lld : lld;

  auto comm   = comm_wrapper.get<ncclComm_t*>();
  auto stream = get_cached_stream();

  int nccl_rank  = -1;
  int nccl_ranks = -1;
  CHECK_NCCL(ncclCommUserRank(*comm, &nccl_rank));
  CHECK_NCCL(ncclCommCount(*comm, &nccl_ranks));
  assert(num_ranks == nccl_ranks);

  // compute sizes/lld/offset for each target rank
  size_t stored_size_per_rank = get_16b_aligned_count(BlockInfo::LAST, sizeof(size_t));
  size_t total_send_elements  = 0;
  Buffer<size_t> send_info =
    create_buffer<size_t>(num_ranks * stored_size_per_rank, Memory::Z_COPY_MEM);
  Buffer<size_t> recv_info =
    create_buffer<size_t>(num_ranks * stored_size_per_rank, Memory::Z_COPY_MEM);
  for (size_t rank_c = 0; rank_c < p_c; ++rank_c) {
    auto [active_columns, offset_columns] =
      elements_for_rank_in_dimension(num_cols, offset_c, rank_c, p_c, tile_c);
    for (size_t rank_r = 0; rank_r < p_r; ++rank_r) {
      auto glob_rank = rank_r + rank_c * p_r;  // target ranks are col major
      auto [active_rows, offset_rows] =
        elements_for_rank_in_dimension(num_rows, offset_r, rank_r, p_r, tile_r);

      auto elements_for_rank = active_columns * active_rows;
      total_send_elements += elements_for_rank;

      send_info[glob_rank * stored_size_per_rank + BlockInfo::TOTAL_SIZE] = elements_for_rank;
      send_info[glob_rank * stored_size_per_rank + BlockInfo::LLD] =
        active_rows;  // col-major send data
      send_info[glob_rank * stored_size_per_rank + BlockInfo::OFFSET_ROW] = offset_rows;
      send_info[glob_rank * stored_size_per_rank + BlockInfo::OFFSET_COL] = offset_columns;
    }
  }

  assert(total_send_elements == volume);

  // TODO / OPTIMIZE
  // in case we have the global partition information of the cuPyNumeric block partition
  // we can compute receive buffers instead and skip this all2all
  // same applies for inverse operation

  // all2all send_info/recv_info
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(
      send_info.ptr(r * stored_size_per_rank), stored_size_per_rank, ncclUint64, r, *comm, stream));
    CHECK_NCCL(ncclRecv(
      recv_info.ptr(r * stored_size_per_rank), stored_size_per_rank, ncclUint64, r, *comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));  // need Z-copy synchronized to Host

  // allocate send/recv buffer
  std::vector<Buffer<VAL>> send_buffers;
  send_buffers.reserve(num_ranks);
  std::vector<Buffer<VAL>> recv_buffers;
  recv_buffers.reserve(num_ranks);
  size_t total_receive = 0;
  size_t target_lld    = 0;
  for (size_t rank_c = 0; rank_c < p_c; ++rank_c) {
    for (size_t rank_r = 0; rank_r < p_r; ++rank_r) {
      auto glob_rank = rank_r + rank_c * p_r;  // target ranks are col major
      assert(send_buffers.size() == glob_rank);
      send_buffers.emplace_back(create_buffer<VAL>(
        send_info[glob_rank * stored_size_per_rank + BlockInfo::TOTAL_SIZE], Memory::GPU_FB_MEM));
      auto receive_size = recv_info[glob_rank * stored_size_per_rank + BlockInfo::TOTAL_SIZE];
      if (receive_size > 0) {
        target_lld =
          std::max(target_lld,
                   recv_info[glob_rank * stored_size_per_rank + BlockInfo::LLD] +
                     recv_info[glob_rank * stored_size_per_rank + BlockInfo::OFFSET_ROW]);
      }
      total_receive += receive_size;
      assert(recv_buffers.size() == glob_rank);
      recv_buffers.emplace_back(create_buffer<VAL>(receive_size, Memory::GPU_FB_MEM));
    }
  }

  // and package data for each target rank
  if (volume > 0) {
    Buffer<VAL*> send_buffers_ptr = create_buffer<VAL*>(num_ranks, Memory::Z_COPY_MEM);
    for (size_t r = 0; r < num_ranks; r++) {
      send_buffers_ptr[r] = send_buffers[r].ptr(0);
    }

    size_t first_tile_r = offset_r / tile_r;
    size_t last_tile_r  = (offset_r + num_rows - 1) / tile_r;
    size_t num_tiles_r  = last_tile_r - first_tile_r + 1;
    size_t first_tile_c = offset_c / tile_c;
    size_t last_tile_c  = (offset_c + num_cols - 1) / tile_c;
    size_t num_tiles_c  = last_tile_c - first_tile_c + 1;

    // simplify - every tile handled by individual block (especially helpful for row/col transpose)
    dim3 grid = dim3(num_tiles_r, num_tiles_c);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    // row based needs shared mem for coalesced read/write
    // col based can access directly? maybe also use shared mem to unify
    copy_to_send_buffer<<<grid, block, 0, stream>>>(input,
                                                    volume,
                                                    send_info,
                                                    stored_size_per_rank,
                                                    send_buffers_ptr,
                                                    row_major,
                                                    offset_r,
                                                    offset_c,
                                                    lld,
                                                    p_r,
                                                    p_c,
                                                    tile_r,
                                                    tile_c);
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
    send_buffers_ptr.destroy();
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  // all2all data
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(send_buffers[r].ptr(0),
                        send_info[r * stored_size_per_rank + BlockInfo::TOTAL_SIZE] * sizeof(VAL),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(recv_buffers[r].ptr(0),
                        recv_info[r * stored_size_per_rank + BlockInfo::TOTAL_SIZE] * sizeof(VAL),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  send_info.destroy();
  for (auto&& buf : send_buffers) {
    buf.destroy();
  }

  // combine data from all buffers
  Buffer<VAL> result_2dbc = create_buffer<VAL>(total_receive, Memory::GPU_FB_MEM);
  if (total_receive > 0) {
    Buffer<VAL*> recv_buffers_ptr = create_buffer<VAL*>(num_ranks, Memory::Z_COPY_MEM);
    for (size_t r = 0; r < num_ranks; r++) {
      recv_buffers_ptr[r] = recv_buffers[r].ptr(0);
    }

    size_t avr_elements_per_rank = (total_receive + num_ranks - 1) / num_ranks;
    // this roughly ensures ~32 elements per thread to copy - not optimized yet
    size_t num_blocks_per_rank =
      ((avr_elements_per_rank + 31) / 32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_shape = dim3(num_blocks_per_rank, num_ranks);
    merge_data_to_result<<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(result_2dbc,
                                                                       total_receive,
                                                                       recv_info,
                                                                       stored_size_per_rank,
                                                                       recv_buffers_ptr,
                                                                       target_lld,
                                                                       tile_r,
                                                                       tile_c,
                                                                       (size_t)nccl_rank,
                                                                       num_ranks);
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
    recv_buffers_ptr.destroy();
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  recv_info.destroy();
  for (auto&& buf : recv_buffers) {
    buf.destroy();
  }

  // returns the buffer/size/lld
  return {result_2dbc, total_receive, target_lld};
}

template <typename VAL>
void repartition_matrix_block(
  Buffer<VAL> input_2dbc_buffer,
  size_t input_volume,
  size_t input_lld,
  size_t local_rank,  // NOTE: this needs to correspond to communicator rank!
  size_t p_r,
  size_t p_c,
  size_t tile_r,
  size_t tile_c,
  VAL* target,
  size_t target_volume,
  size_t target_lld,
  size_t num_target_rows,
  size_t num_target_cols,
  bool target_row_major,
  // TODO optimize -- we would like to provide a global mapping to skip additional communication
  size_t target_offset_r,
  size_t target_offset_c,
  comm::Communicator comm_wrapper)
{
  auto num_ranks = p_r * p_c;

  auto comm   = comm_wrapper.get<ncclComm_t*>();
  auto stream = get_cached_stream();

  size_t num_input_rows = input_volume > 0 ? input_lld : 0;
  size_t num_input_cols = input_volume > 0 ? input_volume / input_lld : 0;

  // will be computed from offset exchange
  size_t target_p_r       = 0;
  size_t target_p_c       = 0;
  size_t target_p_r_valid = 0;
  size_t target_p_c_valid = 0;

  // 1. communicate global offsets
  auto offsets_r = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);
  auto offsets_c = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);
  // for now we need to exchange all offsets
  {
    auto offsets                = create_buffer<size_t>(2 * num_ranks, Memory::Z_COPY_MEM);
    offsets[2 * local_rank]     = num_target_rows > 0 ? target_offset_r + num_target_rows : 0;
    offsets[2 * local_rank + 1] = num_target_cols > 0 ? target_offset_c + num_target_cols : 0;
    CHECK_NCCL(
      ncclAllGather(offsets.ptr(2 * local_rank), offsets.ptr(0), 2, ncclUint64, *comm, stream));
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

    // re-arrange so that all row offsets come first
    for (size_t i = 1; i < num_ranks; i += 2) {
      size_t tmp    = offsets[i];
      size_t idx2   = 2 * num_ranks - 1 - i;
      offsets[i]    = offsets[idx2];
      offsets[idx2] = tmp;
    }
    // sort col/row offsets independently
    std::sort(offsets.ptr(0), offsets.ptr(num_ranks));
    std::sort(offsets.ptr(num_ranks), offsets.ptr(2 * num_ranks));
    // store offsets (we know that we can skip duplicate information)

    size_t last_offset_r = 0;
    size_t empty_p_r     = 0;
    size_t equals_r      = 1;
    for (size_t r = 0; r < num_ranks; r++) {
      if (offsets[r] > last_offset_r) {
        offsets_r[target_p_r_valid++] = offsets[r];
        last_offset_r                 = offsets[r];
      } else if (target_p_r_valid == 1) {
        assert(offsets[r] == last_offset_r);
        equals_r++;
      } else if (target_p_r_valid == 0) {
        empty_p_r++;
      }
    }
    size_t last_offset_c = 0;
    size_t empty_p_c     = 0;
    size_t equals_c      = 1;
    for (size_t c = num_ranks; c < 2 * num_ranks; c++) {
      if (offsets[c] > last_offset_c) {
        offsets_c[target_p_c_valid++] = offsets[c];
        last_offset_c                 = offsets[c];
      } else if (target_p_c_valid == 1) {
        assert(offsets[c] == last_offset_c);
        equals_c++;
      } else if (target_p_c_valid == 0) {
        empty_p_c++;
      }
    }

    // edge-case -- empty in 2D
    // x x x x 0 0
    // x x x x 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // target_p_r_valid = 2 target_p_c_valid = 4
    // empty_p_r = 18 empty_p_c = 10
    // equals_r = 4 equals_c = 2
    if (empty_p_r > 0 && empty_p_c > 0) {
      size_t empty_prod = empty_p_r * empty_p_c;
      assert(empty_prod % num_ranks == 0);
      empty_prod /= num_ranks;
      bool found_match = false;
      for (size_t r = 1; r <= empty_p_r && !found_match; r++) {
        for (size_t c = 1; r <= empty_p_c; r++) {
          if (r * c == empty_prod && (r + target_p_r_valid) * (c + target_p_c_valid) == num_ranks) {
            found_match = true;
            empty_p_r   = r;
            empty_p_c   = c;
            break;
          }
        }
      }
      assert(found_match);
    }

    target_p_r = target_p_r_valid + empty_p_r;
    target_p_c = target_p_c_valid + empty_p_c;

    // update offsets for invalid ranks
    for (int r = target_p_r_valid; r < target_p_r; ++r) {
      offsets_r[r] = offsets_r[r - 1];
    }
    for (int c = target_p_c_valid; c < target_p_c; ++c) {
      offsets_c[c] = offsets_c[c - 1];
    }

    offsets.destroy();
    assert(num_ranks == target_p_r * target_p_c);
  }

  // Assumptions:
  // a. local_rank == nccl_rank == 2dbc-id (col-major)
  // b. local_rank interpreted row-major (cuPyNumeric) should match offsets in offset mappings
  // c. offsets for ranks outside valid bounds are not considered
  size_t rank_r_rm = local_rank / target_p_c;
  size_t rank_c_rm = local_rank % target_p_c;
  size_t rank_r_cm = local_rank % p_r;
  size_t rank_c_cm = local_rank / p_r;
  {
    assert(rank_r_rm >= target_p_r_valid ||
           offsets_r[rank_r_rm] == target_offset_r + num_target_rows);
    assert(rank_c_rm >= target_p_c_valid ||
           offsets_c[rank_c_rm] == target_offset_c + num_target_cols);
  }

  // 2. compute expected send/receive sizes locally
  // first convert global element offsets to local tile offsets
  auto glob2loc =
    [](size_t glob_elem, size_t first_tile_offset, size_t proc_dim, size_t tilesize) -> size_t {
    size_t local_element = 0;
    if (glob_elem > first_tile_offset) {
      size_t remainder = glob_elem - first_tile_offset;
      // full cycles
      size_t cycle_length = proc_dim * tilesize;
      size_t full_cycles  = remainder / cycle_length;
      local_element += tilesize * full_cycles;
      remainder = remainder % cycle_length;
      local_element += min(remainder, tilesize);
    }

    return local_element;
  };

  size_t first_tile_offset_r  = rank_r_cm * tile_r;
  size_t first_tile_offset_c  = rank_c_cm * tile_c;
  size_t stored_size_per_rank = get_16b_aligned_count(BlockInfo::LAST, sizeof(size_t));
  size_t total_send_elements  = 0;
  size_t total_recv_elements  = 0;
  Buffer<size_t> send_info =
    create_buffer<size_t>(num_ranks * stored_size_per_rank, Memory::Z_COPY_MEM);
  Buffer<size_t> recv_info =
    create_buffer<size_t>(num_ranks * stored_size_per_rank, Memory::Z_COPY_MEM);

  // send/recv buffer

  std::vector<Buffer<VAL>> send_buffers;
  send_buffers.reserve(num_ranks);
  std::vector<Buffer<VAL>> recv_buffers;
  recv_buffers.reserve(num_ranks);

  size_t active_send_row_end = 0;
  for (size_t rank_r = 0; rank_r < target_p_r; ++rank_r) {
    size_t active_send_row_start = active_send_row_end;
    active_send_row_end          = glob2loc(offsets_r[rank_r], first_tile_offset_r, p_r, tile_r);
    active_send_row_end = std::min(active_send_row_end, num_input_rows);  // limited by local rows!
    size_t active_send_column_end = 0;
    for (size_t rank_c = 0; rank_c < target_p_c; ++rank_c) {
      size_t active_send_column_start = active_send_column_end;
      auto other_rank        = rank_r * target_p_c + rank_c;  // target ranks are row major!!!
      active_send_column_end = glob2loc(offsets_c[rank_c], first_tile_offset_c, p_c, tile_c);
      active_send_column_end =
        std::min(active_send_column_end, num_input_cols);  // limited by local cols!

      // send information from local_rank to other_rank
      {
        size_t active_send_rows     = active_send_row_end - active_send_row_start;
        size_t active_send_columns  = active_send_column_end - active_send_column_start;
        auto send_elements_for_rank = active_send_columns * active_send_rows;
        total_send_elements += send_elements_for_rank;

        send_info[other_rank * stored_size_per_rank + BlockInfo::TOTAL_SIZE] =
          send_elements_for_rank;
        send_info[other_rank * stored_size_per_rank + BlockInfo::LLD] =
          active_send_rows;  // col-major send data
        send_info[other_rank * stored_size_per_rank + BlockInfo::OFFSET_ROW] =
          active_send_row_start;
        send_info[other_rank * stored_size_per_rank + BlockInfo::OFFSET_COL] =
          active_send_column_start;
        assert(send_buffers.size() == other_rank);
        send_buffers.emplace_back(create_buffer<VAL>(send_elements_for_rank, Memory::GPU_FB_MEM));
      }
    }
  }

  assert(total_send_elements == input_volume);

  // 3. package send data (should be blocks of data)
  if (total_send_elements > 0) {
    VAL* input_2dbc               = input_2dbc_buffer.ptr(0);
    Buffer<VAL*> send_buffers_ptr = create_buffer<VAL*>(num_ranks, Memory::Z_COPY_MEM);
    for (size_t r = 0; r < num_ranks; r++) {
      send_buffers_ptr[r] = send_buffers[r].ptr(0);
    }

    size_t avr_elements_per_rank = (total_send_elements + num_ranks - 1) / num_ranks;
    // this roughly ensures ~32 elements per thread to copy - not optimized yet
    size_t num_blocks_per_rank =
      ((avr_elements_per_rank + 31) / 32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_shape = dim3(num_blocks_per_rank, num_ranks);
    split_data_to_send_buffers<<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(input_2dbc,
                                                                             input_volume,
                                                                             input_lld,
                                                                             send_info,
                                                                             stored_size_per_rank,
                                                                             send_buffers_ptr,
                                                                             p_r,
                                                                             p_c,
                                                                             tile_r,
                                                                             tile_c);

    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
    send_buffers_ptr.destroy();
  }
  // we can destroy the input once we distributed data into the buffers
  input_2dbc_buffer.destroy();

  // compute and allocate receive buffers
  for (size_t rank_r = 0; rank_r < target_p_r; ++rank_r) {
    for (size_t rank_c = 0; rank_c < target_p_c; ++rank_c) {
      auto other_rank = rank_r * target_p_c + rank_c;  // target ranks are row major!!!

      // recv information from other_rank to local_rank
      // other rank sends info for tile based on col-major ordering
      size_t other_rank_r_cm           = other_rank % p_r;
      size_t other_rank_c_cm           = other_rank / p_r;
      size_t other_first_tile_offset_r = other_rank_r_cm * tile_r;
      size_t other_first_tile_offset_c = other_rank_c_cm * tile_c;

      // locate other active rows/cols w.r.t. local target offsets
      size_t active_recv_row_end =
        glob2loc(target_offset_r + num_target_rows, other_first_tile_offset_r, p_r, tile_r);
      size_t active_recv_column_end =
        glob2loc(target_offset_c + num_target_cols, other_first_tile_offset_c, p_c, tile_c);
      size_t active_recv_row_start =
        glob2loc(target_offset_r, other_first_tile_offset_r, p_r, tile_r);
      size_t active_recv_column_start =
        glob2loc(target_offset_c, other_first_tile_offset_c, p_c, tile_c);

      size_t active_recv_rows     = active_recv_row_end - active_recv_row_start;
      size_t active_recv_columns  = active_recv_column_end - active_recv_column_start;
      auto recv_elements_for_rank = active_recv_columns * active_recv_rows;
      total_recv_elements += recv_elements_for_rank;

      recv_info[other_rank * stored_size_per_rank + BlockInfo::TOTAL_SIZE] = recv_elements_for_rank;
      recv_info[other_rank * stored_size_per_rank + BlockInfo::LLD] =
        active_recv_rows;  // col-major recv data
      recv_info[other_rank * stored_size_per_rank + BlockInfo::OFFSET_ROW] = active_recv_row_start;
      recv_info[other_rank * stored_size_per_rank + BlockInfo::OFFSET_COL] =
        active_recv_column_start;
      assert(other_rank == recv_buffers.size());
      recv_buffers.emplace_back(create_buffer<VAL>(recv_elements_for_rank, Memory::GPU_FB_MEM));
    }
  }

  assert(total_recv_elements == target_volume);

  // 4. communicate data
  // all2all data
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(send_buffers[r].ptr(0),
                        send_info[r * stored_size_per_rank + BlockInfo::TOTAL_SIZE] * sizeof(VAL),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(recv_buffers[r].ptr(0),
                        recv_info[r * stored_size_per_rank + BlockInfo::TOTAL_SIZE] * sizeof(VAL),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  send_info.destroy();
  for (auto&& buf : send_buffers) {
    buf.destroy();
  }

  // 5. merge data from recv_buffers
  if (total_recv_elements > 0) {
    Buffer<VAL*> recv_buffers_ptr = create_buffer<VAL*>(num_ranks, Memory::Z_COPY_MEM);
    for (size_t r = 0; r < num_ranks; r++) {
      recv_buffers_ptr[r] = recv_buffers[r].ptr(0);
    }

    size_t first_tile_r = target_offset_r / tile_r;
    size_t last_tile_r  = (target_offset_r + num_target_rows - 1) / tile_r;
    size_t num_tiles_r  = last_tile_r - first_tile_r + 1;
    size_t first_tile_c = target_offset_c / tile_c;
    size_t last_tile_c  = (target_offset_c + num_target_cols - 1) / tile_c;
    size_t num_tiles_c  = last_tile_c - first_tile_c + 1;

    // simplify - every tile handled by individual block (especially helpful for row/col transpose)
    dim3 grid = dim3(num_tiles_r, num_tiles_c);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    assemble_tiles_to_block_result<<<grid, block, 0, stream>>>(target,
                                                               target_volume,
                                                               target_lld,
                                                               target_offset_r,
                                                               target_offset_c,
                                                               target_row_major,
                                                               recv_info,
                                                               stored_size_per_rank,
                                                               recv_buffers_ptr,
                                                               p_r,
                                                               p_c,
                                                               tile_r,
                                                               tile_c);
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
    recv_buffers_ptr.destroy();
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  // cleanup
  offsets_r.destroy();
  offsets_c.destroy();
  recv_info.destroy();
  for (auto&& buf : recv_buffers) {
    buf.destroy();
  }
}

/*
  BOOL         = LEGION_TYPE_BOOL,
  INT8         = LEGION_TYPE_INT8,
  INT16        = LEGION_TYPE_INT16,
  INT32        = LEGION_TYPE_INT32,
  INT64        = LEGION_TYPE_INT64,
  UINT8        = LEGION_TYPE_UINT8,
  UINT16       = LEGION_TYPE_UINT16,
  UINT32       = LEGION_TYPE_UINT32,
  UINT64       = LEGION_TYPE_UINT64,
  FLOAT16      = LEGION_TYPE_FLOAT16,
  FLOAT32      = LEGION_TYPE_FLOAT32,
  FLOAT64      = LEGION_TYPE_FLOAT64,
  COMPLEX64    = LEGION_TYPE_COMPLEX64,
  COMPLEX128   = LEGION_TYPE_COMPLEX128
  */
template std::tuple<Buffer<type_of<Type::Code::BOOL>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::BOOL>>(const type_of<Type::Code::BOOL>*,
                                                   size_t,
                                                   bool,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::BOOL>>(Buffer<type_of<Type::Code::BOOL>>,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  type_of<Type::Code::BOOL>*,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  bool,
                                                                  size_t,
                                                                  size_t,
                                                                  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::INT8>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::INT8>>(const type_of<Type::Code::INT8>*,
                                                   size_t,
                                                   bool,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   size_t,
                                                   comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::INT8>>(Buffer<type_of<Type::Code::INT8>>,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  type_of<Type::Code::INT8>*,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  bool,
                                                                  size_t,
                                                                  size_t,
                                                                  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::INT16>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::INT16>>(const type_of<Type::Code::INT16>*,
                                                    size_t,
                                                    bool,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::INT16>>(
  Buffer<type_of<Type::Code::INT16>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::INT16>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::INT32>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::INT32>>(const type_of<Type::Code::INT32>*,
                                                    size_t,
                                                    bool,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::INT32>>(
  Buffer<type_of<Type::Code::INT32>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::INT32>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::INT64>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::INT64>>(const type_of<Type::Code::INT64>*,
                                                    size_t,
                                                    bool,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::INT64>>(
  Buffer<type_of<Type::Code::INT64>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::INT64>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::UINT8>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::UINT8>>(const type_of<Type::Code::UINT8>*,
                                                    size_t,
                                                    bool,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::UINT8>>(
  Buffer<type_of<Type::Code::UINT8>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::UINT8>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::UINT16>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::UINT16>>(const type_of<Type::Code::UINT16>*,
                                                     size_t,
                                                     bool,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::UINT16>>(
  Buffer<type_of<Type::Code::UINT16>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::UINT16>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::UINT32>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::UINT32>>(const type_of<Type::Code::UINT32>*,
                                                     size_t,
                                                     bool,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::UINT32>>(
  Buffer<type_of<Type::Code::UINT32>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::UINT32>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::UINT64>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::UINT64>>(const type_of<Type::Code::UINT64>*,
                                                     size_t,
                                                     bool,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     size_t,
                                                     comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::UINT64>>(
  Buffer<type_of<Type::Code::UINT64>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::UINT64>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::FLOAT16>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::FLOAT16>>(const type_of<Type::Code::FLOAT16>*,
                                                      size_t,
                                                      bool,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::FLOAT16>>(
  Buffer<type_of<Type::Code::FLOAT16>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::FLOAT16>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::FLOAT32>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::FLOAT32>>(const type_of<Type::Code::FLOAT32>*,
                                                      size_t,
                                                      bool,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::FLOAT32>>(
  Buffer<type_of<Type::Code::FLOAT32>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::FLOAT32>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::FLOAT64>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::FLOAT64>>(const type_of<Type::Code::FLOAT64>*,
                                                      size_t,
                                                      bool,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      size_t,
                                                      comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::FLOAT64>>(
  Buffer<type_of<Type::Code::FLOAT64>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::FLOAT64>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::COMPLEX64>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::COMPLEX64>>(const type_of<Type::Code::COMPLEX64>*,
                                                        size_t,
                                                        bool,
                                                        size_t,
                                                        size_t,
                                                        size_t,
                                                        size_t,
                                                        size_t,
                                                        size_t,
                                                        size_t,
                                                        comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::COMPLEX64>>(
  Buffer<type_of<Type::Code::COMPLEX64>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::COMPLEX64>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);
template std::tuple<Buffer<type_of<Type::Code::COMPLEX128>>, size_t, size_t>
repartition_matrix_2dbc<type_of<Type::Code::COMPLEX128>>(const type_of<Type::Code::COMPLEX128>*,
                                                         size_t,
                                                         bool,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         comm::Communicator);
template void repartition_matrix_block<type_of<Type::Code::COMPLEX128>>(
  Buffer<type_of<Type::Code::COMPLEX128>>,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  size_t,
  type_of<Type::Code::COMPLEX128>*,
  size_t,
  size_t,
  size_t,
  size_t,
  bool,
  size_t,
  size_t,
  comm::Communicator);

}  // namespace cupynumeric