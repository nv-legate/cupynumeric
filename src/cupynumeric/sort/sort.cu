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

#include "cupynumeric/sort/sort.h"
#include "cupynumeric/sort/sort_template.inl"
#include "cupynumeric/sort/cub_sort.h"
#include "cupynumeric/sort/thrust_sort.h"
#include "cupynumeric/utilities/thrust_allocator.h"
#include "cupynumeric/utilities/thrust_util.h"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/thread/thread_search.cuh>

#include "cupynumeric/cuda_help.h"

// above this threshold segment sort will be performed
// by cub::DeviceSegmentedRadixSort instead of thrust::(stable_)sort
// with tuple keys (not available for complex)
#define SEGMENT_THRESHOLD_RADIX_SORT 400

namespace cupynumeric {

namespace {

template <typename T>
legate::Buffer<T> create_non_empty_buffer(
  size_t size, legate::Memory::Kind kind = legate::Memory::Kind::NO_MEMKIND)
{
  return legate::create_buffer<T>(std::max(size, size_t{1}), kind, alignof(T));
}

}  // namespace

template <Type::Code CODE>
struct support_cub : std::true_type {};
template <>
struct support_cub<Type::Code::COMPLEX64> : std::false_type {};
template <>
struct support_cub<Type::Code::COMPLEX128> : std::false_type {};

template <Type::Code CODE, std::enable_if_t<support_cub<CODE>::value>* = nullptr>
void local_sort(const type_of<CODE>* values_in,
                type_of<CODE>* values_out,
                const int64_t* indices_in,
                int64_t* indices_out,
                const size_t volume,
                const size_t sort_dim_size,
                const bool stable,  // cub sort is always stable
                cudaStream_t stream)
{
  using VAL = type_of<CODE>;
  // fallback to thrust approach as segmented radix sort is not suited for small segments
  if (volume == sort_dim_size || sort_dim_size > SEGMENT_THRESHOLD_RADIX_SORT) {
    cub_local_sort(values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stream);
  } else {
    thrust_local_sort(
      values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stable, stream);
  }
}

template <Type::Code CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void local_sort(const type_of<CODE>* values_in,
                type_of<CODE>* values_out,
                const int64_t* indices_in,
                int64_t* indices_out,
                const size_t volume,
                const size_t sort_dim_size,
                const bool stable,
                cudaStream_t stream)
{
  using VAL = type_of<CODE>;
  thrust_local_sort(
    values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stable, stream);
}

namespace {
// auto align to multiples of 16 bytes
auto get_16b_aligned = [](auto bytes) { return std::max<size_t>(16, (bytes + 15) / 16 * 16); };
auto get_16b_aligned_count = [](auto count, auto element_bytes) {
  return (get_16b_aligned(count * element_bytes) + element_bytes - 1) / element_bytes;
};
}  // namespace

// increase number of columns computed per block as long as either
// 1. we have more threads in block than elements in row
// OR
// 2. a) block still large enough to handle full row
//    AND
//    b)  We end up with too many blocks in y-direction otherwise
size_t compute_cols_per_block(size_t row_size, size_t col_size)
{
  size_t result = 1;
  while (result < THREADS_PER_BLOCK &&
         (row_size * result < THREADS_PER_BLOCK ||
          (row_size * result <= THREADS_PER_BLOCK * 16 && col_size / result > 256))) {
    result *= 2;
  }
  return result;
}

// create a launchconfig for 2d data copy kernels with coalesced rows
// the y direction identifies the row to compute
// in x-direction all threads are responsible for all columns of a single row
// the heuristic ensures that
// * every thread is assigned at leat 1 (except y-grid edge) and at most 32 elements
std::tuple<dim3, dim3> generate_launchconfig_for_2d_copy(size_t row_size, size_t col_size)
{
  int cols_per_block        = compute_cols_per_block(row_size, col_size);
  dim3 block_shape          = dim3(THREADS_PER_BLOCK / cols_per_block, cols_per_block);
  const size_t num_blocks_y = (col_size + block_shape.y - 1) / block_shape.y;
  const size_t num_blocks_x = ((row_size + 31) / 32 + block_shape.x - 1) / block_shape.x;
  dim3 grid_shape           = dim3(num_blocks_x, num_blocks_y);
  return std::make_tuple(grid_shape, block_shape);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_split_positions_segments(const VAL* data,
                                   const size_t segment_size_l,
                                   const SegmentSample<VAL>* samples,
                                   const size_t num_segments_l,
                                   const size_t num_samples_per_segment,
                                   const size_t num_usable_samples_per_segment,
                                   size_t* split_positions,
                                   const size_t num_splitters,
                                   const size_t my_sort_rank)
{
  const size_t splitter_idx_g = blockIdx.x * blockDim.x + threadIdx.x;
  if (splitter_idx_g >= num_splitters) {
    return;
  }

  const size_t num_splitters_per_segment = num_splitters / num_segments_l;
  const size_t splitter_pos              = splitter_idx_g % num_splitters_per_segment;
  const size_t splitter_segment          = splitter_idx_g / num_splitters_per_segment;

  const size_t index =
    (splitter_pos + 1) * num_usable_samples_per_segment / (num_splitters_per_segment + 1) - 1;
  const SegmentSample<VAL> splitter = samples[splitter_segment * num_samples_per_segment + index];

  // now perform search on data to receive position *after* last element to be
  // part of the package for my_sort_rank splitter_idx_g
  const size_t offset = splitter_segment * segment_size_l;
  if (my_sort_rank > splitter.rank) {
    // position of the last position with smaller value than splitter.value + 1
    split_positions[splitter_idx_g] =
      cub::LowerBound(data + offset, segment_size_l, splitter.value) + offset;
  } else if (my_sort_rank < splitter.rank) {
    // position of the first position with value larger than splitter.value
    split_positions[splitter_idx_g] =
      cub::UpperBound(data + offset, segment_size_l, splitter.value) + offset;
  } else {
    split_positions[splitter_idx_g] = splitter.position + 1;
  }
}

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  compute_send_dimensions(const size_t segment_size_l,
                          size_t* size_send,
                          const size_t num_segments_l,
                          const size_t num_segments_l_aligned,
                          const size_t* split_positions,
                          const size_t num_sort_ranks,
                          const size_t num_send_parts)
{
  const size_t send_part = blockIdx.x * blockDim.x + threadIdx.x;
  if (send_part >= num_send_parts) {
    return;
  }

  const size_t rank    = send_part / num_segments_l;
  const size_t segment = send_part % num_segments_l;

  size_t start_position = (rank > 0) ? split_positions[segment * (num_sort_ranks - 1) + rank - 1]
                                     : (segment_size_l * segment);
  size_t end_position   = (rank < num_sort_ranks - 1)
                            ? split_positions[segment * (num_sort_ranks - 1) + rank]
                            : ((segment + 1) * segment_size_l);
  size_t size           = end_position - start_position;
  size_send[rank * num_segments_l_aligned + segment] = size;
}

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp {
  // Running prefix
  size_t running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(size_t running_total) : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ size_t operator()(size_t block_aggregate)
  {
    size_t old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  compute_scan_per_rank(size_t* segment_blocks,
                        size_t* size_send,
                        size_t num_segments_l,
                        const size_t num_segments_l_aligned)
{
  assert(blockDim.x == THREADS_PER_BLOCK);

  // Specialize BlockScan for a 1D block of THREADS_PER_BLOCK threads on type size_t
  typedef cub::BlockScan<size_t, THREADS_PER_BLOCK> BlockScan;
  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // now we have 1 block per rank!
  const size_t rank     = blockIdx.x;
  const size_t threadId = threadIdx.x;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_segments_l; block_offset += THREADS_PER_BLOCK) {
    size_t thread_data = 0;
    // Load a segment of consecutive items that are blocked across threads
    if (block_offset + threadId < num_segments_l) {
      thread_data = size_send[rank * num_segments_l_aligned + block_offset + threadId];
    }

    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, prefix_op);
    __syncthreads();

    // Store scanned items to output segment
    if (block_offset + threadId < num_segments_l) {
      segment_blocks[rank * num_segments_l + block_offset + threadId] = thread_data;
    }
  }
  // also store sum of all in last element
  if (threadId == 0) {
    size_send[rank * num_segments_l_aligned + num_segments_l] = prefix_op(0);
  }
}

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  initialize_segment_start_positions(const size_t* start_positions,
                                     const size_t num_segments_l,
                                     size_t* segment_ids,
                                     const size_t num_segment_ids)
{
  const size_t segment_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (segment_idx >= num_segments_l) {
    return;
  }

  unsigned long long int* ptr = (unsigned long long int*)segment_ids;

  const size_t position = start_positions[segment_idx];
  if (position < num_segment_ids) {
    atomicAdd(&(ptr[position]), (unsigned long long int)1l);
  }
}

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_segment_sizes(const size_t* segments,
                        const size_t size,
                        int64_t* segments_diff,
                        const size_t num_segments_l,
                        const size_t segments_size_l)
{
  const size_t segment_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (segment_idx >= num_segments_l) {
    return;
  }

  if (num_segments_l == 1) {
    segments_diff[segment_idx] = size - segments_size_l;
  } else {
    const size_t position = cub::LowerBound(segments, size, segment_idx);
    const size_t next_position =
      cub::LowerBound(segments + position, size - position, segment_idx + 1) + position;

    const size_t segment_size  = next_position - position;
    segments_diff[segment_idx] = segment_size - segments_size_l;
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  copy_data_to_merge_buffers(const Buffer<size_t> segment_blocks_ptr,
                             const Buffer<size_t> size_send,
                             const Buffer<VAL> source_values,
                             Buffer<VAL*> target_values,
                             const size_t num_segments_l,
                             const size_t num_segments_l_aligned,
                             const size_t segment_size_l,
                             const size_t my_rank,
                             const size_t num_sort_ranks)
{
  const size_t thread_offset    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t threadgroup_size = blockDim.x * gridDim.x;
  const size_t segment_id       = blockIdx.y * blockDim.y + threadIdx.y;
  if (segment_id >= num_segments_l) {
    return;
  }

  size_t source_offset = segment_size_l * segment_id;
  for (int r = 0; r < num_sort_ranks; ++r) {
    size_t target_offset = segment_blocks_ptr[r * num_segments_l + segment_id];
    size_t local_size    = size_send[r * num_segments_l_aligned + segment_id];

    for (size_t pos = thread_offset; pos < local_size; pos += threadgroup_size) {
      target_values[r][target_offset + pos] = source_values[source_offset + pos];
    }
    source_offset += local_size;
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  combine_buffers_no_sort(const Buffer<VAL*> source_values,
                          const Buffer<size_t> target_offsets,
                          Buffer<VAL> target_values,
                          const size_t merged_size,
                          const size_t num_sort_ranks)
{
  const size_t thread_offset    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t threadgroup_size = blockDim.x * gridDim.x;
  const size_t rank_id          = blockIdx.y * blockDim.y + threadIdx.y;
  if (rank_id >= num_sort_ranks) {
    return;
  }

  size_t target_offset = target_offsets[rank_id];
  size_t local_size    = (rank_id == num_sort_ranks - 1)
                           ? (merged_size - target_offset)
                           : (target_offsets[rank_id + 1] - target_offset);

  for (size_t pos = thread_offset; pos < local_size; pos += threadgroup_size) {
    target_values[target_offset + pos] = source_values[rank_id][pos];
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  copy_data_to_rebalance_buffers(const Buffer<int64_t> segment_diff_pos,
                                 const Buffer<int64_t> send_left,
                                 const Buffer<int64_t> send_right,
                                 const Buffer<int64_t> send_left_pos,
                                 const Buffer<int64_t> send_right_pos,
                                 const Buffer<VAL> source_values,
                                 const size_t source_size,
                                 Buffer<VAL> target_left_values,
                                 Buffer<VAL> target_right_values,
                                 const size_t num_segments_l,
                                 const size_t segment_size_l,
                                 const size_t my_rank,
                                 const size_t num_sort_ranks)
{
  const size_t thread_offset    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t threadgroup_size = blockDim.x * gridDim.x;
  const size_t segment_id       = blockIdx.y * blockDim.y + threadIdx.y;
  if (segment_id >= num_segments_l) {
    return;
  }

  // copy left
  {
    int64_t send_left_size = send_left[segment_id];
    if (send_left_size > 0) {
      size_t source_start = segment_size_l * segment_id + segment_diff_pos[segment_id];
      size_t target_start = send_left_pos[segment_id];
      for (size_t pos = thread_offset; pos < send_left_size; pos += threadgroup_size) {
        target_left_values[target_start + pos] = source_values[source_start + pos];
      }
    }
  }
  // copy right
  {
    int64_t send_right_size = send_right[segment_id];
    if (send_right_size > 0) {
      size_t source_end   = (segment_id < num_segments_l - 1)
                              ? (segment_size_l * (segment_id + 1) + segment_diff_pos[segment_id + 1])
                              : source_size;
      size_t source_start = source_end - send_right_size;
      size_t target_start = send_right_pos[segment_id];
      for (size_t pos = thread_offset; pos < send_right_size; pos += threadgroup_size) {
        target_right_values[target_start + pos] = source_values[source_start + pos];
      }
    }
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  merge_rebalanced_buffers(const Buffer<int64_t> segment_diff_pos,
                           const Buffer<int64_t> send_left,
                           const Buffer<int64_t> send_right,
                           const Buffer<int64_t> recv_left_pos,
                           const Buffer<int64_t> recv_right_pos,
                           const Buffer<VAL> source_values,
                           const size_t source_size,
                           const Buffer<VAL> recv_left_values,
                           const Buffer<VAL> recv_right_values,
                           VAL* target_values,
                           const size_t num_segments_l,
                           const size_t segment_size_l,
                           const size_t my_rank,
                           const size_t num_sort_ranks)
{
  const size_t thread_offset    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t threadgroup_size = blockDim.x * gridDim.x;
  const size_t segment_id       = blockIdx.y * blockDim.y + threadIdx.y;
  if (segment_id >= num_segments_l) {
    return;
  }

  size_t target_offset = segment_id * segment_size_l;
  size_t source_start  = segment_size_l * segment_id + segment_diff_pos[segment_id];
  size_t source_end    = (segment_id < num_segments_l - 1)
                           ? (segment_size_l * (segment_id + 1) + segment_diff_pos[segment_id + 1])
                           : source_size;

  int64_t recv_left_size  = send_left[segment_id] * -1;
  int64_t recv_right_size = send_right[segment_id] * -1;

  if (recv_left_size < 0) {
    source_start -= recv_left_size;
  }
  if (recv_right_size < 0) {
    source_end += recv_right_size;
  }

  // copy from left
  {
    if (recv_left_size > 0) {
      size_t recv_left_start = (recv_left_pos[segment_id] * -1);
      for (size_t pos = thread_offset; pos < recv_left_size; pos += threadgroup_size) {
        target_values[target_offset + pos] = recv_left_values[recv_left_start + pos];
      }
      target_offset += recv_left_size;
    }
  }

  // copy main part
  {
    int64_t size = source_end - source_start;
    if (size > 0) {
      for (size_t pos = thread_offset; pos < size; pos += threadgroup_size) {
        target_values[target_offset + pos] = source_values[source_start + pos];
      }
      target_offset += size;
    }
  }

  // copy from right
  {
    if (recv_right_size > 0) {
      size_t recv_right_start = (recv_right_pos[segment_id] * -1);
      for (size_t pos = thread_offset; pos < recv_right_size; pos += threadgroup_size) {
        target_values[target_offset + pos] = recv_right_values[recv_right_start + pos];
      }
      target_offset += recv_right_size;
    }
  }

#ifdef DEBUG_CUPYNUMERIC
  assert(target_offset == (segment_id + 1) * segment_size_l);
#endif
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_samples_segment(const VAL* data,
                          const size_t num_segments_l,
                          SegmentSample<VAL>* samples,
                          const size_t num_samples_per_segment_l,
                          const size_t segment_size_l,
                          const size_t offset,
                          const size_t num_sort_ranks,
                          const size_t sort_rank)
{
  const size_t sample_idx    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_samples_l = num_samples_per_segment_l * num_segments_l;
  if (sample_idx >= num_samples_l) {
    return;
  }

  const size_t segment_id_l       = sample_idx / num_samples_per_segment_l;
  const size_t segment_sample_idx = sample_idx % num_samples_per_segment_l;
  const size_t sample_index       = offset + sample_idx;

  if (num_samples_per_segment_l < segment_size_l) {
    const size_t index = segment_id_l * segment_size_l +
                         (segment_sample_idx + 1) * segment_size_l / num_samples_per_segment_l - 1;
    samples[sample_index].value    = data[index];
    samples[sample_index].rank     = sort_rank;
    samples[sample_index].segment  = segment_id_l;
    samples[sample_index].position = index;
  } else {
    // edge case where num_samples_l > volume
    if (segment_sample_idx < segment_size_l) {
      const size_t index             = segment_id_l * segment_size_l + segment_sample_idx;
      samples[sample_index].value    = data[index];
      samples[sample_index].rank     = sort_rank;
      samples[sample_index].segment  = segment_id_l;
      samples[sample_index].position = index;
    } else {
      samples[sample_index].rank    = -1;  // not populated
      samples[sample_index].segment = segment_id_l;
    }
  }
}

// transpose from CUDA SDK
#define BLOCK_DIM 16
__global__ void transpose(int64_t* odata, int64_t* idata, int width, int height)
{
  __shared__ int64_t block[BLOCK_DIM][BLOCK_DIM + 1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  if ((xIndex < width) && (yIndex < height)) {
    unsigned int index_in           = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
  if ((xIndex < height) && (yIndex < width)) {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out]       = block[threadIdx.x][threadIdx.y];
  }
}

struct subtract : public thrust::unary_function<int64_t, int64_t> {
  const int64_t constant_;

  subtract(int64_t constant) : constant_(constant) {}

  __CUDA_HD__ int64_t operator()(const int64_t& input) const { return input - constant_; }
};

struct positive_value : public thrust::unary_function<int64_t, int64_t> {
  __CUDA_HD__ int64_t operator()(const int64_t& x) const { return x > 0 ? x : 0; }
};

struct negative_value : public thrust::unary_function<int64_t, int64_t> {
  __CUDA_HD__ int64_t operator()(const int64_t& x) const { return x < 0 ? -x : 0; }
};

struct positive_plus : public thrust::binary_function<int64_t, int64_t, int64_t> {
  __CUDA_HD__ int64_t operator()(const int64_t& lhs, const int64_t& rhs) const
  {
    return lhs > 0 ? (lhs + (rhs > 0 ? rhs : 0)) : (rhs > 0 ? rhs : 0);
  }
};

struct negative_plus : public thrust::binary_function<int64_t, int64_t, int64_t> {
  __CUDA_HD__ int64_t operator()(const int64_t& lhs, const int64_t& rhs) const
  {
    return (lhs < 0 ? (lhs + (rhs < 0 ? rhs : 0)) : (rhs < 0 ? rhs : 0));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <Type::Code CODE>
SegmentMergePiece<type_of<CODE>> merge_all_buffers(
  std::vector<SegmentMergePiece<type_of<CODE>>>& merge_buffers,
  bool segmented,
  bool argsort,
  ThrustAllocator& alloc,
  cudaStream_t stream)
{
  using VAL = type_of<CODE>;

  // fallback to full sort for 1D and > 64 parts
  if (!segmented && merge_buffers.size() > 64) {
    SegmentMergePiece<VAL> result;

    // initialize target
    size_t merged_size    = 0;
    size_t num_sort_ranks = merge_buffers.size();
    Buffer<size_t> target_offsets =
      create_non_empty_buffer<size_t>(num_sort_ranks, legate::Memory::Z_COPY_MEM);

    // loop comparably small -> no init kernel
    for (size_t i = 0; i < num_sort_ranks; ++i) {
      target_offsets[i] = merged_size;
      merged_size += merge_buffers[i].size;
    }
    result.values   = create_non_empty_buffer<VAL>(merged_size);
    result.indices  = create_non_empty_buffer<int64_t>(argsort ? merged_size : 0);
    result.segments = create_non_empty_buffer<size_t>(segmented ? merged_size : 0);
    result.size     = merged_size;

    // copy data into result
    {
      Buffer<VAL*> val_buffers_ptr =
        create_non_empty_buffer<VAL*>(num_sort_ranks, legate::Memory::Z_COPY_MEM);
      for (size_t r = 0; r < num_sort_ranks; r++) {
        val_buffers_ptr[r] = merge_buffers[r].values.ptr(0);
      }

      auto elements_per_rank = std::max<size_t>(merged_size / num_sort_ranks, 1);
      auto [grid_shape, block_shape] =
        generate_launchconfig_for_2d_copy(elements_per_rank, num_sort_ranks);

      combine_buffers_no_sort<<<grid_shape, block_shape, 0, stream>>>(
        val_buffers_ptr, target_offsets, result.values, merged_size, num_sort_ranks);
      if (argsort) {
        Buffer<int64_t*> idc_buffers_ptr =
          create_non_empty_buffer<int64_t*>(num_sort_ranks, legate::Memory::Z_COPY_MEM);
        for (size_t r = 0; r < num_sort_ranks; r++) {
          idc_buffers_ptr[r] = merge_buffers[r].indices.ptr(0);
        }
        combine_buffers_no_sort<<<grid_shape, block_shape, 0, stream>>>(
          idc_buffers_ptr, target_offsets, result.indices, merged_size, num_sort_ranks);

        CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));  // needed before Z-copy destroy()
        idc_buffers_ptr.destroy();
      } else {
        CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));  // needed before Z-copy destroy()
      }
      val_buffers_ptr.destroy();
      target_offsets.destroy();

      // destroy buffers
      for (size_t i = 0; i < num_sort_ranks; ++i) {
        SegmentMergePiece<VAL> piece = merge_buffers[i];
        piece.values.destroy();
        if (argsort) {
          piece.indices.destroy();
        }
      }
      merge_buffers.clear();
    }

    // sort data (locally)
    auto p_values  = result.values.ptr(0);
    auto p_indices = argsort ? result.indices.ptr(0) : nullptr;
    local_sort<CODE>(
      p_values, p_values, p_indices, p_indices, merged_size, merged_size, true, stream);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
    return result;
  } else {
    // maybe k-way merge is more efficient here...
    auto exec_policy      = DEFAULT_POLICY(alloc).on(stream);
    size_t num_sort_ranks = merge_buffers.size();
    std::vector<SegmentMergePiece<VAL>> destroy_queue;
    for (size_t stride = 1; stride < num_sort_ranks; stride *= 2) {
      for (size_t pos = 0; pos + stride < num_sort_ranks; pos += 2 * stride) {
        SegmentMergePiece<VAL> source1 = merge_buffers[pos];
        SegmentMergePiece<VAL> source2 = merge_buffers[pos + stride];
        auto merged_size               = source1.size + source2.size;
        auto merged_values             = create_non_empty_buffer<VAL>(merged_size);
        auto merged_indices  = create_non_empty_buffer<int64_t>(argsort ? merged_size : 0);
        auto merged_segments = create_non_empty_buffer<size_t>(segmented ? merged_size : 0);
        auto p_merged_values = merged_values.ptr(0);
        auto p_values1       = source1.values.ptr(0);
        auto p_values2       = source2.values.ptr(0);

        if (segmented) {
          auto p_merged_segments = merged_segments.ptr(0);
          auto p_segments1       = source1.segments.ptr(0);
          auto p_segments2       = source2.segments.ptr(0);
          auto comb_keys_1 = thrust::make_zip_iterator(thrust::make_tuple(p_segments1, p_values1));
          auto comb_keys_2 = thrust::make_zip_iterator(thrust::make_tuple(p_segments2, p_values2));
          auto comb_keys_merged =
            thrust::make_zip_iterator(thrust::make_tuple(p_merged_segments, p_merged_values));

          if (argsort) {
            // merge with key/value
            auto p_indices1       = source1.indices.ptr(0);
            auto p_indices2       = source2.indices.ptr(0);
            auto p_merged_indices = merged_indices.ptr(0);
            thrust::merge_by_key(exec_policy,
                                 comb_keys_1,
                                 comb_keys_1 + source1.size,
                                 comb_keys_2,
                                 comb_keys_2 + source2.size,
                                 p_indices1,
                                 p_indices2,
                                 comb_keys_merged,
                                 p_merged_indices,
                                 thrust::less<thrust::tuple<size_t, VAL>>());
          } else {
            thrust::merge(exec_policy,
                          comb_keys_1,
                          comb_keys_1 + source1.size,
                          comb_keys_2,
                          comb_keys_2 + source2.size,
                          comb_keys_merged,
                          thrust::less<thrust::tuple<size_t, VAL>>());
          }
        } else {
          if (argsort) {
            // merge with key/value
            auto p_indices1       = source1.indices.ptr(0);
            auto p_indices2       = source2.indices.ptr(0);
            auto p_merged_indices = merged_indices.ptr(0);
            thrust::merge_by_key(exec_policy,
                                 p_values1,
                                 p_values1 + source1.size,
                                 p_values2,
                                 p_values2 + source2.size,
                                 p_indices1,
                                 p_indices2,
                                 p_merged_values,
                                 p_merged_indices);
          } else {
            thrust::merge(exec_policy,
                          p_values1,
                          p_values1 + source1.size,
                          p_values2,
                          p_values2 + source2.size,
                          p_merged_values);
          }
        }

        destroy_queue.push_back(source1);
        destroy_queue.push_back(source2);

        merge_buffers[pos].values   = merged_values;
        merge_buffers[pos].indices  = merged_indices;
        merge_buffers[pos].segments = merged_segments;
        merge_buffers[pos].size     = merged_size;
      }

      // destroy buffers only after each sweep
      for (size_t i = 0; i < destroy_queue.size(); ++i) {
        SegmentMergePiece<VAL> piece = destroy_queue[i];
        piece.values.destroy();
        if (segmented) {
          piece.segments.destroy();
        }
        if (argsort) {
          piece.indices.destroy();
        }
      }
      destroy_queue.clear();
    }
    SegmentMergePiece<VAL> result = merge_buffers[0];
    merge_buffers.clear();
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
    return result;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VAL>
void rebalance_data(SegmentMergePiece<VAL>& merge_buffer,
                    void* output_ptr,
                    /* global domain information */
                    size_t my_rank,  // global NCCL rank
                    /* domain information in sort dimension */
                    size_t my_sort_rank,    // local rank id in sort dimension
                    size_t num_sort_ranks,  // #ranks that share a sort dimension
                    size_t* sort_ranks,     // rank ids that share a sort dimension with us
                    size_t segment_size_l,  // (local) segment size
                    size_t num_segments_l,
                    /* other */
                    bool argsort,
                    ThrustAllocator& alloc,
                    cudaStream_t stream,
                    ncclComm_t* comm)
{
  // output is either values or indices
  VAL* output_values      = nullptr;
  int64_t* output_indices = nullptr;
  if (argsort) {
    output_indices = static_cast<int64_t*>(output_ptr);
  } else {
    output_values = static_cast<VAL*>(output_ptr);
  }

  auto exec_policy = DEFAULT_POLICY(alloc).on(stream);

  {
    // compute diff for each segment
    const size_t num_segments_l_aligned = get_16b_aligned_count(num_segments_l, sizeof(size_t));
    auto segment_diff =
      create_non_empty_buffer<int64_t>(num_segments_l_aligned, legate::Memory::GPU_FB_MEM);
    {
      // start kernel to search from merge_buffer.segments
      const size_t num_blocks = (num_segments_l + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      extract_segment_sizes<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        merge_buffer.segments.ptr(0),
        merge_buffer.size,
        segment_diff.ptr(0),
        num_segments_l,
        segment_size_l);
    }

    merge_buffer.segments.destroy();
    if (argsort) {
      merge_buffer.values.destroy();
    }

#ifdef DEBUG_CUPYNUMERIC
    {
      size_t reduce =
        thrust::reduce(exec_policy, segment_diff.ptr(0), segment_diff.ptr(0) + num_segments_l, 0);
      size_t volume = segment_size_l * num_segments_l;
      assert(merge_buffer.size - volume == reduce);
    }
#endif

    // allocate target
    Buffer<int64_t> segment_diff_buffers = create_non_empty_buffer<int64_t>(
      num_segments_l_aligned * num_sort_ranks, legate::Memory::GPU_FB_MEM);

    // communicate segment diffs
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_sort_ranks; r++) {
      CHECK_NCCL(ncclSend(
        segment_diff.ptr(0), num_segments_l_aligned, ncclInt64, sort_ranks[r], *comm, stream));
      CHECK_NCCL(ncclRecv(segment_diff_buffers.ptr(r * num_segments_l_aligned),
                          num_segments_l_aligned,
                          ncclInt64,
                          sort_ranks[r],
                          *comm,
                          stream));
    }
    CHECK_NCCL(ncclGroupEnd());

    // copy to transpose structure [segments][ranks]
    auto segment_diff_2d = create_non_empty_buffer<int64_t>(num_segments_l_aligned * num_sort_ranks,
                                                            legate::Memory::GPU_FB_MEM);

    // Transpose
    {
      dim3 grid((num_segments_l_aligned + BLOCK_DIM - 1) / BLOCK_DIM,
                (num_sort_ranks + BLOCK_DIM - 1) / BLOCK_DIM);
      dim3 block(BLOCK_DIM, BLOCK_DIM);
      transpose<<<grid, block, 0, stream>>>(segment_diff_2d.ptr(0),
                                            segment_diff_buffers.ptr(0),
                                            num_segments_l_aligned,
                                            num_sort_ranks);
    }

#ifdef DEBUG_CUPYNUMERIC
    {
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        assert(0 == thrust::reduce(exec_policy,
                                   segment_diff_2d.ptr(segment * num_sort_ranks),
                                   segment_diff_2d.ptr(segment * num_sort_ranks) + num_sort_ranks,
                                   0));
      }
    }
#endif
    segment_diff_buffers.destroy();

    // 2d data [segments][ranks]
    /*
          -2    2    1    1    -3     2    -1
          -2    0    1    2    -1     1     0  (inclusive scan)
          neg --> receive from right
          pos --> send right

          0    2    0    -1   -2    1    -1    (incl.scan right)
          neg --> receive from left
          pos --> send left

          edge case --> send more than whole line should not happen due to sample choice!
    */
    // 2 (signed) arrays - left/right for every segment
    auto send_left  = create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);
    auto send_right = create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);

    // compute data to send....
    auto segment_diff_2d_scan =
      create_non_empty_buffer<int64_t>(num_segments_l * num_sort_ranks, legate::Memory::GPU_FB_MEM);
    thrust::device_ptr<int64_t> segment_diff_2d_ptr(segment_diff_2d.ptr(0));
    thrust::device_ptr<int64_t> segment_diff_2d_scan_ptr(segment_diff_2d_scan.ptr(0));
    thrust::inclusive_scan(exec_policy,
                           segment_diff_2d_ptr,
                           segment_diff_2d_ptr + num_segments_l * num_sort_ranks,
                           segment_diff_2d_scan_ptr);
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpy2DAsync(send_right.ptr(0),
                                             sizeof(int64_t),
                                             segment_diff_2d_scan.ptr(0) + my_sort_rank,
                                             num_sort_ranks * sizeof(int64_t),
                                             sizeof(int64_t),
                                             num_segments_l,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
    thrust::reverse_iterator<thrust::device_vector<int64_t>::iterator> iter_in(
      segment_diff_2d_ptr + num_segments_l * num_sort_ranks);
    thrust::reverse_iterator<thrust::device_vector<int64_t>::iterator> iter_out(
      segment_diff_2d_scan_ptr + num_segments_l * num_sort_ranks);
    thrust::inclusive_scan(
      exec_policy, iter_in, iter_in + num_segments_l * num_sort_ranks, iter_out);
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpy2DAsync(send_left.ptr(0),
                                             sizeof(int64_t),
                                             segment_diff_2d_scan.ptr(0) + my_sort_rank,
                                             num_sort_ranks * sizeof(int64_t),
                                             sizeof(int64_t),
                                             num_segments_l,
                                             cudaMemcpyDeviceToDevice,
                                             stream));

    segment_diff_2d.destroy();
    segment_diff_2d_scan.destroy();

    // package data to send
    size_t send_left_size  = thrust::transform_reduce(exec_policy,
                                                     send_left.ptr(0),
                                                     send_left.ptr(0) + num_segments_l,
                                                     positive_value(),
                                                     0,
                                                     thrust::plus<int64_t>());
    size_t recv_left_size  = thrust::transform_reduce(exec_policy,
                                                     send_left.ptr(0),
                                                     send_left.ptr(0) + num_segments_l,
                                                     negative_value(),
                                                     0,
                                                     thrust::plus<int64_t>());
    size_t send_right_size = thrust::transform_reduce(exec_policy,
                                                      send_right.ptr(0),
                                                      send_right.ptr(0) + num_segments_l,
                                                      positive_value(),
                                                      0,
                                                      thrust::plus<int64_t>());
    size_t recv_right_size = thrust::transform_reduce(exec_policy,
                                                      send_right.ptr(0),
                                                      send_right.ptr(0) + num_segments_l,
                                                      negative_value(),
                                                      0,
                                                      thrust::plus<int64_t>());
    SortPiece<VAL> send_left_data, recv_left_data, send_right_data, recv_right_data;
    send_left_data.size  = send_left_size;
    recv_left_data.size  = recv_left_size;
    send_right_data.size = send_right_size;
    recv_right_data.size = recv_right_size;
    if (argsort) {
      send_left_data.indices =
        create_non_empty_buffer<int64_t>(send_left_size, legate::Memory::GPU_FB_MEM);
      recv_left_data.indices =
        create_non_empty_buffer<int64_t>(recv_left_size, legate::Memory::GPU_FB_MEM);
      send_right_data.indices =
        create_non_empty_buffer<int64_t>(send_right_size, legate::Memory::GPU_FB_MEM);
      recv_right_data.indices =
        create_non_empty_buffer<int64_t>(recv_right_size, legate::Memory::GPU_FB_MEM);
    } else {
      send_left_data.values =
        create_non_empty_buffer<VAL>(send_left_size, legate::Memory::GPU_FB_MEM);
      recv_left_data.values =
        create_non_empty_buffer<VAL>(recv_left_size, legate::Memory::GPU_FB_MEM);
      send_right_data.values =
        create_non_empty_buffer<VAL>(send_right_size, legate::Memory::GPU_FB_MEM);
      recv_right_data.values =
        create_non_empty_buffer<VAL>(recv_right_size, legate::Memory::GPU_FB_MEM);
    }

    Buffer<int64_t> segment_diff_pos;
    {
      // need scan of segment_diff
      // need scan of (positive!) send_left, send_right
      segment_diff_pos =
        create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);
      auto send_left_pos =
        create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);
      auto send_right_pos =
        create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);
      {
        thrust::device_ptr<int64_t> segment_diff_ptr(segment_diff.ptr(0));
        thrust::device_ptr<int64_t> segment_diff_pos_ptr(segment_diff_pos.ptr(0));
        thrust::device_ptr<int64_t> send_left_ptr(send_left.ptr(0));
        thrust::device_ptr<int64_t> send_left_pos_ptr(send_left_pos.ptr(0));
        thrust::device_ptr<int64_t> send_right_ptr(send_right.ptr(0));
        thrust::device_ptr<int64_t> send_right_pos_ptr(send_right_pos.ptr(0));
        thrust::exclusive_scan(
          exec_policy, segment_diff_ptr, segment_diff_ptr + num_segments_l, segment_diff_pos_ptr);
        thrust::exclusive_scan(exec_policy,
                               send_left_ptr,
                               send_left_ptr + num_segments_l,
                               send_left_pos_ptr,
                               0,
                               positive_plus());
        thrust::exclusive_scan(exec_policy,
                               send_right_ptr,
                               send_right_ptr + num_segments_l,
                               send_right_pos_ptr,
                               0,
                               positive_plus());
      }

      auto [grid_shape, block_shape] =
        generate_launchconfig_for_2d_copy(segment_size_l, num_segments_l);

      if (argsort) {
        copy_data_to_rebalance_buffers<<<grid_shape, block_shape, 0, stream>>>(
          segment_diff_pos,
          send_left,
          send_right,
          send_left_pos,
          send_right_pos,
          merge_buffer.indices,
          merge_buffer.size,
          send_left_data.indices,
          send_right_data.indices,
          num_segments_l,
          segment_size_l,
          my_rank,
          num_sort_ranks);
      } else {
        copy_data_to_rebalance_buffers<<<grid_shape, block_shape, 0, stream>>>(
          segment_diff_pos,
          send_left,
          send_right,
          send_left_pos,
          send_right_pos,
          merge_buffer.values,
          merge_buffer.size,
          send_left_data.values,
          send_right_data.values,
          num_segments_l,
          segment_size_l,
          my_rank,
          num_sort_ranks);
      }

      send_left_pos.destroy();
      send_right_pos.destroy();
    }
    assert(send_left_data.size == send_left_size);
    assert(send_right_data.size == send_right_size);

    // send/receive overlapping data
    if (send_left_size + recv_left_size + send_right_size + recv_right_size > 0) {
      if (argsort) {
        CHECK_NCCL(ncclGroupStart());
        if (send_left_size > 0) {
          CHECK_NCCL(ncclSend(send_left_data.indices.ptr(0),
                              send_left_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank - 1],
                              *comm,
                              stream));
        }
        if (recv_left_size > 0) {
          CHECK_NCCL(ncclRecv(recv_left_data.indices.ptr(0),
                              recv_left_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank - 1],
                              *comm,
                              stream));
        }
        if (send_right_size > 0) {
          CHECK_NCCL(ncclSend(send_right_data.indices.ptr(0),
                              send_right_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank + 1],
                              *comm,
                              stream));
        }
        if (recv_right_size > 0) {
          CHECK_NCCL(ncclRecv(recv_right_data.indices.ptr(0),
                              recv_right_data.size,
                              ncclInt64,
                              sort_ranks[my_sort_rank + 1],
                              *comm,
                              stream));
        }
        CHECK_NCCL(ncclGroupEnd());
      } else {
        CHECK_NCCL(ncclGroupStart());
        if (send_left_size > 0) {
          CHECK_NCCL(ncclSend(send_left_data.values.ptr(0),
                              send_left_data.size * sizeof(VAL),
                              ncclInt8,
                              sort_ranks[my_sort_rank - 1],
                              *comm,
                              stream));
        }
        if (recv_left_size > 0) {
          CHECK_NCCL(ncclRecv(recv_left_data.values.ptr(0),
                              recv_left_data.size * sizeof(VAL),
                              ncclInt8,
                              sort_ranks[my_sort_rank - 1],
                              *comm,
                              stream));
        }
        if (send_right_size > 0) {
          CHECK_NCCL(ncclSend(send_right_data.values.ptr(0),
                              send_right_data.size * sizeof(VAL),
                              ncclInt8,
                              sort_ranks[my_sort_rank + 1],
                              *comm,
                              stream));
        }
        if (recv_right_size > 0) {
          CHECK_NCCL(ncclRecv(recv_right_data.values.ptr(0),
                              recv_right_data.size * sizeof(VAL),
                              ncclInt8,
                              sort_ranks[my_sort_rank + 1],
                              *comm,
                              stream));
        }
        CHECK_NCCL(ncclGroupEnd());
      }
    }

    if (argsort) {
      send_left_data.indices.destroy();
      send_right_data.indices.destroy();
    } else {
      send_left_data.values.destroy();
      send_right_data.values.destroy();
    }

    // merge data into target
    {
      // need scan of (negative!) send_left, send_right
      auto recv_left_pos =
        create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);
      auto recv_right_pos =
        create_non_empty_buffer<int64_t>(num_segments_l, legate::Memory::GPU_FB_MEM);
      {
        thrust::device_ptr<int64_t> recv_left_ptr(send_left.ptr(0));
        thrust::device_ptr<int64_t> recv_left_pos_ptr(recv_left_pos.ptr(0));
        thrust::device_ptr<int64_t> recv_right_ptr(send_right.ptr(0));
        thrust::device_ptr<int64_t> recv_right_pos_ptr(recv_right_pos.ptr(0));
        thrust::exclusive_scan(exec_policy,
                               recv_left_ptr,
                               recv_left_ptr + num_segments_l,
                               recv_left_pos_ptr,
                               0,
                               negative_plus());
        thrust::exclusive_scan(exec_policy,
                               recv_right_ptr,
                               recv_right_ptr + num_segments_l,
                               recv_right_pos_ptr,
                               0,
                               negative_plus());
      }

      auto [grid_shape, block_shape] =
        generate_launchconfig_for_2d_copy(segment_size_l, num_segments_l);

      if (argsort) {
        merge_rebalanced_buffers<<<grid_shape, block_shape, 0, stream>>>(segment_diff_pos,
                                                                         send_left,
                                                                         send_right,
                                                                         recv_left_pos,
                                                                         recv_right_pos,
                                                                         merge_buffer.indices,
                                                                         merge_buffer.size,
                                                                         recv_left_data.indices,
                                                                         recv_right_data.indices,
                                                                         output_indices,
                                                                         num_segments_l,
                                                                         segment_size_l,
                                                                         my_rank,
                                                                         num_sort_ranks);
      } else {
        merge_rebalanced_buffers<<<grid_shape, block_shape, 0, stream>>>(segment_diff_pos,
                                                                         send_left,
                                                                         send_right,
                                                                         recv_left_pos,
                                                                         recv_right_pos,
                                                                         merge_buffer.values,
                                                                         merge_buffer.size,
                                                                         recv_left_data.values,
                                                                         recv_right_data.values,
                                                                         output_values,
                                                                         num_segments_l,
                                                                         segment_size_l,
                                                                         my_rank,
                                                                         num_sort_ranks);
      }

      segment_diff_pos.destroy();
      recv_left_pos.destroy();
      recv_right_pos.destroy();
    }

    // remove segment_sizes, all buffers should be destroyed...
    segment_diff.destroy();
    send_left.destroy();
    send_right.destroy();
    if (argsort) {
      merge_buffer.indices.destroy();
      recv_left_data.indices.destroy();
      recv_right_data.indices.destroy();
    } else {
      merge_buffer.values.destroy();
      recv_left_data.values.destroy();
      recv_right_data.values.destroy();
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <Type::Code CODE>
void sample_sort_nccl_nd(
  TaskContext& context,
  SortPiece<type_of<CODE>> local_sorted,
  legate::PhysicalStore& output_array_unbound,  // only for unbound usage when !rebalance
  void* output_ptr,
  /* global domain information */
  size_t my_rank,  // global NCCL rank
  size_t num_ranks,
  size_t segment_size_g,
  /* domain information in sort dimension */
  size_t my_sort_rank,    // local rank id in sort dimension
  size_t num_sort_ranks,  // #ranks that share a sort dimension
  size_t* sort_ranks,     // rank ids that share a sort dimension with us
  size_t segment_size_l,  // (local) segment size
  /* other */
  bool rebalance,
  bool argsort,
  cudaStream_t stream,
  ncclComm_t* comm)
{
  using VAL = type_of<CODE>;

  size_t volume              = local_sorted.size;
  bool is_unbound_1d_storage = output_array_unbound.is_unbound_store();

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 0: detection of empty nodes
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // first of all we need to check for processes that don't want
  // to take part in the computation. This might lead to a reduction of
  // sort ranks. Note that if segment_size_l>0 && volume==0 means that we have
  // a full sort group being empty, this should not affect local sort rank size.
  {
    auto worker_count_d = create_non_empty_buffer<int32_t>(1, legate::Memory::GPU_FB_MEM);
    size_t worker_count = (segment_size_l > 0 ? 1 : 0);
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      worker_count_d.ptr(0), &worker_count, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    context.concurrent_task_barrier();
    CHECK_NCCL(ncclAllReduce(
      worker_count_d.ptr(0), worker_count_d.ptr(0), 1, ncclInt32, ncclSum, *comm, stream));
    context.concurrent_task_barrier();
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      &worker_count, worker_count_d.ptr(0), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (worker_count < num_ranks) {
      const size_t number_sort_groups = num_ranks / num_sort_ranks;
      num_sort_ranks                  = worker_count / number_sort_groups;
    }
    worker_count_d.destroy();

    // early out
    if (volume == 0) {
      if (is_unbound_1d_storage) {
        // we need to return an empty buffer here
        if (argsort) {
          auto buffer = create_non_empty_buffer<int64_t>(0, legate::Memory::GPU_FB_MEM);
          output_array_unbound.bind_data(buffer, Point<1>(0));
        } else {
          auto buffer = create_non_empty_buffer<VAL>(0, legate::Memory::GPU_FB_MEM);
          output_array_unbound.bind_data(buffer, Point<1>(0));
        }
      }
      return;
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 1: select and share samples accross sort domain
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // collect local samples - for now we take num_sort_ranks samples for every node/line
  // worst case this leads to imbalance of x2
  size_t num_segments_l            = volume / segment_size_l;
  size_t num_samples_per_segment_l = num_sort_ranks;
  size_t num_samples_l             = num_samples_per_segment_l * num_segments_l;
  size_t num_samples_per_segment_g = num_samples_per_segment_l * num_sort_ranks;
  size_t num_samples_g             = num_samples_per_segment_g * num_segments_l;
  auto samples =
    create_non_empty_buffer<SegmentSample<VAL>>(num_samples_g, legate::Memory::GPU_FB_MEM);

  size_t offset = num_samples_l * my_sort_rank;
  {
    const size_t num_blocks = (num_samples_l + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    extract_samples_segment<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      local_sorted.values.ptr(0),
      num_segments_l,
      samples.ptr(0),
      num_samples_per_segment_l,
      segment_size_l,
      offset,
      num_sort_ranks,
      my_sort_rank);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }

  // AllGather does not work here as not all have the same amount!
  // This is all2all restricted to one sort row
  {
    // allocate receive buffer
    const size_t aligned_count = get_16b_aligned_count(num_samples_l, sizeof(SegmentSample<VAL>));
    auto send_buffer =
      create_non_empty_buffer<SegmentSample<VAL>>(aligned_count, legate::Memory::GPU_FB_MEM);
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(send_buffer.ptr(0),
                                           samples.ptr(offset),
                                           sizeof(SegmentSample<VAL>) * num_samples_l,
                                           cudaMemcpyDeviceToDevice,
                                           stream));

    auto recv_buffer = create_non_empty_buffer<SegmentSample<VAL>>(aligned_count * num_sort_ranks,
                                                                   legate::Memory::GPU_FB_MEM);

    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_sort_ranks; r++) {
      if (r != my_sort_rank) {
        CHECK_NCCL(ncclSend(send_buffer.ptr(0),
                            aligned_count * sizeof(SegmentSample<VAL>),
                            ncclInt8,
                            sort_ranks[r],
                            *comm,
                            stream));
        CHECK_NCCL(ncclRecv(recv_buffer.ptr(aligned_count * r),
                            aligned_count * sizeof(SegmentSample<VAL>),
                            ncclInt8,
                            sort_ranks[r],
                            *comm,
                            stream));
      }
    }
    CHECK_NCCL(ncclGroupEnd());

    // copy back
    for (size_t r = 0; r < num_sort_ranks; r++) {
      if (r != my_sort_rank) {
        CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(samples.ptr(num_samples_l * r),
                                               recv_buffer.ptr(aligned_count * r),
                                               sizeof(SegmentSample<VAL>) * num_samples_l,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
      }
    }

    // destroy
    send_buffer.destroy();
    recv_buffer.destroy();

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 2: select splitters from samples and collect positions in local data
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // sort samples on device
  auto alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto exec_policy = DEFAULT_POLICY(alloc).on(stream);
  thrust::stable_sort(
    exec_policy, samples.ptr(0), samples.ptr(0) + num_samples_g, SegmentSampleComparator<VAL>());

  // check whether we have invalid samples (in case one participant did not have enough)
  SegmentSample<VAL> invalid_sample;
  invalid_sample.segment                = 0;
  invalid_sample.rank                   = -1;
  auto lower_bound                      = thrust::lower_bound(exec_policy,
                                         samples.ptr(0),
                                         samples.ptr(0) + num_samples_per_segment_g,
                                         invalid_sample,
                                         SegmentSampleComparator<VAL>());
  size_t num_usable_samples_per_segment = lower_bound - samples.ptr(0);

  // select splitters / positions based on samples (on device)
  // the indexing is split_positions[segments][positions]
  const size_t num_splitters = (num_sort_ranks - 1) * num_segments_l;
  auto split_positions = create_non_empty_buffer<size_t>(num_splitters, legate::Memory::GPU_FB_MEM);
  {
    const size_t num_blocks = (num_splitters + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    extract_split_positions_segments<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      local_sorted.values.ptr(0),
      segment_size_l,
      samples.ptr(0),
      num_segments_l,
      num_samples_per_segment_g,
      num_usable_samples_per_segment,
      split_positions.ptr(0),
      num_splitters,
      my_sort_rank);
  }

  // segment_blocks[r][segment]->position of data in segment for process r
  // perform blocksize wide scan on size_send[r][block*blocksize] within warp
  Buffer<size_t> segment_blocks =
    create_non_empty_buffer<size_t>(num_segments_l * num_sort_ranks, legate::Memory::GPU_FB_MEM);

  // initialize sizes to send
  const size_t num_segments_l_aligned = get_16b_aligned_count(num_segments_l + 1, sizeof(size_t));
  Buffer<size_t> size_send            = create_non_empty_buffer<size_t>(
    num_segments_l_aligned * num_sort_ranks, legate::Memory::GPU_FB_MEM);

  {
    const size_t num_send_parts = num_sort_ranks * num_segments_l;
    const size_t num_blocks     = (num_send_parts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_send_dimensions<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(segment_size_l,
                                                                          size_send.ptr(0),
                                                                          num_segments_l,
                                                                          num_segments_l_aligned,
                                                                          split_positions.ptr(0),
                                                                          num_sort_ranks,
                                                                          num_send_parts);

    compute_scan_per_rank<<<num_sort_ranks, THREADS_PER_BLOCK, 0, stream>>>(
      segment_blocks.ptr(0), size_send.ptr(0), num_segments_l, num_segments_l_aligned);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }

  // cleanup intermediate data structures
  samples.destroy();
  split_positions.destroy();

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 3: communicate data in sort domain
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // all2all exchange send/receive sizes
  Buffer<size_t> size_recv = create_non_empty_buffer<size_t>(
    num_segments_l_aligned * num_sort_ranks, legate::Memory::GPU_FB_MEM);
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_sort_ranks; r++) {
    CHECK_NCCL(ncclSend(size_send.ptr(r * num_segments_l_aligned),
                        num_segments_l_aligned,
                        ncclUint64,
                        sort_ranks[r],
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(size_recv.ptr(r * num_segments_l_aligned),
                        num_segments_l_aligned,
                        ncclUint64,
                        sort_ranks[r],
                        *comm,
                        stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  // we need the amount of data to transfer on the host --> get it
  Buffer<size_t> size_send_total =
    create_non_empty_buffer<size_t>(num_sort_ranks, legate::Memory::Z_COPY_MEM);
  Buffer<size_t> size_recv_total =
    create_non_empty_buffer<size_t>(num_sort_ranks, legate::Memory::Z_COPY_MEM);
  {
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpy2DAsync(size_send_total.ptr(0),
                                             1 * sizeof(size_t),
                                             size_send.ptr(num_segments_l),
                                             num_segments_l_aligned * sizeof(size_t),
                                             sizeof(int64_t),
                                             num_sort_ranks,
                                             cudaMemcpyDeviceToHost,
                                             stream));
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpy2DAsync(size_recv_total.ptr(0),
                                             1 * sizeof(size_t),
                                             size_recv.ptr(num_segments_l),
                                             num_segments_l_aligned * sizeof(size_t),
                                             sizeof(int64_t),
                                             num_sort_ranks,
                                             cudaMemcpyDeviceToHost,
                                             stream));

    // need to sync as we share values in between host/device
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // copy values into aligned send buffer
  std::vector<Buffer<VAL>> val_send_buffers(num_sort_ranks);
  std::vector<Buffer<int64_t>> idc_send_buffers(num_sort_ranks);
  {
    for (size_t r = 0; r < num_sort_ranks; r++) {
      val_send_buffers[r] =
        create_non_empty_buffer<VAL>(size_send_total[r], legate::Memory::GPU_FB_MEM);
      if (argsort) {
        idc_send_buffers[r] =
          create_non_empty_buffer<int64_t>(size_send_total[r], legate::Memory::GPU_FB_MEM);
      }
    }

    {
      Buffer<VAL*> val_send_buffers_ptr =
        create_non_empty_buffer<VAL*>(num_sort_ranks, legate::Memory::Z_COPY_MEM);
      for (size_t r = 0; r < num_sort_ranks; r++) {
        val_send_buffers_ptr[r] = val_send_buffers[r].ptr(0);
      }

      auto [grid_shape, block_shape] =
        generate_launchconfig_for_2d_copy(segment_size_l, num_segments_l);

      copy_data_to_merge_buffers<<<grid_shape, block_shape, 0, stream>>>(segment_blocks,
                                                                         size_send,
                                                                         local_sorted.values,
                                                                         val_send_buffers_ptr,
                                                                         num_segments_l,
                                                                         num_segments_l_aligned,
                                                                         segment_size_l,
                                                                         my_rank,
                                                                         num_sort_ranks);

      if (argsort) {
        Buffer<int64_t*> idc_send_buffers_ptr =
          create_non_empty_buffer<int64_t*>(num_sort_ranks, legate::Memory::Z_COPY_MEM);
        for (size_t r = 0; r < num_sort_ranks; r++) {
          idc_send_buffers_ptr[r] = idc_send_buffers[r].ptr(0);
        }
        // need to sync as we share values in between host/device
        copy_data_to_merge_buffers<<<grid_shape, block_shape, 0, stream>>>(segment_blocks,
                                                                           size_send,
                                                                           local_sorted.indices,
                                                                           idc_send_buffers_ptr,
                                                                           num_segments_l,
                                                                           num_segments_l_aligned,
                                                                           segment_size_l,
                                                                           my_rank,
                                                                           num_sort_ranks);
        CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));  // needed before Z-copy destroy()
        idc_send_buffers_ptr.destroy();
      } else {
        CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));  // needed before Z-copy destroy()
      }
      val_send_buffers_ptr.destroy();
      CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
    }

    local_sorted.values.destroy();
    if (argsort) {
      local_sorted.indices.destroy();
    }
    segment_blocks.destroy();
  }

  // allocate target buffers
  std::vector<SegmentMergePiece<VAL>> merge_buffers(num_sort_ranks);
  {
    for (size_t r = 0; r < num_sort_ranks; ++r) {
      auto size = size_recv_total[r];

      merge_buffers[r].size = size;

      // initialize segment information
      if (num_segments_l > 1) {
        merge_buffers[r].segments =
          create_non_empty_buffer<size_t>(size, legate::Memory::GPU_FB_MEM);
        // 0  1  2  1  3      // counts per segment to receive
        // 0  1  3  4  7
        // 0 1 2 3 4 5 6
        // 1 1 0 1 1 0 0
        // 1 2 2 3 4 4 4      // segment id for all received elements
        thrust::inclusive_scan(exec_policy,
                               size_recv.ptr(r * num_segments_l_aligned),
                               size_recv.ptr(r * num_segments_l_aligned) + num_segments_l + 1,
                               size_recv.ptr(r * num_segments_l_aligned));
        CUPYNUMERIC_CHECK_CUDA(
          cudaMemsetAsync(merge_buffers[r].segments.ptr(0), 0, size * sizeof(size_t), stream));
        const size_t num_blocks = (num_segments_l + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        assert(sizeof(unsigned long long int) ==
               sizeof(size_t));  // kernel needs to cast for atomicAdd...
        initialize_segment_start_positions<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          size_recv.ptr(r * num_segments_l_aligned),
          num_segments_l - 1,
          merge_buffers[r].segments.ptr(0),
          merge_buffers[r].size);
        thrust::inclusive_scan(exec_policy,
                               merge_buffers[r].segments.ptr(0),
                               merge_buffers[r].segments.ptr(0) + size,
                               merge_buffers[r].segments.ptr(0));
      }

      merge_buffers[r].values = create_non_empty_buffer<VAL>(size, legate::Memory::GPU_FB_MEM);
      if (argsort) {
        merge_buffers[r].indices =
          create_non_empty_buffer<int64_t>(size, legate::Memory::GPU_FB_MEM);
      } else {
        merge_buffers[r].indices = create_non_empty_buffer<int64_t>(0, legate::Memory::GPU_FB_MEM);
      }
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }

  // communicate all2all (in sort dimension)
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_sort_ranks; r++) {
    if (size_send_total[r] > 0) {
      CHECK_NCCL(ncclSend(val_send_buffers[r].ptr(0),
                          size_send_total[r] * sizeof(VAL),
                          ncclInt8,
                          sort_ranks[r],
                          *comm,
                          stream));
    }
    if (merge_buffers[r].size > 0) {
      CHECK_NCCL(ncclRecv(merge_buffers[r].values.ptr(0),
                          merge_buffers[r].size * sizeof(VAL),
                          ncclInt8,
                          sort_ranks[r],
                          *comm,
                          stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());

  if (argsort) {
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_sort_ranks; r++) {
      if (size_send_total[r] > 0) {
        CHECK_NCCL(ncclSend(
          idc_send_buffers[r].ptr(0), size_send_total[r], ncclInt64, sort_ranks[r], *comm, stream));
      }
      if (merge_buffers[r].size > 0) {
        CHECK_NCCL(ncclRecv(merge_buffers[r].indices.ptr(0),
                            merge_buffers[r].size,
                            ncclInt64,
                            sort_ranks[r],
                            *comm,
                            stream));
      }
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  // cleanup remaining buffers
  size_send.destroy();
  size_recv.destroy();
  size_send_total.destroy();
  size_recv_total.destroy();
  for (size_t r = 0; r < num_sort_ranks; r++) {
    val_send_buffers[r].destroy();
    if (argsort) {
      idc_send_buffers[r].destroy();
    }
  }
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 4: merge data
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // now merge sort all into the result buffer
  SegmentMergePiece<VAL> merged_result =
    merge_all_buffers<CODE>(merge_buffers, num_segments_l > 1, argsort, alloc, stream);

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 5: re-balance data to match input/output dimensions
  /////////////////////////////////////////////////////////////////////////////////////////////////

  if (rebalance) {
    assert(!is_unbound_1d_storage);
    rebalance_data(merged_result,
                   output_ptr,
                   my_rank,
                   my_sort_rank,
                   num_sort_ranks,
                   sort_ranks,
                   segment_size_l,
                   num_segments_l,
                   argsort,
                   alloc,
                   stream,
                   comm);
  } else {
    assert(is_unbound_1d_storage);
    merged_result.segments.destroy();
    if (argsort) {
      merged_result.values.destroy();
      output_array_unbound.bind_data(merged_result.indices, Point<1>(merged_result.size));
    } else {
      output_array_unbound.bind_data(merged_result.values, Point<1>(merged_result.size));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <Type::Code CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input_array,
                  legate::PhysicalStore& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t segment_size_l,
                  const size_t segment_size_g,
                  const bool argsort,
                  const bool stable,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const size_t num_sort_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input = input_array.read_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input.accessor.is_dense_row_major(rect));

    auto stream = get_cached_stream();

    bool is_unbound_1d_storage = output_array.is_unbound_store();
    bool need_distributed_sort = segment_size_l != segment_size_g || is_unbound_1d_storage;
    bool rebalance             = !is_unbound_1d_storage;
    assert(DIM == 1 || !is_unbound_1d_storage);

    // initialize sort pointers
    SortPiece<VAL> local_sorted;
    int64_t* indices_ptr = nullptr;
    VAL* values_ptr      = nullptr;
    if (argsort) {
      // make a buffer for input
      auto input_copy     = create_non_empty_buffer<VAL>(volume, legate::Memory::Kind::GPU_FB_MEM);
      local_sorted.values = input_copy;
      values_ptr          = input_copy.ptr(0);

      // initialize indices
      if (need_distributed_sort) {
        auto indices_buffer =
          create_non_empty_buffer<int64_t>(volume, legate::Memory::Kind::GPU_FB_MEM);
        indices_ptr          = indices_buffer.ptr(0);
        local_sorted.indices = indices_buffer;
        local_sorted.size    = volume;
      } else {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        assert(rect.empty() || output.accessor.is_dense_row_major(rect));
        indices_ptr = output.ptr(rect.lo);
      }
      size_t offset = rect.lo[DIM - 1];
      if (volume > 0) {
        if (DIM == 1) {
          thrust::sequence(DEFAULT_POLICY.on(stream), indices_ptr, indices_ptr + volume, offset);
        } else {
          thrust::transform(DEFAULT_POLICY.on(stream),
                            thrust::make_counting_iterator<int64_t>(0),
                            thrust::make_counting_iterator<int64_t>(volume),
                            thrust::make_constant_iterator<int64_t>(segment_size_l),
                            indices_ptr,
                            modulusWithOffset(offset));
        }
      }
    } else {
      // initialize output
      if (need_distributed_sort) {
        auto input_copy = create_non_empty_buffer<VAL>(volume, legate::Memory::Kind::GPU_FB_MEM);
        values_ptr      = input_copy.ptr(0);
        local_sorted.values = input_copy;
        local_sorted.indices =
          create_non_empty_buffer<int64_t>(0, legate::Memory::Kind::GPU_FB_MEM);
        local_sorted.size = volume;
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        assert(rect.empty() || output.accessor.is_dense_row_major(rect));
        values_ptr = output.ptr(rect.lo);
      }
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

    if (volume > 0) {
      // sort data (locally)
      local_sort<CODE>(input.ptr(rect.lo),
                       values_ptr,
                       indices_ptr,
                       indices_ptr,
                       volume,
                       segment_size_l,
                       stable,
                       stream);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

    if (need_distributed_sort) {
      if (is_index_space) {
        assert(is_index_space || is_unbound_1d_storage);
        std::vector<size_t> sort_ranks(num_sort_ranks);
        size_t rank_group = local_rank / num_sort_ranks;
        for (size_t r = 0; r < num_sort_ranks; ++r) {
          sort_ranks[r] = rank_group * num_sort_ranks + r;
        }

        void* output_ptr = nullptr;
        // in case the storage *is NOT* unbound -- we provide a target pointer
        // in case the storage *is* unbound -- the result will be appended to output_array
        if (volume > 0 && !is_unbound_1d_storage) {
          if (argsort) {
            auto output = output_array.write_accessor<int64_t, DIM>(rect);
            assert(output.accessor.is_dense_row_major(rect));
            output_ptr = static_cast<void*>(output.ptr(rect.lo));
          } else {
            auto output = output_array.write_accessor<VAL, DIM>(rect);
            assert(output.accessor.is_dense_row_major(rect));
            output_ptr = static_cast<void*>(output.ptr(rect.lo));
          }
        }

        sample_sort_nccl_nd<CODE>(context,
                                  local_sorted,
                                  output_array,
                                  output_ptr,
                                  local_rank,
                                  num_ranks,
                                  segment_size_g,
                                  local_rank % num_sort_ranks,
                                  num_sort_ranks,
                                  sort_ranks.data(),
                                  segment_size_l,
                                  rebalance,
                                  argsort,
                                  stream,
                                  comms[0].get<ncclComm_t*>());
      } else {
        // edge case where we have an unbound store but only 1 GPU was assigned with the task
        if (argsort) {
          local_sorted.values.destroy();
          output_array.bind_data(local_sorted.indices, Point<1>(local_sorted.size));
        } else {
          output_array.bind_data(local_sorted.values, Point<1>(local_sorted.size));
        }
      }
    } else if (argsort) {
      // cleanup for non distributed argsort
      local_sorted.values.destroy();
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
