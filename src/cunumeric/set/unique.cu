/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/set/unique.h"
#include "cunumeric/set/unique_template.inl"
#include "cunumeric/utilities/thrust_util.h"

#include "cunumeric/cuda_help.h"

#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace cunumeric {

using namespace legate;

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  copy_into_buffer(VAL* out,
                   const AccessorRO<VAL, DIM> accessor,
                   const Point<DIM> lo,
                   const Pitches<DIM - 1> pitches,
                   const size_t volume)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  auto point  = pitches.unflatten(offset, lo);
  out[offset] = accessor[point];
}

template <int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  fill_subset_indices(int64_t* out,
                      const Point<DIM> lo,
                      const Pitches<DIM - 1> pitches,
                      const size_t volume,
                      const DomainPoint parent_point)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) return;
  auto point     = pitches.unflatten(offset, lo);
  int multiplier = 1;
  size_t index   = 0;
  for (int i = DIM - 1; i >= 0; i--) {
    index += point[i] * multiplier;
    multiplier *= parent_point[i];
  }
  out[offset] = index;
}

template <typename VAL>
using Piece = std::pair<Buffer<VAL>, size_t>;

auto get_aligned_size = [](auto size) { return std::max<size_t>(16, (size + 15) / 16 * 16); };

template <typename VAL>
static std::pair<Piece<VAL>, Piece<int64_t>> tree_reduce(std::vector<Array>& outputs,
                                                         Piece<VAL> my_piece,
                                                         Piece<int64_t> indices,
                                                         size_t my_id,
                                                         size_t num_ranks,
                                                         cudaStream_t stream,
                                                         ncclComm_t* comm,
                                                         bool return_index)
{
  auto& output     = outputs[0];
  size_t remaining = num_ranks;
  size_t radix     = 2;
  auto all_sizes   = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);

  while (remaining > 1) {
    // TODO: This could be point-to-point, as we don't need all the sizes,
    //       but I suspect point-to-point can be slower...
    all_sizes[my_id] = my_piece.second;
    CHECK_NCCL(ncclAllGather(all_sizes.ptr(my_id), all_sizes.ptr(0), 1, ncclUint64, *comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    Piece<VAL> other_piece;
    Piece<int64_t> other_index;
    size_t offset           = radix / 2;
    bool received_something = false;
    CHECK_NCCL(ncclGroupStart());
    if (my_id % radix == 0)  // This is one of the receivers
    {
      auto other_id = my_id + offset;
      if (other_id < num_ranks)  // Make sure someone's sending anything
      {
        auto other_size = all_sizes[other_id];
        auto recv_size  = get_aligned_size(other_size * sizeof(VAL));
        auto buf_size   = (recv_size + sizeof(VAL) - 1) / sizeof(VAL);
        assert(other_size <= buf_size);
        other_piece.second = other_size;
        other_piece.first  = create_buffer<VAL>(buf_size);
        CHECK_NCCL(
          ncclRecv(other_piece.first.ptr(0), recv_size, ncclInt8, other_id, *comm, stream));
        if (return_index) {
          other_index.second   = other_size;
          auto recv_size_index = get_aligned_size(other_size * sizeof(int64_t));
          auto buf_size_index  = (recv_size_index + sizeof(int64_t) - 1) / sizeof(int64_t);
          other_index.first    = create_buffer<int64_t>(buf_size_index);
          CHECK_NCCL(
            ncclRecv(other_index.first.ptr(0), recv_size_index, ncclInt8, other_id, *comm, stream));
        }
        received_something = true;
      }
    } else if (my_id % radix == offset)  // This is one of the senders
    {
      auto other_id  = my_id - offset;
      auto send_size = get_aligned_size(my_piece.second * sizeof(VAL));
      CHECK_NCCL(ncclSend(my_piece.first.ptr(0), send_size, ncclInt8, other_id, *comm, stream));
      if (return_index) {
        auto send_size_index = get_aligned_size(indices.second * sizeof(int64_t));
        CHECK_NCCL(
          ncclSend(indices.first.ptr(0), send_size_index, ncclInt8, other_id, *comm, stream));
      }
    }
    CHECK_NCCL(ncclGroupEnd());

    // Now we merge our pieces with others and deduplicate the merged ones
    if (received_something) {
      auto merged_size = my_piece.second + other_piece.second;
      auto merged      = create_buffer<VAL>(merged_size);
      auto p_merged    = merged.ptr(0);
      auto p_mine      = my_piece.first.ptr(0);
      auto p_other     = other_piece.first.ptr(0);

      Buffer<int64_t, 1> merged_index;
      if (return_index) {
        merged_index        = create_buffer<int64_t>(merged_size);
        auto p_merged_index = merged_index.ptr(0);
        auto p_mine_index   = indices.first.ptr(0);
        auto p_other_index  = other_index.first.ptr(0);

        auto my_zip    = thrust::make_zip_iterator(thrust::make_tuple(p_mine, p_mine_index));
        auto other_zip = thrust::make_zip_iterator(thrust::make_tuple(p_other, p_other_index));
        auto final_zip = thrust::make_zip_iterator(thrust::make_tuple(p_merged, p_merged_index));

        thrust::merge(DEFAULT_POLICY.on(stream),
                      my_zip,
                      my_zip + my_piece.second,
                      other_zip,
                      other_zip + other_piece.second,
                      final_zip);

        auto end = thrust::unique_by_key(
          DEFAULT_POLICY.on(stream), p_merged, p_merged + merged_size, p_merged_index);

        my_piece.second = end.first - p_merged;
        indices.second  = my_piece.second;
      } else {
        thrust::merge(DEFAULT_POLICY.on(stream),
                      p_mine,
                      p_mine + my_piece.second,
                      p_other,
                      p_other + other_piece.second,
                      p_merged);
        auto* end = thrust::unique(DEFAULT_POLICY.on(stream), p_merged, p_merged + merged_size);
        my_piece.second = end - p_merged;
      }

      // Make sure we release the memory so that we can reuse it
      my_piece.first.destroy();
      other_piece.first.destroy();

      auto buf_size =
        (get_aligned_size(my_piece.second * sizeof(VAL)) + sizeof(VAL) - 1) / sizeof(VAL);
      assert(my_piece.second <= buf_size);
      my_piece.first = output.create_output_buffer<VAL, 1>(buf_size);

      CHECK_CUDA(cudaMemcpyAsync(my_piece.first.ptr(0),
                                 p_merged,
                                 sizeof(VAL) * my_piece.second,
                                 cudaMemcpyDeviceToDevice,
                                 stream));
      merged.destroy();

      if (return_index) {
        indices.first.destroy();
        other_index.first.destroy();

        auto buf_size_index =
          (get_aligned_size(my_piece.second * sizeof(int64_t)) + sizeof(int64_t) - 1) /
          sizeof(int64_t);
        assert(my_piece.second <= buf_size_index);
        indices.first = outputs[1].create_output_buffer<int64_t, 1>(buf_size_index);

        CHECK_CUDA(cudaMemcpyAsync(indices.first.ptr(0),
                                   merged_index.ptr(0),
                                   sizeof(int64_t) * my_piece.second,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
        merged_index.destroy();
      }
    }

    remaining = (remaining + 1) / 2;
    radix *= 2;
  }

  if (my_id != 0) {
    my_piece.second = 0;
    my_piece.first  = output.create_output_buffer<VAL, 1>(0);
    indices.second  = 0;
    indices.first   = output.create_output_buffer<int64_t, 1>(0);
  }

  return {my_piece, indices};
}

template <Type::Code CODE, int32_t DIM>
struct UniqueImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(std::vector<Array>& outputs,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const std::vector<comm::Communicator>& comms,
                  const DomainPoint& point,
                  const Domain& launch_domain,
                  const bool return_index,
                  const DomainPoint& parent_point)
  {
    auto& output = outputs[0];
    auto stream  = get_cached_stream();

    // Make a copy of the input as we're going to sort it
    auto temp = create_buffer<VAL>(volume);
    VAL* ptr  = temp.ptr(0);
    VAL* end  = ptr;

    int64_t* index_ptr = nullptr;
    if (return_index) {
      auto index_temp         = create_buffer<int64_t>(volume);
      index_ptr               = index_temp.ptr(0);
      const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      fill_subset_indices<DIM><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        index_ptr, rect.lo, pitches, volume, parent_point);
    }

    if (volume > 0) {
      if (in.accessor.is_dense_arbitrary(rect)) {
        auto* src = in.ptr(rect.lo);
        CHECK_CUDA(
          cudaMemcpyAsync(ptr, src, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
      } else {
        const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        copy_into_buffer<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          ptr, in, rect.lo, pitches, volume);
      }
      CHECK_CUDA_STREAM(stream);

      if (return_index) {
        // Find unique values with corresponding index
        auto zip_start = thrust::make_zip_iterator(thrust::make_tuple(ptr, index_ptr));

        thrust::sort(DEFAULT_POLICY.on(stream), zip_start, zip_start + volume);
        auto tuple_end =
          thrust::unique_by_key(DEFAULT_POLICY.on(stream), ptr, ptr + volume, index_ptr);
        end = tuple_end.first;
      } else {
        // Find unique values
        thrust::sort(DEFAULT_POLICY.on(stream), ptr, ptr + volume);
        end = thrust::unique(DEFAULT_POLICY.on(stream), ptr, ptr + volume);
      }
    }

    Piece<VAL> result;
    Piece<int64_t> indices;
    result.second  = end - ptr;
    indices.second = end - ptr;
    auto buf_size = (get_aligned_size(result.second * sizeof(VAL)) + sizeof(VAL) - 1) / sizeof(VAL);
    assert(end - ptr <= buf_size);
    result.first = output.create_output_buffer<VAL, 1>(buf_size);
    if (return_index) {
      auto buf_size =
        (get_aligned_size(result.second * sizeof(int64_t)) + sizeof(int64_t) - 1) / sizeof(int64_t);
      indices.first = outputs[1].create_output_buffer<int64_t, 1>(buf_size);
    }
    if (result.second > 0) {
      CHECK_CUDA(cudaMemcpyAsync(
        result.first.ptr(0), ptr, sizeof(VAL) * result.second, cudaMemcpyDeviceToDevice, stream));
      if (return_index)
        CHECK_CUDA(cudaMemcpyAsync(indices.first.ptr(0),
                                   index_ptr,
                                   sizeof(int64_t) * indices.second,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }

    if (comms.size() > 0) {
      // The launch domain is 1D because of the output region
      assert(point.dim == 1);
      auto comm = comms[0].get<ncclComm_t*>();
      auto ret  = tree_reduce(
        outputs, result, indices, point[0], launch_domain.get_volume(), stream, comm, return_index);
      result = ret.first;
      if (return_index) indices = ret.second;
    }
    CHECK_CUDA_STREAM(stream);

    // Finally we pack the result
    output.bind_data(result.first, Point<1>(result.second));
    if (return_index) { outputs[1].bind_data(indices.first, Point<1>(indices.second)); }
  }
};

/*static*/ void UniqueTask::gpu_variant(TaskContext& context)
{
  unique_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
