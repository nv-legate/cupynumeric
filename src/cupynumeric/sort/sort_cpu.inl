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

// Useful for IDEs
#include "cupynumeric/sort/sort.h"
#include "cupynumeric/pitches.h"
#include "legate/comm/coll.h"

#include <thrust/detail/config.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

#include <functional>
#include <numeric>

namespace cupynumeric {

namespace {

template <typename T>
legate::Buffer<T> create_non_empty_buffer(size_t size)
{
  return legate::create_buffer<T>(
    std::max(size, size_t{1}), legate::Memory::Kind::NO_MEMKIND, alignof(T));
}

}  // namespace

using namespace legate;

// sorts inptr in-place, if argptr not nullptr it returns sort indices
template <typename VAL, typename DerivedPolicy>
void thrust_local_sort_inplace(VAL* inptr,
                               int64_t* argptr,
                               const size_t volume,
                               const size_t sort_dim_size,
                               const bool stable_argsort,
                               const DerivedPolicy& exec)
{
  if (argptr == nullptr) {
    // sort (in place)
    for (size_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
      if (stable_argsort) {
        thrust::stable_sort(exec, inptr + start_idx, inptr + start_idx + sort_dim_size);
      } else {
        thrust::sort(exec, inptr + start_idx, inptr + start_idx + sort_dim_size);
      }
    }
  } else {
    // argsort
    for (uint64_t start_idx = 0; start_idx < volume; start_idx += sort_dim_size) {
      int64_t* segmentValues = argptr + start_idx;
      VAL* segmentKeys       = inptr + start_idx;
      if (stable_argsort) {
        thrust::stable_sort_by_key(exec, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
      } else {
        thrust::sort_by_key(exec, segmentKeys, segmentKeys + sort_dim_size, segmentValues);
      }
    }
  }
}

template <typename VAL, typename DerivedPolicy>
void rebalance_data(SegmentMergePiece<VAL>& merge_buffer,
                    void* output_ptr,
                    /* global domain information */
                    size_t my_rank,    // global rank
                    size_t num_ranks,  // global number of ranks
                    /* domain information in sort dimension */
                    size_t my_sort_rank,    // local rank id in sort dimension
                    size_t num_sort_ranks,  // #ranks that share a sort dimension
                    size_t* sort_ranks,     // rank ids that share a sort dimension with us
                    size_t segment_size_l,  // (local) segment size
                    size_t num_segments_l,
                    /* other */
                    bool argsort,
                    const DerivedPolicy& exec,
                    comm::coll::CollComm comm)
{
  // output is either values or indices
  VAL* output_values      = nullptr;
  int64_t* output_indices = nullptr;
  if (argsort) {
    output_indices = static_cast<int64_t*>(output_ptr);
  } else {
    output_values = static_cast<VAL*>(output_ptr);
  }

  {
    // compute diff for each segment
    auto segment_diff = create_non_empty_buffer<int64_t>(num_segments_l);
    {
      if (num_segments_l > 1) {
        auto* p_segments = merge_buffer.segments.ptr(0);
        size_t position  = 0;
        size_t count     = 0;
        for (size_t segment = 0; segment < num_segments_l; ++segment) {
          while (position < merge_buffer.size && p_segments[position] == segment) {
            position++;
            count++;
          }
          segment_diff[segment] = count - segment_size_l;
          count                 = 0;
        }
      } else {
        segment_diff[0] = merge_buffer.size - segment_size_l;
      }
    }

    merge_buffer.segments.destroy();
    if (argsort) {
      merge_buffer.values.destroy();
    }

#ifdef DEBUG_CUPYNUMERIC
    {
      size_t reduce =
        thrust::reduce(exec, segment_diff.ptr(0), segment_diff.ptr(0) + num_segments_l, 0);
      size_t volume = segment_size_l * num_segments_l;
      assert(merge_buffer.size - volume == reduce);
    }
#endif

    // allocate target
    auto segment_diff_buffers = create_non_empty_buffer<int64_t>(num_segments_l * num_sort_ranks);

    {
      // using alltoallv to mimic allgather on subset
      auto comm_size = create_buffer<int32_t>(num_ranks);
      auto sdispls   = create_buffer<int32_t>(num_ranks);
      auto rdispls   = create_buffer<int32_t>(num_ranks);

      std::fill(comm_size.ptr(0), comm_size.ptr(0) + num_ranks, 0);
      std::fill(sdispls.ptr(0), sdispls.ptr(0) + num_ranks, 0);
      std::fill(rdispls.ptr(0), rdispls.ptr(0) + num_ranks, 0);

      for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
        comm_size[sort_ranks[sort_rank]] = num_segments_l;
        rdispls[sort_ranks[sort_rank]]   = sort_rank * num_segments_l;
      }

      static_cast<void>(
        comm::coll::collAlltoallv(segment_diff.ptr(0),
                                  comm_size.ptr(0),  // num_segments_l for all in sort group
                                  sdispls.ptr(0),    // zero for all
                                  segment_diff_buffers.ptr(0),
                                  comm_size.ptr(0),  // num_segments_l for all in sort group
                                  rdispls.ptr(0),    // exclusive_scan of recv size
                                  comm::coll::CollDataType::CollInt64,
                                  comm));

      comm_size.destroy();
      sdispls.destroy();
      rdispls.destroy();
    }

    // copy to transpose structure [segments][ranks]  (not in-place for now)
    auto segment_diff_2d = create_non_empty_buffer<int64_t>(num_segments_l * num_sort_ranks);
    {
      int pos = 0;
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
          segment_diff_2d[pos++] = segment_diff_buffers[sort_rank * num_segments_l + segment];
        }
      }
      segment_diff_buffers.destroy();
    }

#ifdef DEBUG_CUPYNUMERIC
    {
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        assert(0 == thrust::reduce(exec,
                                   segment_diff_2d.ptr(segment * num_sort_ranks),
                                   segment_diff_2d.ptr(segment * num_sort_ranks) + num_sort_ranks,
                                   0));
      }
    }
#endif

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
    auto send_left  = create_non_empty_buffer<int64_t>(num_segments_l);
    auto send_right = create_non_empty_buffer<int64_t>(num_segments_l);

    // compute data to send....
    auto segment_diff_2d_scan = create_non_empty_buffer<int64_t>(num_segments_l * num_sort_ranks);

    auto* segment_diff_2d_ptr      = segment_diff_2d.ptr(0);
    auto* segment_diff_2d_scan_ptr = segment_diff_2d_scan.ptr(0);
    thrust::inclusive_scan(exec,
                           segment_diff_2d_ptr,
                           segment_diff_2d_ptr + num_segments_l * num_sort_ranks,
                           segment_diff_2d_scan_ptr);
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      send_right[segment] = segment_diff_2d_scan_ptr[segment * num_sort_ranks + my_sort_rank];
    }

    auto iter_in = std::reverse_iterator(segment_diff_2d_ptr + num_segments_l * num_sort_ranks);
    auto iter_out =
      std::reverse_iterator(segment_diff_2d_scan_ptr + num_segments_l * num_sort_ranks);
    thrust::inclusive_scan(exec, iter_in, iter_in + num_segments_l * num_sort_ranks, iter_out);
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      send_left[segment] = segment_diff_2d_scan_ptr[segment * num_sort_ranks + my_sort_rank];
    }

    segment_diff_2d.destroy();
    segment_diff_2d_scan.destroy();

    // package data to send
    size_t send_left_size  = 0;
    size_t recv_left_size  = 0;
    size_t send_right_size = 0;
    size_t recv_right_size = 0;
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      if (send_left[segment] > 0) {
        send_left_size += send_left[segment];
      } else {
        recv_left_size -= send_left[segment];
      }
      if (send_right[segment] > 0) {
        send_right_size += send_right[segment];
      } else {
        recv_right_size -= send_right[segment];
      }
    }

    SortPiece<VAL> send_leftright_data, recv_leftright_data;
    send_leftright_data.size = send_left_size + send_right_size;
    recv_leftright_data.size = recv_left_size + recv_right_size;

    if (argsort) {
      send_leftright_data.indices = create_non_empty_buffer<int64_t>(send_leftright_data.size);
      recv_leftright_data.indices = create_non_empty_buffer<int64_t>(recv_leftright_data.size);
    } else {
      send_leftright_data.values = create_non_empty_buffer<VAL>(send_leftright_data.size);
      recv_leftright_data.values = create_non_empty_buffer<VAL>(recv_leftright_data.size);
    }

    // copy into send buffer
    {
      size_t segment_start  = 0;
      size_t send_left_pos  = 0;
      size_t send_right_pos = send_left_size;
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        // copy left
        if (send_left[segment] > 0) {
          size_t size = send_left[segment];
          if (argsort) {
            std::memcpy(send_leftright_data.indices.ptr(send_left_pos),
                        merge_buffer.indices.ptr(segment_start),
                        size * sizeof(int64_t));
          } else {
            std::memcpy(send_leftright_data.values.ptr(send_left_pos),
                        merge_buffer.values.ptr(segment_start),
                        size * sizeof(VAL));
          }
          send_left_pos += size;
        }

        segment_start += segment_diff[segment] + segment_size_l;

        // copy right
        if (send_right[segment] > 0) {
          size_t size = send_right[segment];
          if (argsort) {
            std::memcpy(send_leftright_data.indices.ptr(send_right_pos),
                        merge_buffer.indices.ptr(segment_start - size),
                        size * sizeof(int64_t));
          } else {
            std::memcpy(send_leftright_data.values.ptr(send_right_pos),
                        merge_buffer.values.ptr(segment_start - size),
                        size * sizeof(VAL));
          }
          send_right_pos += size;
        }
      }
      assert(send_left_pos == send_left_size);
      assert(send_right_pos == send_left_size + send_right_size);
      assert(segment_start == merge_buffer.size);
    }

    {
      // using alltoallv to mimic allgather on subset
      auto comm_send_size = create_buffer<int32_t>(num_ranks);
      auto comm_recv_size = create_buffer<int32_t>(num_ranks);
      auto sdispls        = create_buffer<int32_t>(num_ranks);
      auto rdispls        = create_buffer<int32_t>(num_ranks);

      std::fill(comm_send_size.ptr(0), comm_send_size.ptr(0) + num_ranks, 0);
      std::fill(comm_recv_size.ptr(0), comm_recv_size.ptr(0) + num_ranks, 0);
      std::fill(sdispls.ptr(0), sdispls.ptr(0) + num_ranks, 0);
      std::fill(rdispls.ptr(0), rdispls.ptr(0) + num_ranks, 0);

      size_t bytesize = argsort ? sizeof(int64_t) : sizeof(VAL);

      // left-comm
      if (send_left_size > 0) {
        comm_send_size[sort_ranks[my_sort_rank - 1]] = send_left_size * bytesize;
      }
      if (recv_left_size > 0) {
        comm_recv_size[sort_ranks[my_sort_rank - 1]] = recv_left_size * bytesize;
      }

      // right-comm
      if (send_right_size > 0) {
        comm_send_size[sort_ranks[my_sort_rank + 1]] = send_right_size * bytesize;
        sdispls[sort_ranks[my_sort_rank + 1]]        = send_left_size * bytesize;
      }
      if (recv_right_size > 0) {
        comm_recv_size[sort_ranks[my_sort_rank + 1]] = recv_right_size * bytesize;
        rdispls[sort_ranks[my_sort_rank + 1]]        = recv_left_size * bytesize;
      }

      if (argsort) {
        static_cast<void>(comm::coll::collAlltoallv(send_leftright_data.indices.ptr(0),
                                                    comm_send_size.ptr(0),
                                                    sdispls.ptr(0),
                                                    recv_leftright_data.indices.ptr(0),
                                                    comm_recv_size.ptr(0),
                                                    rdispls.ptr(0),
                                                    comm::coll::CollDataType::CollInt8,
                                                    comm));
      } else {
        static_cast<void>(comm::coll::collAlltoallv(send_leftright_data.values.ptr(0),
                                                    comm_send_size.ptr(0),
                                                    sdispls.ptr(0),
                                                    recv_leftright_data.values.ptr(0),
                                                    comm_recv_size.ptr(0),
                                                    rdispls.ptr(0),
                                                    comm::coll::CollDataType::CollInt8,
                                                    comm));
      }

      comm_send_size.destroy();
      comm_recv_size.destroy();
      sdispls.destroy();
      rdispls.destroy();
    }

    if (argsort) {
      send_leftright_data.indices.destroy();
    } else {
      send_leftright_data.values.destroy();
    }

    // merge data into output_values or output_indices
    {
      size_t bytesize          = argsort ? sizeof(int64_t) : sizeof(VAL);
      size_t segment_start_src = 0;
      size_t recv_left_pos     = 0;
      size_t recv_right_pos    = recv_left_size;
      size_t target_pos        = 0;
      for (size_t segment = 0; segment < num_segments_l; ++segment) {
        // copy from left
        if (send_left[segment] < 0) {
          auto copy_size = -send_left[segment];
          if (argsort) {
            std::memcpy(output_indices + target_pos,
                        recv_leftright_data.indices.ptr(recv_left_pos),
                        copy_size * bytesize);
          } else {
            std::memcpy(output_values + target_pos,
                        recv_leftright_data.values.ptr(recv_left_pos),
                        copy_size * bytesize);
          }
          recv_left_pos += copy_size;
          target_pos += copy_size;
        }

        // copy own
        {
          size_t segment_size_src = segment_diff[segment] + segment_size_l;
          size_t copy_size        = segment_size_src;
          size_t offset_src       = 0;
          if (send_left[segment] > 0) {
            offset_src = send_left[segment];
            copy_size -= offset_src;
          }
          if (send_right[segment] > 0) {
            copy_size -= send_right[segment];
          }

          if (argsort) {
            std::memcpy(output_indices + target_pos,
                        merge_buffer.indices.ptr(segment_start_src + offset_src),
                        copy_size * bytesize);
          } else {
            std::memcpy(output_values + target_pos,
                        merge_buffer.values.ptr(segment_start_src + offset_src),
                        copy_size * bytesize);
          }
          segment_start_src += segment_size_src;
          target_pos += copy_size;
        }

        // copy from right
        if (send_right[segment] < 0) {
          auto copy_size = -send_right[segment];
          if (argsort) {
            std::memcpy(output_indices + target_pos,
                        recv_leftright_data.indices.ptr(recv_right_pos),
                        copy_size * bytesize);
          } else {
            std::memcpy(output_values + target_pos,
                        recv_leftright_data.values.ptr(recv_right_pos),
                        copy_size * bytesize);
          }
          recv_right_pos += copy_size;
          target_pos += copy_size;
        }

        assert(target_pos == (segment + 1) * segment_size_l);
      }
      assert(recv_left_pos == recv_left_size);
      assert(recv_right_pos == recv_left_size + recv_right_size);
      assert(segment_start_src == merge_buffer.size);
    }

    // remove segment_sizes, all buffers should be destroyed...
    segment_diff.destroy();
    send_left.destroy();
    send_right.destroy();
    if (argsort) {
      merge_buffer.indices.destroy();
      recv_leftright_data.indices.destroy();
    } else {
      merge_buffer.values.destroy();
      recv_leftright_data.values.destroy();
    }
  }
}

template <Type::Code CODE, typename DerivedPolicy>
void sample_sort_nd(
  SortPiece<type_of<CODE>> local_sorted,
  legate::PhysicalStore output_array_unbound,  // only for unbound usage when !rebalance
  void* output_ptr,
  /* global domain information */
  size_t my_rank,  // global rank
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
  const DerivedPolicy& exec,
  comm::coll::CollComm comm)
{
  using VAL = type_of<CODE>;

  size_t volume              = local_sorted.size;
  bool is_unbound_1d_storage = output_array_unbound.is_unbound_store();

  assert((volume > 0 && segment_size_l > 0) || volume == segment_size_l);

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 0: detection of empty nodes
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // first of all we need to check for processes that don't want
  // to take part in the computation. This might lead to a reduction of
  // sort ranks. Note that if segment_size_l>0 && volume==0 means that we have
  // a full sort group being empty, this should not affect local sort rank size.
  {
    auto worker_counts     = create_buffer<int32_t>(num_ranks);
    worker_counts[my_rank] = (segment_size_l > 0 ? 1 : 0);
    static_cast<void>(comm::coll::collAllgather(worker_counts.ptr(my_rank),
                                                worker_counts.ptr(0),
                                                1,
                                                comm::coll::CollDataType::CollInt,
                                                comm));

    auto p_worker_count = worker_counts.ptr(0);
    int32_t worker_count =
      std::accumulate(p_worker_count, p_worker_count + num_ranks, 0, std::plus<int32_t>());

    if (static_cast<size_t>(worker_count) < num_ranks) {
      const size_t number_sort_groups = num_ranks / num_sort_ranks;
      num_sort_ranks                  = worker_count / number_sort_groups;

      if (volume == 0) {
        // unfortunately we cannot early out here... we still need to participate in collective
        // communication
        assert(my_sort_rank >= num_sort_ranks);
        num_sort_ranks = 0;
      }
    }
    worker_counts.destroy();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 1: select and share samples accross sort domain
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // collect local samples - for now we take num_sort_ranks samples for every node/line
  // worst case this leads to imbalance of x2
  size_t num_segments_l            = segment_size_l > 0 ? volume / segment_size_l : 0;
  size_t num_samples_per_segment_l = num_sort_ranks;
  size_t num_samples_l             = num_samples_per_segment_l * num_segments_l;
  size_t num_samples_per_segment_g = num_samples_per_segment_l * num_sort_ranks;
  size_t num_samples_g             = num_samples_per_segment_g * num_segments_l;
  auto samples_l                   = create_non_empty_buffer<SegmentSample<VAL>>(num_samples_l);
  auto samples_g                   = create_non_empty_buffer<SegmentSample<VAL>>(num_samples_g);
  auto* p_samples                  = samples_l.ptr(0);
  auto* local_values               = local_sorted.values.ptr(0);

  {
    size_t position = 0;
    for (size_t segment_id_l = 0; segment_id_l < num_segments_l; ++segment_id_l) {
      for (size_t segment_sample_idx = 0; segment_sample_idx < num_samples_per_segment_l;
           ++segment_sample_idx) {
        if (num_samples_per_segment_l < segment_size_l) {
          const size_t index =
            segment_id_l * segment_size_l +
            (segment_sample_idx + 1) * segment_size_l / num_samples_per_segment_l - 1;
          p_samples[position].value    = local_values[index];
          p_samples[position].rank     = my_sort_rank;
          p_samples[position].segment  = segment_id_l;
          p_samples[position].position = index;
        } else {
          // edge case where num_samples_l > volume
          if (segment_sample_idx < segment_size_l) {
            const size_t index           = segment_id_l * segment_size_l + segment_sample_idx;
            p_samples[position].value    = local_values[index];
            p_samples[position].rank     = my_sort_rank;
            p_samples[position].segment  = segment_id_l;
            p_samples[position].position = index;
          } else {
            p_samples[position].rank    = -1;  // not populated
            p_samples[position].segment = segment_id_l;
          }
        }
        position++;
      }
    }

    p_samples = samples_g.ptr(0);

    {
      // This does not work! num_segments_l & num_samples_l not the same for all sort groups!
      /*comm::coll::collAllgather(p_samples + num_samples_l * my_sort_rank,
                                p_samples,
                                num_samples_l * sizeof(SegmentSample<VAL>),
                                comm::coll::CollDataType::Code::CollUint8,
                                comm);*/

      // workaround - using alltoallv to mimic allgather on subset
      auto comm_size = create_buffer<int32_t>(num_ranks);
      auto sdispls   = create_buffer<int32_t>(num_ranks);
      auto rdispls   = create_buffer<int32_t>(num_ranks);

      std::fill(comm_size.ptr(0), comm_size.ptr(0) + num_ranks, 0);
      std::fill(sdispls.ptr(0), sdispls.ptr(0) + num_ranks, 0);
      for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
        comm_size[sort_ranks[sort_rank]] = num_samples_l * sizeof(SegmentSample<VAL>);
      }
      auto p_comm_size = comm_size.ptr(0);
      thrust::exclusive_scan(exec, p_comm_size, p_comm_size + num_ranks, rdispls.ptr(0), 0);

      static_cast<void>(
        comm::coll::collAlltoallv(samples_l.ptr(0),
                                  comm_size.ptr(0),  // num_samples_l*size for all in sort group
                                  sdispls.ptr(0),    // zero for all
                                  p_samples,
                                  comm_size.ptr(0),  // num_samples_l*size for all in sort group
                                  rdispls.ptr(0),    // exclusive_scan of recv size
                                  comm::coll::CollDataType::CollUint8,
                                  comm));

      samples_l.destroy();
      comm_size.destroy();
      sdispls.destroy();
      rdispls.destroy();
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 2: select splitters from samples and collect positions in local data
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // sort samples on device
  if (num_samples_g > 0) {
    thrust::stable_sort(exec, p_samples, p_samples + num_samples_g, SegmentSampleComparator<VAL>());
  }

  // check whether we have invalid samples (in case one participant did not have enough)
  int32_t num_usable_samples_per_segment = num_samples_per_segment_g;
  for (int32_t i = num_samples_per_segment_g - 1; i >= 0; i--) {
    if (p_samples[i].rank != -1) {
      break;
    } else {
      num_usable_samples_per_segment--;
    }
  }

  // segment_blocks[r][segment]->global position in data for segment and r
  // perform blocksize wide scan on size_send[r][block*blocksize] within warp
  auto segment_blocks = create_non_empty_buffer<int32_t>(num_sort_ranks * num_segments_l);

  // initialize sizes to send [r][segment]
  auto size_send   = create_non_empty_buffer<int32_t>(num_sort_ranks * (num_segments_l + 1));
  auto p_size_send = size_send.ptr(0);
  std::fill(p_size_send, p_size_send + num_sort_ranks * (num_segments_l + 1), 0);

  {
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      int32_t start_position = segment_size_l * segment;
      for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
        int32_t end_position = (segment + 1) * segment_size_l;
        if (sort_rank < num_sort_ranks - 1) {
          // actually search for split position in data
          const int32_t index =
            (sort_rank + 1) * num_usable_samples_per_segment / (num_sort_ranks)-1;
          auto& splitter = p_samples[segment * num_samples_per_segment_g + index];
          if (my_sort_rank > static_cast<size_t>(splitter.rank)) {
            // position of the last position with smaller value than splitter.value + 1
            end_position =
              std::lower_bound(
                local_values + start_position, local_values + end_position, splitter.value) -
              local_values;
          } else if (my_sort_rank < static_cast<size_t>(splitter.rank)) {
            // position of the first position with value larger than splitter.value
            end_position =
              std::upper_bound(
                local_values + start_position, local_values + end_position, splitter.value) -
              local_values;
          } else {
            end_position = splitter.position + 1;
          }
        }

        int32_t size = end_position - start_position;

        size_send[sort_rank * (num_segments_l + 1) + segment] = size;

        // collect sum for rank
        size_send[sort_rank * (num_segments_l + 1) + num_segments_l] += size;

        segment_blocks[sort_rank * num_segments_l + segment] = start_position;

        start_position = end_position;
      }
    }
  }

  // cleanup intermediate data structures
  samples_g.destroy();

#ifdef DEBUG_CUPYNUMERIC
  {
    size_t total_send = 0;
    for (size_t r = 0; r < num_sort_ranks; ++r) {
      total_send += size_send[r * (num_segments_l + 1) + num_segments_l];
    }
    assert(total_send == volume);
  }
#endif

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 3: communicate data in sort domain
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // all2all exchange send/receive sizes  [r][segment]
  auto size_recv = create_non_empty_buffer<int32_t>(num_sort_ranks * (num_segments_l + 1));

  {
    // workaround - using alltoallv
    auto comm_size = create_buffer<int32_t>(num_ranks);
    auto displs    = create_buffer<int32_t>(num_ranks);

    std::fill(comm_size.ptr(0), comm_size.ptr(0) + num_ranks, 0);
    for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
      comm_size[sort_ranks[sort_rank]] = num_segments_l + 1;
    }
    auto p_comm_size = comm_size.ptr(0);
    thrust::exclusive_scan(exec, p_comm_size, p_comm_size + num_ranks, displs.ptr(0), 0);

    static_cast<void>(comm::coll::collAlltoallv(
      size_send.ptr(0),
      comm_size.ptr(0),  // (num_segments_l+1)*size for all in sort group
      displs.ptr(0),     // exclusive_scan of comm_size
      size_recv.ptr(0),
      comm_size.ptr(0),  // (num_segments_l+1)*valuesize for all in sort group
      displs.ptr(0),     // exclusive_scan of comm_size
      comm::coll::CollDataType::CollInt,
      comm));

    comm_size.destroy();
    displs.destroy();
  }

  // copy values into send buffer
  auto val_send_buffer = create_non_empty_buffer<VAL>(volume);
  auto idc_send_buffer = create_non_empty_buffer<int64_t>(argsort ? volume : 0);
  auto* local_indices  = local_sorted.indices.ptr(0);

  // This line is particularly problematic, as the following line makes an out-of-bounds access when
  // teh buffer is empty
  auto positions = create_non_empty_buffer<int32_t>(num_sort_ranks);
  positions[0]   = 0;
  for (size_t sort_rank = 1; sort_rank < num_sort_ranks; ++sort_rank) {
    positions[sort_rank] =
      positions[sort_rank - 1] + size_send[(sort_rank - 1) * (num_segments_l + 1) + num_segments_l];
  }

  // fill send buffers
  {
    for (size_t segment = 0; segment < num_segments_l; ++segment) {
      for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
        int32_t start_position = segment_blocks[sort_rank * num_segments_l + segment];
        int32_t size           = size_send[sort_rank * (num_segments_l + 1) + segment];
        std::memcpy(val_send_buffer.ptr(0) + positions[sort_rank],
                    local_values + start_position,
                    size * sizeof(VAL));
        if (argsort) {
          std::memcpy(idc_send_buffer.ptr(0) + positions[sort_rank],
                      local_indices + start_position,
                      size * sizeof(int64_t));
        }
        positions[sort_rank] += size;
      }
    }
  }

  local_sorted.values.destroy();
  if (argsort) {
    local_sorted.indices.destroy();
  }
  segment_blocks.destroy();
  positions.destroy();

  // allocate target buffers
  SegmentMergePiece<VAL> merge_buffer;
  {
    int32_t total_receive = 0;
    for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
      total_receive += size_recv[sort_rank * (num_segments_l + 1) + num_segments_l];
    }

    merge_buffer.segments = create_non_empty_buffer<size_t>(num_segments_l > 1 ? total_receive : 0);
    if (num_segments_l > 1) {
      auto* p_segments = merge_buffer.segments.ptr(0);
      // initialize segment information
      int32_t start_pos = 0;
      for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
        for (size_t segment = 0; segment < num_segments_l; ++segment) {
          int32_t size = size_recv[sort_rank * (num_segments_l + 1) + segment];
          std::fill(p_segments + start_pos, p_segments + start_pos + size, segment);
          start_pos += size;
        }
      }
      assert(start_pos == total_receive);
    }

    merge_buffer.values  = create_non_empty_buffer<VAL>(total_receive);
    merge_buffer.indices = create_non_empty_buffer<int64_t>(argsort ? total_receive : 0);
    merge_buffer.size    = total_receive;
  }

  // communicate all2all (in sort dimension)
  {
    auto send_size_total = create_buffer<int32_t>(num_ranks);
    auto recv_size_total = create_buffer<int32_t>(num_ranks);
    std::fill(send_size_total.ptr(0), send_size_total.ptr(0) + num_ranks, 0);
    std::fill(recv_size_total.ptr(0), recv_size_total.ptr(0) + num_ranks, 0);

    auto sdispls = create_buffer<int32_t>(num_ranks);
    auto rdispls = create_buffer<int32_t>(num_ranks);

    for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
      send_size_total[sort_ranks[sort_rank]] =
        sizeof(VAL) * size_send[sort_rank * (num_segments_l + 1) + num_segments_l];
      recv_size_total[sort_ranks[sort_rank]] =
        sizeof(VAL) * size_recv[sort_rank * (num_segments_l + 1) + num_segments_l];
    }
    auto p_send_size_total = send_size_total.ptr(0);
    auto p_recv_size_total = recv_size_total.ptr(0);
    thrust::exclusive_scan(
      exec, p_send_size_total, p_send_size_total + num_ranks, sdispls.ptr(0), 0);
    thrust::exclusive_scan(
      exec, p_recv_size_total, p_recv_size_total + num_ranks, rdispls.ptr(0), 0);

    static_cast<void>(comm::coll::collAlltoallv(val_send_buffer.ptr(0),
                                                send_size_total.ptr(0),
                                                sdispls.ptr(0),
                                                merge_buffer.values.ptr(0),
                                                recv_size_total.ptr(0),
                                                rdispls.ptr(0),
                                                comm::coll::CollDataType::CollUint8,
                                                comm));

    if (argsort) {
      for (size_t sort_rank = 0; sort_rank < num_sort_ranks; ++sort_rank) {
        send_size_total[sort_ranks[sort_rank]] =
          size_send[sort_rank * (num_segments_l + 1) + num_segments_l];
        recv_size_total[sort_ranks[sort_rank]] =
          size_recv[sort_rank * (num_segments_l + 1) + num_segments_l];
      }

      thrust::exclusive_scan(
        exec, p_send_size_total, p_send_size_total + num_ranks, sdispls.ptr(0), 0);
      thrust::exclusive_scan(
        exec, p_recv_size_total, p_recv_size_total + num_ranks, rdispls.ptr(0), 0);
      static_cast<void>(comm::coll::collAlltoallv(idc_send_buffer.ptr(0),
                                                  send_size_total.ptr(0),
                                                  sdispls.ptr(0),
                                                  merge_buffer.indices.ptr(0),
                                                  recv_size_total.ptr(0),
                                                  rdispls.ptr(0),
                                                  comm::coll::CollDataType::CollInt64,
                                                  comm));
    }

    send_size_total.destroy();
    recv_size_total.destroy();
    sdispls.destroy();
    rdispls.destroy();
  }

  // cleanup remaining buffers
  size_send.destroy();
  size_recv.destroy();
  val_send_buffer.destroy();
  if (argsort) {
    idc_send_buffer.destroy();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 4: merge data
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // now merge sort all into the result buffer
  if (merge_buffer.size > 0) {
    // sort data (locally)
    if (num_segments_l == 1) {
      auto* p_values  = merge_buffer.values.ptr(0);
      auto* p_indices = argsort ? merge_buffer.indices.ptr(0) : nullptr;
      thrust_local_sort_inplace(
        p_values, p_indices, merge_buffer.size, merge_buffer.size, true, exec);
    } else {
      // we need to consider segments as well
      auto combined = thrust::make_zip_iterator(
        thrust::make_tuple(merge_buffer.segments.ptr(0), merge_buffer.values.ptr(0)));
      if (argsort) {
        auto* p_indices = merge_buffer.indices.ptr(0);
        thrust::stable_sort_by_key(exec,
                                   combined,
                                   combined + merge_buffer.size,
                                   p_indices,
                                   thrust::less<thrust::tuple<size_t, VAL>>());
      } else {
        thrust::stable_sort(
          exec, combined, combined + merge_buffer.size, thrust::less<thrust::tuple<size_t, VAL>>());
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////// Part 5: re-balance data to match input/output dimensions
  /////////////////////////////////////////////////////////////////////////////////////////////////

  if (rebalance) {
    assert(!is_unbound_1d_storage);
    rebalance_data(merge_buffer,
                   output_ptr,
                   my_rank,
                   num_ranks,
                   my_sort_rank,
                   num_sort_ranks,
                   sort_ranks,
                   segment_size_l,
                   num_segments_l,
                   argsort,
                   exec,
                   comm);
  } else {
    assert(is_unbound_1d_storage);
    merge_buffer.segments.destroy();
    if (argsort) {
      merge_buffer.values.destroy();
      output_array_unbound.bind_data(merge_buffer.indices, Point<1>(merge_buffer.size));
    } else {
      output_array_unbound.bind_data(merge_buffer.values, Point<1>(merge_buffer.size));
    }
  }
}

template <Type::Code CODE, int32_t DIM>
struct SortImplBodyCpu {
  using VAL = type_of<CODE>;

  template <typename DerivedPolicy>
  void operator()(legate::PhysicalStore input_array,
                  legate::PhysicalStore output_array,
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
                  const DerivedPolicy& exec,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input = input_array.read_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input.accessor.is_dense_row_major(rect));

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
      auto input_copy = create_buffer<VAL>(volume, legate::Memory::Kind::NO_MEMKIND, alignof(VAL));
      local_sorted.values = input_copy;
      values_ptr          = input_copy.ptr(0);

      // initialize indices
      if (need_distributed_sort) {
        auto indices_buffer =
          create_buffer<int64_t>(volume, legate::Memory::Kind::NO_MEMKIND, alignof(int64_t));
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
          thrust::sequence(exec, indices_ptr, indices_ptr + volume, offset);
        } else {
          thrust::transform(exec,
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
        auto input_copy =
          create_buffer<VAL>(volume, legate::Memory::Kind::NO_MEMKIND, alignof(VAL));
        values_ptr          = input_copy.ptr(0);
        local_sorted.values = input_copy;
        local_sorted.indices =
          create_buffer<int64_t>(0, legate::Memory::Kind::NO_MEMKIND, alignof(int64_t));
        local_sorted.size = volume;
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        assert(rect.empty() || output.accessor.is_dense_row_major(rect));
        values_ptr = output.ptr(rect.lo);
      }
    }

    if (volume > 0) {
      // sort data (locally)
      auto* src = input.ptr(rect.lo);
      if (src != values_ptr) {
        std::copy(src, src + volume, values_ptr);
      }
      thrust_local_sort_inplace(values_ptr, indices_ptr, volume, segment_size_l, stable, exec);
    }

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

        sample_sort_nd<CODE>(local_sorted,
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
                             exec,
                             comms[0].get<comm::coll::CollComm>());
      } else {
        // edge case where we have an unbound store but only 1 CPU was assigned with the task
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
  }
};

}  // namespace cupynumeric
