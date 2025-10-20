/* Copyright 2025 NVIDIA Corporation
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

#include <vector>

#include "legate/comm/coll.h"

// Useful for IDEs
#include "cupynumeric/matrix/mp_qr.h"
#include "cupynumeric/cuda_help.h"

#include "cupynumeric/utilities/thrust_util.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

// repartition logic:
//
#include "cupynumeric/utilities/repartition.h"

#include <algorithm>

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct MpQRImplBody;

namespace utils {
// copy from upper-triangular part of R (src_arr_ptr)
// into R store block (tgt_arr_ptr):
// where R block = Rect{offset_r, offset_c,
//                      extends{num_rows, num_cols}}
//
template <typename VAL>
void extract_by_col_above_diag(VAL* tgt_arr_ptr,
                               size_t tgt_volume,
                               size_t tgt_ld,
                               const VAL* src_arr_ptr,
                               size_t src_ld,
                               size_t offset_r,
                               size_t offset_c,
                               cudaStream_t stream)
{
  auto extractor = [=] __device__(auto counter) mutable {
    // counter col-major index of R
    auto rel_row_index = counter % tgt_ld;
    auto rel_col_index = counter / tgt_ld;

    bool is_upper = offset_r + rel_row_index <= offset_c + rel_col_index;

    // compute col major index of A
    auto src_indx_a = rel_col_index * src_ld + rel_row_index;

    return is_upper ? src_arr_ptr[src_indx_a] : VAL(0);
  };

  thrust::transform(DEFAULT_POLICY.on(stream),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(tgt_volume),
                    tgt_arr_ptr,
                    extractor);
}

// copy from square part of R (src_arr_ptr)
// into R store block (tgt_arr_ptr):
//
template <typename VAL>
void extract_by_col(VAL* tgt_arr_ptr,
                    size_t tgt_volume,
                    size_t tgt_ld,
                    const VAL* src_arr_ptr,
                    size_t src_ld,
                    cudaStream_t stream)
{
  auto extractor = [=] __device__(auto counter) mutable {
    // counter col-major index of R
    auto rel_row_index = counter % tgt_ld;
    auto rel_col_index = counter / tgt_ld;

    // compute col major index of A
    auto src_indx_a = rel_col_index * src_ld + rel_row_index;

    return src_arr_ptr[src_indx_a];
  };

  thrust::transform(DEFAULT_POLICY.on(stream),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(tgt_volume),
                    tgt_arr_ptr,
                    extractor);
}

// initialize Q block to I_{mxm}:
// where Q block = Rect{offset_r, offset_c,
//                      extends{num_rows, num_cols}}
//
template <typename VAL>
void init_eye_block(VAL* q_arr_ptr,
                    size_t m,
                    size_t n,
                    size_t offset_r,
                    size_t offset_c,
                    size_t num_rows,
                    size_t num_cols,
                    cudaStream_t stream)
{
  auto num_elems = num_rows * num_cols;

  auto extractor = [num_rows, offset_r, offset_c, m, n] __device__(auto counter) {
    auto rel_row_index = counter % num_rows;
    auto rel_col_index = counter / num_rows;

    if (offset_c + rel_col_index == offset_r + rel_row_index) {
      return VAL{1};
    } else {
      return VAL{0};
    }
  };

  thrust::transform(DEFAULT_POLICY.on(stream),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(num_elems),
                    q_arr_ptr,
                    extractor);
}
}  // namespace utils

template <Type::Code CODE>
struct support_mp_qr : std::false_type {};
template <>
struct support_mp_qr<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_mp_qr<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_mp_qr<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_mp_qr<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct MpQRImpl {
  TaskContext context;
  explicit MpQRImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE, std::enable_if_t<support_mp_qr<CODE>::value>* = nullptr>
  void operator()(int64_t m,
                  int64_t n,
                  int64_t mb,
                  int64_t nb,
                  legate::PhysicalStore a_array,
                  legate::PhysicalStore q_array,
                  legate::PhysicalStore r_array,
                  std::vector<comm::Communicator> comms,
                  const Domain& launch_domain) const
  {
    using VAL = type_of<CODE>;

    auto* p_nccl_comm = comms[0].get<ncclComm_t*>();
    int rank, num_ranks;
    assert(p_nccl_comm);
    auto nccl_comm = *p_nccl_comm;
    CHECK_NCCL(ncclCommUserRank(nccl_comm, &rank));
    CHECK_NCCL(ncclCommCount(nccl_comm, &num_ranks));

    auto stream = context.get_task_stream();

    assert(launch_domain.get_volume() == num_ranks);
    assert(launch_domain.get_dim() <= 2);
    assert(mb == nb);

    // enforce 2D shape for simplicity -- although b/x might be 1D
    auto a_shape = a_array.shape<2>();
    size_t a_strides[2];
    auto* a_arr =
      a_shape.empty() ? nullptr : a_array.read_accessor<VAL, 2>(a_shape).ptr(a_shape, a_strides);

    auto q_shape = q_array.shape<2>();
    size_t q_strides[2];
    auto* q_arr =
      q_shape.empty() ? nullptr : q_array.write_accessor<VAL, 2>(q_shape).ptr(q_shape, q_strides);

    auto r_shape = r_array.shape<2>();
    size_t r_strides[2];
    auto* r_arr =
      r_shape.empty() ? nullptr : r_array.write_accessor<VAL, 2>(r_shape).ptr(r_shape, r_strides);

    // assume col-major input
    auto llda = a_shape.empty() ? 1 : (a_shape.hi[0] - a_shape.lo[0] + 1);
    auto lldq = q_shape.empty() ? 1 : (q_shape.hi[0] - q_shape.lo[0] + 1);
    auto lldr = r_shape.empty() ? 1 : (r_shape.hi[0] - r_shape.lo[0] + 1);

    // the 2dbc process domain should go in both dimensions (8x1) -> (4x2)
    //
    size_t nprow = 1;
    size_t npcol = 1;
    if (m >= n) {
      nprow = num_ranks;
      while (npcol * 2 * m <= nprow * n && nprow % 2 == 0) {
        npcol *= 2;
        nprow /= 2;
      }
    } else {
      npcol = num_ranks;
      while (nprow * 2 * n <= npcol * m && npcol % 2 == 0) {
        nprow *= 2;
        npcol /= 2;
      }
    }

    assert(nprow * npcol == num_ranks);
    assert(n > 0 && m > 0 && mb > 0 && nb > 0 && llda > 0 && nprow > 0 && npcol > 0);

    // col-based in the mapper
    bool a_col_major = a_shape.empty() ||
                       a_array.read_accessor<VAL, 2>(a_shape).accessor.is_dense_col_major(a_shape);
    bool q_col_major = q_shape.empty() ||
                       q_array.write_accessor<VAL, 2>(q_shape).accessor.is_dense_col_major(q_shape);
    bool r_col_major = r_shape.empty() ||
                       r_array.write_accessor<VAL, 2>(r_shape).accessor.is_dense_col_major(r_shape);

    assert(a_col_major);
    assert(q_col_major);
    assert(r_col_major);

    auto a_offset_r = a_shape.lo[0];
    auto a_offset_c = a_shape.lo[1];
    auto a_volume   = a_shape.empty() ? 0 : llda * (a_shape.hi[1] - a_shape.lo[1] + 1);
    auto num_rows   = a_shape.hi[0] < a_shape.lo[0] ? 0 : a_shape.hi[0] - a_shape.lo[0] + 1;
    auto num_cols   = a_shape.hi[1] < a_shape.lo[1] ? 0 : a_shape.hi[1] - a_shape.lo[1] + 1;

    auto q_offset_r = q_shape.lo[0];
    auto q_offset_c = q_shape.lo[1];
    auto q_volume   = q_shape.empty() ? 0 : lldq * (q_shape.hi[1] - q_shape.lo[1] + 1);
    assert(q_shape.empty() || a_offset_r == q_offset_r);
    assert(q_shape.empty() || a_offset_c == q_offset_c);
    assert(m < n || q_volume == a_volume);
    assert(m < n || lldq == llda);

    auto r_offset_r = r_shape.lo[0];
    auto r_offset_c = r_shape.lo[1];
    auto r_volume   = r_shape.empty() ? 0 : lldr * (r_shape.hi[1] - r_shape.lo[1] + 1);
    assert(r_shape.empty() || a_offset_r == r_offset_r);
    assert(r_shape.empty() || a_offset_c == r_offset_c);
    assert(m > n || r_volume == a_volume);
    assert(m > n || lldr == llda);

    // repartition logic:
    //
    auto [a_buffer_2dbc, a_volume_2dbc, a_lld_2dbc] = repartition_matrix_2dbc(a_arr,
                                                                              a_volume,
                                                                              false,
                                                                              a_offset_r,
                                                                              a_offset_c,
                                                                              llda,
                                                                              nprow,
                                                                              npcol,
                                                                              mb,
                                                                              nb,
                                                                              comms[0],
                                                                              context);

    // initialize with ID
    if (!a_shape.empty()) {
      if (m >= n) {
        utils::init_eye_block(q_arr, m, n, q_offset_r, q_offset_c, num_rows, num_cols, stream);
      } else {
        // use R as tmp storage
        utils::init_eye_block(r_arr, m, n, r_offset_r, r_offset_c, num_rows, num_cols, stream);
      }
    }

    // choose temporary storage of same alignment as A
    auto [tmp_buffer_2dbc, tmp_volume_2dbc, tmp_lld_2dbc] =
      repartition_matrix_2dbc(m >= n ? q_arr : r_arr,
                              a_volume,
                              false,
                              a_offset_r,
                              a_offset_c,
                              llda,
                              nprow,
                              npcol,
                              mb,
                              nb,
                              comms[0],
                              context);

    assert(tmp_volume_2dbc == a_volume_2dbc);
    assert(tmp_lld_2dbc == a_lld_2dbc);

    MpQRImplBody<KIND, CODE>{context}(nccl_comm,
                                      nprow,
                                      npcol,
                                      m,
                                      n,
                                      mb,
                                      nb,
                                      a_buffer_2dbc.ptr(0),
                                      tmp_buffer_2dbc.ptr(0),
                                      a_volume_2dbc,
                                      a_lld_2dbc,
                                      rank);

    if (m >= n) {
      // M >= N: Q aligned to A
      // a_2dbc  ->  Q
      //   Q->R upper
      // tmp_2dbc  ->  Q
      repartition_matrix_block(a_buffer_2dbc,
                               a_volume_2dbc,
                               a_lld_2dbc,
                               rank,
                               nprow,
                               npcol,
                               mb,
                               nb,
                               q_arr,
                               q_volume,
                               lldq,
                               num_rows,
                               num_cols,
                               false,
                               q_offset_r,
                               q_offset_c,
                               comms[0],
                               context);

      // R.shape = (n, n), A.shape = (m, n)
      // -> the local R partition may be empty
      if (!r_shape.empty()) {
        utils::extract_by_col_above_diag(
          r_arr, r_volume, lldr, q_arr, lldq, r_offset_r, r_offset_c, stream);
      }

      repartition_matrix_block(tmp_buffer_2dbc,
                               tmp_volume_2dbc,
                               tmp_lld_2dbc,
                               rank,
                               nprow,
                               npcol,
                               mb,
                               nb,
                               q_arr,
                               q_volume,
                               lldq,
                               num_rows,
                               num_cols,
                               false,
                               q_offset_r,
                               q_offset_c,
                               comms[0],
                               context);
    } else {
      // M < N: R aligned to A
      // tmp_2dbc  ->  R
      // R -> Q copy
      // a_2dbc  ->  R
      // R->upper inplace
      repartition_matrix_block(tmp_buffer_2dbc,
                               tmp_volume_2dbc,
                               tmp_lld_2dbc,
                               rank,
                               nprow,
                               npcol,
                               mb,
                               nb,
                               r_arr,
                               r_volume,
                               lldr,
                               num_rows,
                               num_cols,
                               false,
                               r_offset_r,
                               r_offset_c,
                               comms[0],
                               context);

      // Q.shape = (n, n), A.shape = (m, n)
      // -> the local Q partition may be empty
      if (!q_shape.empty()) {
        utils::extract_by_col(q_arr, q_volume, lldq, r_arr, lldr, stream);
      }

      repartition_matrix_block(a_buffer_2dbc,
                               a_volume_2dbc,
                               a_lld_2dbc,
                               rank,
                               nprow,
                               npcol,
                               mb,
                               nb,
                               r_arr,
                               r_volume,
                               lldr,
                               num_rows,
                               num_cols,
                               false,
                               r_offset_r,
                               r_offset_c,
                               comms[0],
                               context);

      // Zero out lower diagonal of R in-place
      utils::extract_by_col_above_diag(
        r_arr, r_volume, lldr, r_arr, lldr, r_offset_r, r_offset_c, stream);
    }
  }

  template <Type::Code CODE, std::enable_if_t<!support_mp_qr<CODE>::value>* = nullptr>
  void operator()(int64_t m,
                  int64_t n,
                  int64_t mb,
                  int64_t nb,
                  legate::PhysicalStore a_array,
                  legate::PhysicalStore q_array,
                  legate::PhysicalStore r_array,
                  std::vector<comm::Communicator> comms,
                  const Domain& launch_domain) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void mp_qr_template(TaskContext& context)
{
  legate::PhysicalStore a_array = context.input(0);
  legate::PhysicalStore q_array = context.output(0);
  legate::PhysicalStore r_array = context.output(1);
  auto m                        = context.scalar(0).value<int64_t>();
  auto n                        = context.scalar(1).value<int64_t>();
  auto mb                       = context.scalar(2).value<int64_t>();
  auto nb                       = context.scalar(3).value<int64_t>();
  type_dispatch(a_array.code(),
                MpQRImpl<KIND>{context},
                m,
                n,
                mb,
                nb,
                a_array,
                q_array,
                r_array,
                context.communicators(),
                context.get_launch_domain());
}

}  // namespace cupynumeric
