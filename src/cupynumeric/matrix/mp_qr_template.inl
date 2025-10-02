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
// copy from upper-triangular part of R (a_arr_ptr)
// into R store block (r_arr_ptr):
// where R block = Rect{offset_r, offset_c,
//                      extends{num_rows, num_cols}}
//
template <typename VAL>
void extract_by_col_above_diag(VAL* r_arr_ptr,
                               size_t r_volume,
                               size_t ldr,
                               const VAL* a_arr_ptr,
                               size_t lda,
                               size_t offset_r,
                               size_t offset_c,
                               cudaStream_t stream)
{
  auto extractor = [=] __device__(auto counter) mutable {
    // counter col-major index of R
    auto rel_row_index = counter % ldr;
    auto rel_col_index = counter / ldr;

    bool is_upper = offset_r + rel_row_index <= offset_c + rel_col_index;

    // compute col major index of A
    auto src_indx_a = rel_col_index * lda + rel_row_index;

    return is_upper ? a_arr_ptr[src_indx_a] : VAL(0);
  };

  thrust::transform(DEFAULT_POLICY.on(stream),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(r_volume),
                    r_arr_ptr,
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

    // FIXME (separate PR): implement support for m < n
    assert(m >= n);

    auto stream = context.get_task_stream();

    assert(launch_domain.get_volume() == num_ranks);
    assert(launch_domain.get_dim() <= 2);

    // enforce 2D shape for simplicity -- although b/x might be 1D
    auto a_shape = a_array.shape<2>();
    size_t a_strides[2];
    auto* a_arr =
      a_shape.empty() ? nullptr : a_array.read_accessor<VAL, 2>(a_shape).ptr(a_shape, a_strides);

    auto q_shape = q_array.shape<2>();
    size_t q_strides[2];
    auto* q_arr =
      q_shape.empty() ? nullptr : q_array.write_accessor<VAL, 2>(q_shape).ptr(q_shape, q_strides);

    // assume col-major input
    auto llda = a_shape.empty() ? 1 : (a_shape.hi[0] - a_shape.lo[0] + 1);
    auto lldq = q_shape.empty() ? 1 : (q_shape.hi[0] - q_shape.lo[0] + 1);

    // the 2dbc process domain should go in both dimensions (8x1) -> (4x2)
    //
    size_t nprow = num_ranks;
    size_t npcol = 1;
    while (npcol * 2 * m <= nprow * n && nprow % 2 == 0) {
      npcol *= 2;
      nprow /= 2;
    }

    assert(nprow * npcol == num_ranks);
    assert(n > 0 && m > 0 && mb > 0 && nb > 0 && llda > 0 && nprow > 0 && npcol > 0);

    // col-based in the mapper
    bool a_col_major = a_shape.empty() ||
                       a_array.read_accessor<VAL, 2>(a_shape).accessor.is_dense_col_major(a_shape);
    bool q_col_major = q_shape.empty() ||
                       q_array.write_accessor<VAL, 2>(q_shape).accessor.is_dense_col_major(q_shape);

    assert(a_col_major);
    assert(q_col_major);

    auto a_offset_r = a_shape.lo[0];
    auto a_offset_c = a_shape.lo[1];
    auto a_volume   = a_shape.empty() ? 0 : llda * (a_shape.hi[1] - a_shape.lo[1] + 1);

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

    auto q_offset_r = q_shape.lo[0];
    auto q_offset_c = q_shape.lo[1];
    auto q_volume   = q_shape.empty() ? 0 : lldq * (q_shape.hi[1] - q_shape.lo[1] + 1);

    if (!q_shape.empty()) {
      auto q_num_rows = q_shape.hi[0] < q_shape.lo[0] ? 0 : q_shape.hi[0] - q_shape.lo[0] + 1;
      auto q_num_cols = q_shape.hi[1] < q_shape.lo[1] ? 0 : q_shape.hi[1] - q_shape.lo[1] + 1;

      utils::init_eye_block(q_arr, m, n, q_offset_r, q_offset_c, q_num_rows, q_num_cols, stream);
    }
    auto [q_buffer_2dbc, q_volume_2dbc, q_lld_2dbc] = repartition_matrix_2dbc(q_arr,
                                                                              q_volume,
                                                                              false,
                                                                              q_offset_r,
                                                                              q_offset_c,
                                                                              lldq,
                                                                              nprow,
                                                                              npcol,
                                                                              mb,
                                                                              nb,
                                                                              comms[0],
                                                                              context);

    assert(q_volume_2dbc == a_volume_2dbc);
    assert(q_lld_2dbc == a_lld_2dbc);

    MpQRImplBody<KIND, CODE>{context}(nccl_comm,
                                      nprow,
                                      npcol,
                                      m,
                                      n,
                                      mb,
                                      nb,
                                      a_buffer_2dbc.ptr(0),
                                      q_buffer_2dbc.ptr(0),
                                      a_volume_2dbc,
                                      a_lld_2dbc,
                                      q_lld_2dbc,
                                      rank);

    // re-tile A:
    //
    auto num_rows  = a_shape.hi[0] < a_shape.lo[0] ? 0 : a_shape.hi[0] - a_shape.lo[0] + 1;
    auto num_cols  = a_shape.hi[1] < a_shape.lo[1] ? 0 : a_shape.hi[1] - a_shape.lo[1] + 1;
    auto a_arr_tmp = create_buffer<VAL>(
      num_rows * num_cols,
      Memory::Kind::GPU_FB_MEM);  // need a temp copy, a_arr_tmp, of shape = (num_rows, num_cols))

    repartition_matrix_block(a_buffer_2dbc,
                             a_volume_2dbc,
                             a_lld_2dbc,
                             rank,
                             nprow,
                             npcol,
                             mb,
                             nb,
                             a_arr_tmp.ptr(0),  // TODO: (1) pass R store, directly (?)
                             a_volume,
                             llda,
                             num_rows,
                             num_cols,
                             false,
                             a_offset_r,
                             a_offset_c,
                             comms[0],
                             context);

    // extract upper-triangular block from A into R:
    //
    auto r_shape = r_array.shape<2>();
    size_t r_strides[2];
    auto* r_arr =
      r_shape.empty() ? nullptr : r_array.write_accessor<VAL, 2>(r_shape).ptr(r_shape, r_strides);
    bool r_col_major = r_shape.empty() ||
                       r_array.write_accessor<VAL, 2>(r_shape).accessor.is_dense_col_major(r_shape);
    assert(r_col_major);
    auto r_volume = r_shape.empty()
                      ? 0
                      : (r_shape.hi[0] - r_shape.lo[0] + 1) * (r_shape.hi[1] - r_shape.lo[1] + 1);

    // R.shape = (n, n), A.shape = (m, n)
    // but the local R partition may be empty for most PEs
    //
    if (!r_shape.empty()) {
      auto ldr = r_shape.hi[0] - r_shape.lo[0] + 1;
      utils::extract_by_col_above_diag(
        r_arr, r_volume, ldr, a_arr_tmp.ptr(0), llda, a_offset_r, a_offset_c, stream);
    }

    num_rows = q_shape.hi[0] < q_shape.lo[0] ? 0 : q_shape.hi[0] - q_shape.lo[0] + 1;
    num_cols = q_shape.hi[1] < q_shape.lo[1] ? 0 : q_shape.hi[1] - q_shape.lo[1] + 1;

    // re-tile Q:
    //
    repartition_matrix_block(q_buffer_2dbc,
                             q_volume_2dbc,
                             q_lld_2dbc,
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
