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

#include <vector>

#include "legate/comm/coll.h"

// Useful for IDEs
#include "cupynumeric/matrix/mp_solve.h"
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/utilities/repartition.h"

#include <cal.h>

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct MpSolveImplBody;

template <Type::Code CODE>
struct support_mp_solve : std::false_type {};
template <>
struct support_mp_solve<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_mp_solve<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_mp_solve<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_mp_solve<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct MpSolveImpl {
  template <Type::Code CODE, std::enable_if_t<support_mp_solve<CODE>::value>* = nullptr>
  void operator()(int64_t n,
                  int64_t nrhs,
                  int64_t nb,
                  legate::PhysicalStore a_array,
                  legate::PhysicalStore b_array,
                  legate::PhysicalStore x_array,
                  std::vector<comm::Communicator> comms,
                  const Domain& launch_domain) const
  {
    using VAL = type_of<CODE>;

    auto nccl_comm = comms[0];
    auto cal_comm  = comms[1].get<cal_comm_t>();

    int rank, num_ranks;
    assert(cal_comm);
    CHECK_CAL(cal_comm_get_rank(cal_comm, &rank));
    CHECK_CAL(cal_comm_get_size(cal_comm, &num_ranks));
    assert(launch_domain.get_volume() == num_ranks);
    assert(launch_domain.get_dim() <= 2);

    // enforce 2D shape for simplicity -- although b/x might be 1D
    auto a_shape = a_array.shape<2>();
    auto b_shape = b_array.shape<2>();
    auto x_shape = x_array.shape<2>();

    assert(x_shape == b_shape);

    auto a_arr = a_shape.empty() ? nullptr : a_array.read_accessor<VAL, 2>(a_shape).ptr(a_shape.lo);
    auto b_arr = b_shape.empty() ? nullptr : b_array.read_accessor<VAL, 2>(b_shape).ptr(b_shape.lo);
    auto x_arr =
      x_shape.empty() ? nullptr : x_array.write_accessor<VAL, 2>(x_shape).ptr(x_shape.lo);

    // assume col-major input
    auto llda = a_shape.empty() ? 1 : (a_shape.hi[0] - a_shape.lo[0] + 1);
    auto lldb = b_shape.empty() ? 1 : (b_shape.hi[0] - b_shape.lo[0] + 1);

    // the 2dbc process domain should go in both dimensions (8x1) -> (4x2)
    size_t nprow = num_ranks;
    size_t npcol = 1;
    while (npcol * 2 <= nprow && nprow % 2 == 0) {
      npcol *= 2;
      nprow /= 2;
    }

    assert(nprow * npcol == num_ranks);
    assert(n > 0 && nrhs > 0 && nb > 0 && llda > 0 && lldb > 0 && nprow > 0 && npcol > 0);

    // the 2dbc conversion seems to have issues for row-wise ordered data, therefore we enforce
    // col-based in the mapper
    bool a_col_major = a_shape.empty() ||
                       a_array.read_accessor<VAL, 2>(a_shape).accessor.is_dense_col_major(a_shape);
    bool b_col_major = b_shape.empty() ||
                       b_array.read_accessor<VAL, 2>(b_shape).accessor.is_dense_col_major(b_shape);
    bool x_col_major = x_shape.empty() ||
                       x_array.write_accessor<VAL, 2>(x_shape).accessor.is_dense_col_major(x_shape);

    assert(a_col_major);
    assert(b_col_major);
    assert(x_col_major);

    auto a_offset_r = a_shape.lo[0];
    auto a_offset_c = a_shape.lo[1];
    auto a_volume   = a_shape.empty() ? 0 : llda * (a_shape.hi[1] - a_shape.lo[1] + 1);

    auto [a_buffer_2dbc, a_volume_2dbc, a_lld_2dbc] = repartition_matrix_2dbc(
      a_arr, a_volume, false, a_offset_r, a_offset_c, llda, nprow, npcol, nb, nb, nccl_comm);

    auto b_offset_r = b_shape.lo[0];
    auto b_offset_c = b_shape.lo[1];
    auto b_volume   = b_shape.empty() ? 0 : lldb * (b_shape.hi[1] - b_shape.lo[1] + 1);

    auto [b_buffer_2dbc, b_volume_2dbc, b_lld_2dbc] = repartition_matrix_2dbc(
      b_arr, b_volume, false, b_offset_r, b_offset_c, lldb, nprow, npcol, nb, nb, nccl_comm);

    MpSolveImplBody<KIND, CODE>()(cal_comm,
                                  nprow,
                                  npcol,
                                  n,
                                  nrhs,
                                  nb,
                                  a_buffer_2dbc.ptr(0),
                                  a_lld_2dbc,
                                  b_buffer_2dbc.ptr(0),
                                  b_lld_2dbc);

    auto b_num_rows = b_shape.hi[0] < b_shape.lo[0] ? 0 : b_shape.hi[0] - b_shape.lo[0] + 1;
    auto b_num_cols = b_shape.hi[1] < b_shape.lo[1] ? 0 : b_shape.hi[1] - b_shape.lo[1] + 1;

    repartition_matrix_block(b_buffer_2dbc,
                             b_volume_2dbc,
                             b_lld_2dbc,
                             rank,
                             nprow,
                             npcol,
                             nb,
                             nb,
                             x_arr,
                             b_volume,
                             lldb,
                             b_num_rows,
                             b_num_cols,
                             false,  // x_shape is enforced col-major
                             b_offset_r,
                             b_offset_c,
                             nccl_comm);
  }

  template <Type::Code CODE, std::enable_if_t<!support_mp_solve<CODE>::value>* = nullptr>
  void operator()(int64_t n,
                  int64_t nrhs,
                  int64_t nb,
                  legate::PhysicalStore a_array,
                  legate::PhysicalStore b_array,
                  legate::PhysicalStore x_array,
                  std::vector<comm::Communicator> comms,
                  const Domain& launch_domain) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void mp_solve_template(TaskContext& context)
{
  legate::PhysicalStore a_array = context.input(0);
  legate::PhysicalStore b_array = context.input(1);
  legate::PhysicalStore x_array = context.output(0);
  auto n                        = context.scalar(0).value<int64_t>();
  auto nrhs                     = context.scalar(1).value<int64_t>();
  auto nb                       = context.scalar(2).value<int64_t>();
  type_dispatch(a_array.code(),
                MpSolveImpl<KIND>{},
                n,
                nrhs,
                nb,
                a_array,
                b_array,
                x_array,
                context.communicators(),
                context.get_launch_domain());
}

}  // namespace cupynumeric
