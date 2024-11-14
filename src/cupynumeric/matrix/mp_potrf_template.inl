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
#include "cupynumeric/matrix/mp_potrf.h"
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/utilities/repartition.h"

#include <cal.h>

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct MpPotrfImplBody;

template <Type::Code CODE>
struct support_mp_potrf : std::false_type {};
template <>
struct support_mp_potrf<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_mp_potrf<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_mp_potrf<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_mp_potrf<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct MpPotrfImpl {
  template <Type::Code CODE, std::enable_if_t<support_mp_potrf<CODE>::value>* = nullptr>
  void operator()(int64_t n,
                  int64_t nb,
                  legate::PhysicalStore input_array,
                  legate::PhysicalStore output_array,
                  std::vector<comm::Communicator> comms,
                  const Domain& launch_domain) const
  {
    using VAL = type_of<CODE>;

    auto input_shape  = input_array.shape<2>();
    auto output_shape = output_array.shape<2>();

    int rank, num_ranks;
    auto nccl_comm = comms[0];
    auto cal_comm  = comms[1].get<cal_comm_t>();
    assert(cal_comm);
    CHECK_CAL(cal_comm_get_rank(cal_comm, &rank));
    CHECK_CAL(cal_comm_get_size(cal_comm, &num_ranks));
    assert(launch_domain.get_volume() == num_ranks);
    assert(launch_domain.get_dim() <= 2);

    assert(input_shape == output_shape);

    bool input_col_major =
      input_shape.empty() ||
      input_array.read_accessor<VAL, 2>(input_shape).accessor.is_dense_col_major(input_shape);
    bool output_col_major =
      output_shape.empty() ||
      output_array.write_accessor<VAL, 2>(output_shape).accessor.is_dense_col_major(output_shape);
    assert(input_col_major);
    assert(output_col_major);

    size_t strides[2];

    auto input_arr = input_shape.empty()
                       ? nullptr
                       : input_array.read_accessor<VAL, 2>(input_shape).ptr(input_shape, strides);
    auto output_arr =
      output_shape.empty()
        ? nullptr
        : output_array.write_accessor<VAL, 2>(output_shape).ptr(output_shape, strides);
    auto num_rows =
      input_shape.hi[0] < input_shape.lo[0] ? 0 : input_shape.hi[0] - input_shape.lo[0] + 1;
    auto num_cols =
      input_shape.hi[1] < input_shape.lo[1] ? 0 : input_shape.hi[1] - input_shape.lo[1] + 1;
    auto lld = input_shape.empty() ? 1 : (input_col_major ? num_rows : num_cols);

    // the 2dbc process domain should go in both dimensions (8x1) -> (4x2)
    size_t nprow = num_ranks;
    size_t npcol = 1;
    while (npcol * 2 <= nprow && nprow % 2 == 0) {
      npcol *= 2;
      nprow /= 2;
    }

    assert(nprow * npcol == num_ranks);
    assert(n > 0 && nb > 0 && lld > 0 && nprow > 0 && npcol > 0);

    auto offset_r = input_shape.lo[0];
    auto offset_c = input_shape.lo[1];
    auto volume   = num_rows * num_cols;

    auto [buffer_2dbc, volume_2dbc, lld_2dbc] = repartition_matrix_2dbc(
      input_arr, volume, false, offset_r, offset_c, lld, nprow, npcol, nb, nb, nccl_comm);

    MpPotrfImplBody<KIND, CODE>()(cal_comm, nprow, npcol, n, nb, buffer_2dbc.ptr(0), lld_2dbc);

    repartition_matrix_block(buffer_2dbc,
                             volume_2dbc,
                             lld_2dbc,
                             rank,
                             nprow,
                             npcol,
                             nb,
                             nb,
                             output_arr,
                             volume,
                             lld,
                             num_rows,
                             num_cols,
                             false,
                             offset_r,
                             offset_c,
                             nccl_comm);
  }

  template <Type::Code CODE, std::enable_if_t<!support_mp_potrf<CODE>::value>* = nullptr>
  void operator()(int64_t n,
                  int64_t nb,
                  legate::PhysicalStore input_array,
                  legate::PhysicalStore output_array,
                  std::vector<comm::Communicator> comms,
                  const Domain& launch_domain) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void mp_potrf_template(TaskContext& context)
{
  legate::PhysicalStore input_array  = context.input(0);
  legate::PhysicalStore output_array = context.output(0);
  auto n                             = context.scalar(0).value<int64_t>();
  auto nb                            = context.scalar(1).value<int64_t>();
  type_dispatch(input_array.code(),
                MpPotrfImpl<KIND>{},
                n,
                nb,
                input_array,
                output_array,
                context.communicators(),
                context.get_launch_domain());
}

}  // namespace cupynumeric
