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

#include "legate.h"
#include "cupynumeric/cupynumeric_task.h"

namespace cupynumeric {

enum BlockInfo {
  TOTAL_SIZE,  // # values send
  LLD,         // local leading dimension ( num rows for col-based)
  OFFSET_ROW,  // global row offset w.r.t. elements of proc id
  OFFSET_COL,  // global col offset w.r.t. elements of proc id
  LAST         // keep as last element
};

// TODO(mförster) optimize -- we would like to provide a global mapping to skip additional
// communication

/*
 * performs collective repartition to 2d block cyclic pattern
 * returns tuple(buffer, volume, lld)
 */
template <typename VAL>
[[nodiscard]] std::tuple<legate::Buffer<VAL>, size_t, size_t> repartition_matrix_2dbc(
  // dense input data block (only GPU mem supported)
  const VAL* input,
  size_t volume,
  bool row_major,
  // offset of local block w.r.t. global dimensions
  size_t offset_r,
  size_t offset_c,
  // lld of input data, corresponds to numRows/numCols
  size_t lld,
  // target process grid layout (p_r*p_c need to match communicator size)
  size_t p_r,
  size_t p_c,
  // tile layout
  size_t tile_r,
  size_t tile_c,
  // communicator
  legate::comm::Communicator comm,
  legate::TaskContext context);

/*
 * performs collective repartition from 2d block cyclic pattern
 * back to block
 */
template <typename VAL>
void repartition_matrix_block(
  // dense input data block (only GPU mem supported)
  // will be released as soon as consumed
  legate::Buffer<VAL> input_2dbc,
  size_t input_volume,
  size_t input_lld,
  // should match NCCL rank and 2dbc ID column major
  size_t local_rank,
  // 2dbc process grid layout (p_r*p_c need to match communicator size)
  size_t p_r,
  size_t p_c,
  // tile layout
  size_t tile_r,
  size_t tile_c,
  // dense output data pointer (only GPU mem supported)
  VAL* target,
  size_t target_volume,
  size_t target_lld,
  // cuPyNumeric process grid layout (needs to match communicator size)
  size_t num_target_rows,
  size_t num_target_cols,
  bool target_row_major,
  // offset of local block w.r.t. global dimensions
  size_t target_offset_r,
  size_t target_offset_c,
  // communicator
  legate::comm::Communicator comm,
  legate::TaskContext context);

[[nodiscard]] std::tuple<size_t, size_t> elements_for_rank_in_dimension(
  size_t dim_length, size_t offset_id, size_t proc_id, size_t num_dim_procs, size_t tilesize);

}  // namespace cupynumeric
