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

namespace cupynumeric {

std::tuple<size_t, size_t> elements_for_rank_in_dimension(
  size_t dim_length, size_t offset_id, size_t proc_id, size_t num_dim_procs, size_t tilesize)
{
  size_t start_tile_idx     = offset_id / tilesize;
  size_t start_tile_proc_id = start_tile_idx % num_dim_procs;
  size_t start_pos_offset   = proc_id >= start_tile_proc_id
                                ? (proc_id - start_tile_proc_id) * tilesize
                                : (num_dim_procs + proc_id - start_tile_proc_id) * tilesize;
  size_t start_tile_offset  = offset_id % tilesize;

  if (start_tile_offset > 0 && start_pos_offset > 0) {
    // we can move the start position left to the start of the tile
    start_pos_offset -= start_tile_offset;
  }

  // calc global offset for procId
  size_t offset_tiles = (start_tile_idx + num_dim_procs - proc_id - 1) / num_dim_procs;

  if (start_pos_offset > dim_length) {
    return {0ul, offset_tiles};
  }

  size_t full_cycles  = (dim_length - start_pos_offset) / (tilesize * num_dim_procs);
  size_t num_elements = full_cycles * tilesize;
  size_t remainder    = dim_length - start_pos_offset - num_elements * num_dim_procs;
  if (start_pos_offset > 0 || start_tile_offset == 0) {
    // we have a clean start
    if (remainder > 0) {
      num_elements += std::min(tilesize, remainder);
    }
  } else {
    // we start with a partial tile
    size_t tile_remainder = tilesize - start_tile_offset;
    if (remainder <= tile_remainder) {
      num_elements += remainder;
    } else {
      remainder -= tile_remainder;
      num_elements += tile_remainder;
      if (remainder > (num_dim_procs - 1) * tilesize) {
        num_elements += std::min(tilesize, remainder - (num_dim_procs - 1) * tilesize);
      }
    }
  }

  size_t offset_elements =
    offset_tiles * tilesize + (start_pos_offset == 0 ? start_tile_offset : 0);

  return {num_elements, offset_elements};
}

}  // namespace cupynumeric
