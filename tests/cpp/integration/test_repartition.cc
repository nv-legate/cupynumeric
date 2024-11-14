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

#include <gtest/gtest.h>
#include <ostream>
#include <numeric>
#include "legate.h"
#include "cupynumeric.h"
#include "util.inl"
#include "cupynumeric/utilities/repartition.h"

namespace repartition_test {

constexpr const char* library_name = "test_repartition";

constexpr bool debug = false;

enum TaskIDs {
  CHECK_REPARTITION_TASK = 0,
};

template <bool I_ROW_MAJOR, bool O_ROW_MAJOR>
struct CheckRepartitionTask
  : public legate::LegateTask<CheckRepartitionTask<I_ROW_MAJOR, O_ROW_MAJOR>> {
  static constexpr auto TASK_ID =
    legate::LocalTaskID{CHECK_REPARTITION_TASK + I_ROW_MAJOR * 2 + O_ROW_MAJOR};

  static void gpu_variant(legate::TaskContext context);
};

class RepartitionLayoutMapper : public legate::mapping::Mapper {
  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& /*task*/,
    const std::vector<legate::mapping::TaskTarget>& options) override
  {
    return options.front();
  }
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    auto task_id       = static_cast<std::int64_t>(task.task_id());
    bool out_row_major = task_id % 2 == 1;
    bool in_row_major  = task_id > 1;

    std::vector<legate::mapping::StoreMapping> mappings;
    auto inputs  = task.inputs();
    auto outputs = task.outputs();
    for (auto& input : inputs) {
      mappings.push_back(legate::mapping::StoreMapping::default_mapping(
        input.data(), options.front(), true /*exact*/));
      if (in_row_major) {
        mappings.back().policy().ordering.set_c_order();
      } else {
        mappings.back().policy().ordering.set_fortran_order();
      }
    }
    for (auto& output : outputs) {
      mappings.push_back(legate::mapping::StoreMapping::default_mapping(
        output.data(), options.front(), true /*exact*/));
      if (out_row_major) {
        mappings.back().policy().ordering.set_c_order();
      } else {
        mappings.back().policy().ordering.set_fortran_order();
      }
    }
    return mappings;
  }
  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }
};

int get_rank_row_major(legate::Domain domain, legate::DomainPoint index_point)
{
  int domain_index = 0;
  auto hi          = domain.hi();
  auto lo          = domain.lo();
  for (int i = 0; i < domain.get_dim(); ++i) {
    if (i > 0) {
      domain_index *= hi[i] - lo[i] + 1;
    }
    domain_index += index_point[i];
  }
  return domain_index;
}

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
void repartition_2dbc_test(legate::AccessorRO<int32_t, 2> input,
                           legate::Rect<2> in_rect,
                           bool in_row_major,
                           legate::AccessorWO<int32_t, 2> output,
                           legate::Rect<2> out_rect,
                           bool out_row_major,
                           int32_t proc_r,
                           int32_t proc_c,
                           int32_t tile_r,
                           int32_t tile_c,
                           int32_t local_rank,
                           legate::comm::Communicator comm)
{
  const int32_t* input_ptr = input.ptr(in_rect.lo);
  size_t input_volume      = in_rect.volume();
  size_t input_offset_r    = in_rect.lo[0];
  size_t input_offset_c    = in_rect.lo[1];
  size_t input_lld =
    in_rect.empty() ? 1 : (in_rect.hi[in_row_major ? 1 : 0] - in_rect.lo[in_row_major ? 1 : 0] + 1);

  auto [buffer_2dbc, volume_2dbc, lld_2dbc] = cupynumeric::repartition_matrix_2dbc(input_ptr,
                                                                                   input_volume,
                                                                                   in_row_major,
                                                                                   input_offset_r,
                                                                                   input_offset_c,
                                                                                   input_lld,
                                                                                   proc_r,
                                                                                   proc_c,
                                                                                   tile_r,
                                                                                   tile_c,
                                                                                   comm);

  int32_t* output_ptr    = output.ptr(out_rect.lo);
  size_t output_volume   = out_rect.volume();
  size_t output_offset_r = out_rect.lo[0];
  size_t output_offset_c = out_rect.lo[1];
  size_t num_rows   = out_rect.hi[0] < out_rect.lo[0] ? 0 : out_rect.hi[0] - out_rect.lo[0] + 1;
  size_t num_cols   = out_rect.hi[1] < out_rect.lo[1] ? 0 : out_rect.hi[1] - out_rect.lo[1] + 1;
  size_t output_lld = out_rect.empty() ? 1 : (out_row_major ? num_cols : num_rows);

  if (debug) {
    std::ostringstream stringStream;
    stringStream << "DEBUG: volume_2dbc = " << volume_2dbc << ", lld_2dbc = " << lld_2dbc
                 << ", in_row_major = " << in_row_major << ", out_row_major = " << out_row_major
                 << ", num_rows = " << num_rows << ", num_cols = " << num_cols
                 << ", output_offset_r = " << output_offset_r
                 << ", output_offset_c = " << output_offset_c << ", output_lld = " << output_lld
                 << ", rank = " << local_rank << std::endl;
    std::cerr << stringStream.str();
  }

  cupynumeric::repartition_matrix_block(buffer_2dbc,
                                        volume_2dbc,
                                        lld_2dbc,
                                        local_rank,
                                        proc_r,
                                        proc_c,
                                        tile_r,
                                        tile_c,
                                        output_ptr,
                                        output_volume,
                                        output_lld,
                                        num_rows,
                                        num_cols,
                                        out_row_major,
                                        output_offset_r,
                                        output_offset_c,
                                        comm);
}
#endif

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(
    library_name, legate::ResourceConfig{}, std::make_unique<RepartitionLayoutMapper>());

  CheckRepartitionTask<false, false>::register_variants(library);
  CheckRepartitionTask<true, true>::register_variants(library);
  CheckRepartitionTask<false, true>::register_variants(library);
  CheckRepartitionTask<true, false>::register_variants(library);
}

template <bool I_ROW_MAJOR, bool O_ROW_MAJOR>
/*static*/ void CheckRepartitionTask<I_ROW_MAJOR, O_ROW_MAJOR>::gpu_variant(
  legate::TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  auto input     = context.input(0);
  auto output    = context.output(0);
  auto shape_in  = input.shape<2>();
  auto shape_out = output.shape<2>();

  size_t tile_r = context.scalar(0).value<size_t>();
  size_t tile_c = context.scalar(1).value<size_t>();

  auto total_ranks = context.get_launch_domain().get_volume();
  auto local_rank  = get_rank_row_major(context.get_launch_domain(), context.get_task_index());

  if (total_ranks == 1) {
    std::cerr << "Error: aborting due to single task launch. Ensure LEGATE_TEST=1 to force "
                 "parallel execution for small test dimensions."
              << std::endl;
    return;
  }

  int32_t pr = total_ranks;
  int32_t pc = 1;
  while (pc * 2 <= pr && pr % 2 == 0) {
    pr /= 2;
    pc *= 2;
  }

  auto input_acc  = input.data().read_accessor<int32_t, 2>(shape_in);
  auto output_acc = output.data().write_accessor<int32_t, 2>(shape_out);

  bool in_row_major  = shape_in.empty() || input_acc.accessor.is_dense_row_major(shape_in);
  bool in_col_major  = shape_in.empty() || input_acc.accessor.is_dense_col_major(shape_in);
  bool out_row_major = shape_out.empty() || output_acc.accessor.is_dense_row_major(shape_out);
  bool out_col_major = shape_out.empty() || output_acc.accessor.is_dense_col_major(shape_out);

  if (debug) {
    std::ostringstream stringStream;
    stringStream << "DEBUG: Domain = " << context.get_launch_domain()
                 << ", index = " << context.get_task_index() << ", I_ROW_MAJOR = " << I_ROW_MAJOR
                 << ", O_ROW_MAJOR = " << O_ROW_MAJOR << ", shape_in = " << shape_in
                 << "(order=" << in_row_major << "," << in_col_major << ")"
                 << ", shape_out = " << shape_out << "(order=" << out_row_major << ", "
                 << out_col_major << ")"
                 << ", communicators = " << context.num_communicators() << ", rank = " << local_rank
                 << ", tile = (" << tile_r << "," << tile_c << ")"
                 << ", procs_2dbc = (" << pr << "," << pc << ")" << std::endl;
    std::cerr << stringStream.str();
  }

  EXPECT_EQ(true, I_ROW_MAJOR ? in_row_major : in_col_major);
  EXPECT_EQ(true, O_ROW_MAJOR ? out_row_major : out_col_major);

  repartition_2dbc_test(input_acc,
                        shape_in,
                        I_ROW_MAJOR,
                        output_acc,
                        shape_out,
                        O_ROW_MAJOR,
                        pr,
                        pc,
                        tile_r,
                        tile_c,
                        local_rank,
                        context.communicator(0));
#endif
}

template <bool I_ROW_MAJOR, bool O_ROW_MAJOR>
void run_test_aligned_default_launch(std::vector<uint64_t>& data_shape,
                                     std::vector<uint64_t>& tile_shape)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto library  = runtime->find_library(library_name);
  auto machine  = runtime->get_machine();
  auto num_gpus = machine.count(legate::mapping::TaskTarget::GPU);
  if (num_gpus < 2) {
    GTEST_SKIP();
  }

  // generate data
  size_t volume    = data_shape[0] * data_shape[1];
  auto data_input  = cupynumeric::zeros(data_shape, legate::int32());
  auto data_output = cupynumeric::zeros(data_shape, legate::int32());
  if (volume != 0) {
    if (volume == 1) {
      data_input.fill(legate::Scalar(0));
    } else {
      std::vector<int32_t> numbers(volume);
      std::iota(numbers.data(), numbers.data() + volume, 0);
      assign_values_to_array<int32_t, 2>(data_input, numbers.data(), numbers.size());
    }
  }

  // start custom test-task with aligned in/out
  auto task = runtime->create_task(
    library, legate::LocalTaskID{CHECK_REPARTITION_TASK + I_ROW_MAJOR * 2 + O_ROW_MAJOR});
  auto part_in  = task.add_input(data_input.get_store());
  auto part_out = task.add_output(data_output.get_store());
  task.add_scalar_arg(legate::Scalar{tile_shape[0]});
  task.add_scalar_arg(legate::Scalar{tile_shape[1]});
  task.add_constraint(legate::align(part_in, part_out));
  task.add_communicator("nccl");
  runtime->submit(std::move(task));

  check_array_eq<int32_t, 2>(data_input, data_output);
}

void run_tests_with_shape(std::vector<uint64_t>& data_shape, std::vector<uint64_t>& tile_shape)
{
  auto machine  = legate::Runtime::get_runtime()->get_machine();
  auto num_gpus = machine.count(legate::mapping::TaskTarget::GPU);
  if (num_gpus < 2) {
    GTEST_SKIP();
  }

  run_test_aligned_default_launch<true, true>(data_shape, tile_shape);
  run_test_aligned_default_launch<false, false>(data_shape, tile_shape);
  run_test_aligned_default_launch<false, true>(data_shape, tile_shape);
  run_test_aligned_default_launch<true, false>(data_shape, tile_shape);
}

std::vector<std::vector<uint64_t>> NICE_SHAPES   = {{64, 64}, {64, 32}, {256, 256}, {512, 1}};
std::vector<std::vector<uint64_t>> NICE_TILESIZE = {{4, 4}, {32, 32}, {64, 64}, {256, 256}};

TEST(Repartition, NiceValues_C_C)
{
  register_tasks();

  for (size_t shape_idx = 0; shape_idx < NICE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < NICE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<true, true>(NICE_SHAPES[shape_idx], NICE_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, NiceValues_F_F)
{
  register_tasks();

  for (size_t shape_idx = 0; shape_idx < NICE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < NICE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<false, false>(NICE_SHAPES[shape_idx],
                                                    NICE_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, NiceValues_C_F)
{
  register_tasks();

  for (size_t shape_idx = 0; shape_idx < NICE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < NICE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<true, false>(NICE_SHAPES[shape_idx], NICE_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, NiceValues_F_C)
{
  register_tasks();

  for (size_t shape_idx = 0; shape_idx < NICE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < NICE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<false, true>(NICE_SHAPES[shape_idx], NICE_TILESIZE[tile_idx]);
    }
  }
}

std::vector<std::vector<uint64_t>> ODD_SHAPES = {
  {120, 257}, {148, 12}, {12, 2325}, {1112, 31}, {256, 256}, {12, 1}};

std::vector<std::vector<uint64_t>> ODD_TILESIZE = {
  {2, 2}, {64, 32}, {255, 256}, {16, 5}, {1, 1}, {4, 4}};

TEST(Repartition, OddValues_C_C)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < ODD_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < ODD_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<true, true>(ODD_SHAPES[shape_idx], ODD_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, OddValues_F_F)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < ODD_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < ODD_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<false, false>(ODD_SHAPES[shape_idx], ODD_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, OddValues_C_F)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < ODD_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < ODD_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<true, false>(ODD_SHAPES[shape_idx], ODD_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, OddValues_F_C)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < ODD_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < ODD_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<false, true>(ODD_SHAPES[shape_idx], ODD_TILESIZE[tile_idx]);
    }
  }
}

std::vector<std::vector<uint64_t>> STRANGE_SHAPES = {
  {120, 257}, {148, 12}, {12, 2325}, {1112, 31}, {256, 256}, {12, 1}};

std::vector<std::vector<uint64_t>> STRANGE_TILESIZE = {
  {2, 2}, {64, 32}, {255, 256}, {16, 5}, {1, 1}, {4, 4}};

TEST(Repartition, StrangeValues_C_C)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < STRANGE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < STRANGE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<true, true>(STRANGE_SHAPES[shape_idx],
                                                  STRANGE_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, StrangeValues_F_F)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < STRANGE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < STRANGE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<false, false>(STRANGE_SHAPES[shape_idx],
                                                    STRANGE_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, StrangeValues_C_F)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < STRANGE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < STRANGE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<true, false>(STRANGE_SHAPES[shape_idx],
                                                   STRANGE_TILESIZE[tile_idx]);
    }
  }
}

TEST(Repartition, StrangeValues_F_C)
{
  register_tasks();
  for (size_t shape_idx = 0; shape_idx < STRANGE_SHAPES.size(); ++shape_idx) {
    for (size_t tile_idx = 0; tile_idx < STRANGE_TILESIZE.size(); ++tile_idx) {
      run_test_aligned_default_launch<false, true>(STRANGE_SHAPES[shape_idx],
                                                   STRANGE_TILESIZE[tile_idx]);
    }
  }
}

}  // namespace repartition_test
