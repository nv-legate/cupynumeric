/* Copyright 2026 NVIDIA Corporation
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

#include <cstdint>
#include <numeric>
#include <vector>

#include "legate.h"
#include "cupynumeric.h"
#include "util.inl"
#include "cupynumeric/all2all/all2all.cuh"

namespace all2all_test {

constexpr const char* library_name = "test_all2all";

enum TaskIDs {
  ALL2ALL_GATHER_U32 = 0,
  ALL2ALL_GATHER_U64 = 1,
  // Offset-width selection tests register at ALL2ALL_OFFSET_SELECT + DIM.
  ALL2ALL_OFFSET_SELECT = 10,
};

// One task per offset width. The LocalTaskID is derived from OffsetT so the
// two instantiations register under distinct IDs.
template <typename OffsetT>
struct All2AllGatherWidthTestTask : public legate::LegateTask<All2AllGatherWidthTestTask<OffsetT>> {
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{
    std::is_same_v<OffsetT, uint32_t> ? ALL2ALL_GATHER_U32 : ALL2ALL_GATHER_U64}};

  // Similar to All2AllTask
  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context);
};

// Exercises the runtime offset-width decision (partition_offsets_fit_uint32)
// with synthetic, metadata-only partition rects.
template <int DIM>
struct All2AllOffsetSelectTestTask : public legate::LegateTask<All2AllOffsetSelectTestTask<DIM>> {
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{ALL2ALL_OFFSET_SELECT + DIM}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

// Minimal mapper: default store mappings, unbounded allocation pool. Similar to
// RepartitionLayoutMapper.
class All2AllTestMapper : public legate::mapping::Mapper {
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    std::vector<legate::mapping::StoreMapping> mappings;
    for (auto& input : task.inputs()) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(input, options.front(), true /*exact*/));
    }
    for (auto& output : task.outputs()) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(output, options.front(), true /*exact*/));
    }
    return mappings;
  }
  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }
  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task& /*task*/,
                                                  legate::mapping::StoreTarget memory_kind) override
  {
    switch (memory_kind) {
      case legate::mapping::StoreTarget::FBMEM: return std::size_t{256} << 20;  // 256 MiB
      case legate::mapping::StoreTarget::ZCMEM:
        return std::size_t{16} << 20;  // 16 MiB
      // The CPU-variant offset-selection task allocates its synthetic-rect
      // buffer from SYSMEM.
      case legate::mapping::StoreTarget::SYSMEM: return std::size_t{1} << 20;  // 1 MiB
      case legate::mapping::StoreTarget::SOCKETMEM: return 0;
    }
    return 0;
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(
    library_name, legate::ResourceConfig{}, std::make_unique<All2AllTestMapper>());

  All2AllGatherWidthTestTask<uint32_t>::register_variants(library);
  All2AllGatherWidthTestTask<uint64_t>::register_variants(library);
  All2AllOffsetSelectTestTask<1>::register_variants(library);
  All2AllOffsetSelectTestTask<2>::register_variants(library);
}

template <typename OffsetT>
/*static*/ void All2AllGatherWidthTestTask<OffsetT>::gpu_variant(legate::TaskContext context)
{
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  using namespace cupynumeric::detail;

  constexpr auto CODE = legate::Type::Code::FLOAT64;
  using VAL           = legate::type_of_t<CODE>;
  constexpr int DIM   = 1;
  using INDEX_VAL     = legate::Point<DIM>;

  const auto stream      = context.get_task_stream();
  const auto total_ranks = context.get_launch_domain().get_volume();
  if (total_ranks == 1) {
    std::cerr << "Error: aborting due to single task launch. Ensure LEGATE_TEST=1 to force "
                 "parallel execution for small test dimensions."
              << std::endl;
    return;
  }
  const int num_ranks = static_cast<int>(total_ranks);

  // Step 0: task inputs/outputs. input(0) = source (partitioned across ranks),
  // input(1) = materialized Point<1> index store, output(0) = result.
  legate::PhysicalStore input_array{context.input(0)};
  legate::PhysicalStore output_array{context.output(0)};
  const auto input_rect  = input_array.shape<DIM>();
  const auto output_rect = output_array.shape<DIM>();

  const auto input_acc  = input_array.read_accessor<VAL, DIM>(input_rect);
  const auto output_acc = output_array.read_write_accessor<VAL, DIM>(output_rect);

  auto* nccl_comm = context.communicators()[0].get<ncclComm_t*>();

  // Scalars mirror the production launcher: staging factor + global index volume.
  const auto staging_factor = context.scalar(0).value<double>();
  const auto global_index   = context.scalar(1).value<uint64_t>();
  const size_t max_staging_bytes =
    compute_max_staging_bytes(staging_factor, global_index, sizeof(VAL), num_ranks);

  legate::PhysicalStore index_array{context.input(1)};
  const auto index_rect = index_array.shape<DIM>();
  const auto index_acc  = index_array.read_accessor<INDEX_VAL, DIM>(index_rect);
  StoreView<AccessMode::Read, INDEX_VAL, DIM> index_view(index_acc, index_rect);
  PointIndexLoaderProvider<DIM, DIM> provider{&index_view};
  const size_t local_index_count = index_rect.volume();

  StoreView<AccessMode::Read, VAL, DIM> input_view(input_acc, input_rect);
  StoreView<AccessMode::Write, VAL, DIM> output_view(output_acc, output_rect);

  // Step 1: AllGather partition rects and pre-compute linearization strides.
  auto partition_rects = allgather_partition_rects<DIM>(context, input_rect, nccl_comm, stream);
  auto partition_rect_infos = build_linearized_rect_infos<DIM>(partition_rects, num_ranks, stream);

  const size_t request_alloc_count = std::max<size_t>(local_index_count, 1);
  auto request_positions =
    legate::create_buffer<uint64_t>(request_alloc_count, legate::Memory::Kind::GPU_FB_MEM);
  auto target_ranks =
    legate::create_buffer<int>(request_alloc_count, legate::Memory::Kind::GPU_FB_MEM);
  auto send_offsets_per_rank =
    legate::create_buffer<unsigned long long>(num_ranks, legate::Memory::Kind::Z_COPY_MEM);

  if (local_index_count > 0) {
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(target_ranks.ptr(0), -1, local_index_count * sizeof(int), stream));
  }

  // Step 2: classify + exchange counts.
  auto plan = create_shuffle_information<DIM>(context,
                                              provider,
                                              local_index_count,
                                              partition_rects.ptr(0),
                                              target_ranks.ptr(0),
                                              send_offsets_per_rank.ptr(0),
                                              nccl_comm,
                                              stream);

  const unsigned long long global_max =
    allreduce_global_max_pair_count(context,
                                    plan.h_send_counts_per_rank,
                                    plan.h_receive_counts_per_rank,
                                    num_ranks,
                                    nccl_comm,
                                    stream);

  if (global_max == 0) {
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
    return;
  }

  plan.max_elems_per_peer =
    compute_max_elems_per_peer(max_staging_bytes, sizeof(VAL), num_ranks, global_max);
  plan.num_rounds =
    (static_cast<size_t>(global_max) + plan.max_elems_per_peer - 1) / plan.max_elems_per_peer;

  pack_request_positions(target_ranks.ptr(0),
                         local_index_count,
                         send_offsets_per_rank.ptr(0),
                         num_ranks,
                         request_positions.ptr(0),
                         stream);

  // Step 3+4+5: the offset width is forced by OffsetT (the whole point of this
  // test) rather than chosen via partition_offsets_fit_uint32.
  local_gather_and_exchange<CODE, DIM, DIM, DIM, /*IS_GATHER=*/true, OffsetT>(
    context,
    provider,
    request_positions.ptr(0),
    plan,
    partition_rect_infos.ptr(0),
    input_view,
    output_view,
    sizeof(VAL),
    nccl_comm,
    stream);

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
#endif
}

// Drives one width of the gather task: source = arange(source_len) (float64),
// a Point<1> index store filled with `indices`, and a zeroed result. After the
// launch, result[p] must equal source[indices[p]] == indices[p].
void run_gather_width_test(uint64_t source_len,
                           const std::vector<int64_t>& indices,
                           legate::LocalTaskID task_id)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);
  auto machine = runtime->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) < 2) {
    GTEST_SKIP();
  }

  const size_t num_indices = indices.size();

  // source = arange(source_len) as float64, so source[k] == k.
  std::vector<uint64_t> src_shape{source_len};
  auto source = cupynumeric::zeros(src_shape, legate::float64());
  {
    std::vector<double> vals(source_len);
    std::iota(vals.begin(), vals.end(), 0.0);
    assign_values_to_array<double, 1>(source, vals.data(), vals.size());
  }

  // index store: a materialized Point<1> per element
  auto index_store = runtime->create_store(legate::Shape{num_indices}, legate::point_type(1));
  {
    auto phys = index_store.get_physical_store();
    auto rect = phys.shape<1>();
    auto acc  = phys.write_accessor<legate::Point<1>, 1>(rect);
    size_t i  = 0;
    for (legate::PointInRectIterator<1> it(rect, false); it.valid(); ++it) {
      acc[*it] = legate::Point<1>(indices[i++]);
    }
  }

  std::vector<uint64_t> res_shape{num_indices};
  auto result = cupynumeric::zeros(res_shape, legate::float64());

  auto task = runtime->create_task(library, task_id);
  task.add_input(source.get_store());
  auto part_idx = task.add_input(index_store);
  auto part_res = task.add_output(result.get_store());
  task.add_scalar_arg(legate::Scalar{double{2.0}});
  task.add_scalar_arg(legate::Scalar{static_cast<uint64_t>(num_indices)});
  task.add_constraint(legate::align(part_res, part_idx));
  task.add_communicator("nccl");
  runtime->submit(std::move(task));

  std::vector<double> expected(num_indices);
  for (size_t p = 0; p < num_indices; ++p) {
    expected[p] = static_cast<double>(indices[p]);
  }
  check_array_eq<double, 1>(result, expected.data(), expected.size());
}

std::vector<int64_t> make_scattered_indices(uint64_t source_len, size_t count)
{
  // Deterministic spread across [0, source_len) so requests hit every rank.
  std::vector<int64_t> indices(count);
  for (size_t i = 0; i < count; ++i) {
    indices[i] = static_cast<int64_t>((i * 2654435761ULL) % source_len);
  }
  return indices;
}

TEST(All2All, GatherOffsetWidthUint32)
{
  register_tasks();
  auto machine = legate::Runtime::get_runtime()->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) < 2) {
    GTEST_SKIP();
  }

  const uint64_t source_len = 4096;
  auto indices              = make_scattered_indices(source_len, 1024);
  run_gather_width_test(source_len, indices, legate::LocalTaskID{ALL2ALL_GATHER_U32});
}

TEST(All2All, GatherOffsetWidthUint64)
{
  register_tasks();
  auto machine = legate::Runtime::get_runtime()->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) < 2) {
    GTEST_SKIP();
  }

  const uint64_t source_len = 4096;
  auto indices              = make_scattered_indices(source_len, 1024);
  run_gather_width_test(source_len, indices, legate::LocalTaskID{ALL2ALL_GATHER_U64});
}

template <int DIM>
/*static*/ void All2AllOffsetSelectTestTask<DIM>::cpu_variant(legate::TaskContext context)
{
  using namespace cupynumeric::detail;

  bool all_ok = true;
  auto check  = [&all_ok](const std::vector<legate::Rect<DIM>>& rects, bool expected) {
    const size_t n = rects.size();
    auto buf = legate::create_buffer<int8_t>(std::max<size_t>(n, 1) * sizeof(legate::Rect<DIM>),
                                             legate::Memory::Kind::SYSTEM_MEM);
    if (n > 0) {
      std::memcpy(buf.ptr(0), rects.data(), n * sizeof(legate::Rect<DIM>));
    }
    const bool actual = partition_offsets_fit_uint32<DIM>(buf, n);
    EXPECT_EQ(actual, expected);
    all_ok = all_ok && (actual == expected);
  };

  if constexpr (DIM == 1) {
    check({legate::Rect<1>(0, 1023)}, true);
    check({legate::Rect<1>(0, 5'000'000'000LL)}, false);
    check({legate::Rect<1>(0, 4294967295LL)}, true);
    check({legate::Rect<1>(0, 4294967296LL)}, false);
    check({legate::Rect<1>(1, 0)}, true);
    check({legate::Rect<1>(0, 1023), legate::Rect<1>(0, 5'000'000'000LL)}, false);
  } else {
    const legate::Point<DIM> lo(0, 0);
    check({legate::Rect<DIM>(lo, legate::Point<DIM>(65535, 65535))}, true);
    check({legate::Rect<DIM>(lo, legate::Point<DIM>(65535, 65536))}, false);
  }

  const int64_t sentinel_value = all_ok ? static_cast<int64_t>(DIM) : int64_t{-1};
  legate::PhysicalStore out{context.output(0)};
  auto rect = out.shape<1>();
  auto acc  = out.write_accessor<int64_t, 1>(rect);
  for (legate::PointInRectIterator<1> it(rect, false); it.valid(); ++it) {
    acc[*it] = sentinel_value;
  }
}

template <int DIM>
void run_offset_select_test()
{
  auto runtime  = legate::Runtime::get_runtime();
  auto library  = runtime->find_library(library_name);
  auto sentinel = cupynumeric::zeros({1}, legate::int64());

  auto task = runtime->create_task(library, legate::LocalTaskID{ALL2ALL_OFFSET_SELECT + DIM});
  task.add_output(sentinel.get_store());
  runtime->submit(std::move(task));

  const int64_t expected_sentinel = DIM;
  check_array_eq<int64_t, 1>(sentinel, &expected_sentinel, 1);
}

TEST(All2AllOffsetSelect, OneD)
{
  register_tasks();
  run_offset_select_test<1>();
}

TEST(All2AllOffsetSelect, TwoD)
{
  register_tasks();
  run_offset_select_test<2>();
}

}  // namespace all2all_test
