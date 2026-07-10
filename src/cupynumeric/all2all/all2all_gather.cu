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

#include "cupynumeric/all2all/all2all_gather.h"
#include "cupynumeric/all2all/all2all.cuh"
#include "cupynumeric/index/gather_template.inl"
#include "cupynumeric/index/zip_template.inl"
#include "cupynumeric/index/zipgather.h"
#include "cupynumeric/all2all/all2all.inl"

namespace cupynumeric {

// Distributed gather via all-to-all shuffle (NCCL)
//
// Implements `output[p] = source[index[p]]` where `source` is the array
// partitioned across N ranks. The 5-step pipeline (AllGather rects,
// classify + exchange counts, linearize + exchange offsets, pack + exchange
// values owner -> requester, unpack on requester) is factored into
// all2all.cuh so All2AllTask and All2AllScatterTask can share helpers. See
// all2all.cuh for the step-by-step algorithm and all2all_scatter.cu for
// the scatter counterpart.
//
// Two index-input modes coexist behind the `narrays` scalar:
//   narrays == 0  - Legacy path: input(1) is a materialized Point<DIM_input>
//                   store built by the CUPYNUMERIC_ZIP task.
//   narrays  > 0  - Fused path: input(1..narrays) are per-dim int64 arrays
//                   (no ZIP launch); the Point<DIM_input> is reconstructed on
//                   the fly inside the kernels via
//                   `FusedIndexLoaderProvider`.

using namespace legate;

namespace {

using namespace cupynumeric::detail;

// ============================================================================
// Orchestrator - Runs steps 1-5 for gather
// ============================================================================

template <Type::Code CODE, int DIM_input, int DIM_output, typename IndexLoaderProvider>
void global_all2all_gather_impl(TaskContext& context,
                                const IndexLoaderProvider& index_provider,
                                AccessorRO<type_of<CODE>, DIM_input> input_acc,
                                AccessorRW<type_of<CODE>, DIM_output> output_acc,
                                const legate::Rect<DIM_input>& input_rect,
                                const legate::Rect<DIM_output>& output_rect,
                                size_t local_index_count,
                                size_t max_staging_bytes,
                                ncclComm_t* nccl_comm,
                                cudaStream_t stream)
{
  using VAL           = type_of<CODE>;
  const int num_ranks = static_cast<int>(context.get_launch_domain().get_volume());

  StoreView<AccessMode::Read, VAL, DIM_input> input_view(input_acc, input_rect);
  StoreView<AccessMode::Write, VAL, DIM_output> output_view(output_acc, output_rect);

  auto partition_rects =
    allgather_partition_rects<DIM_input>(context, input_rect, nccl_comm, stream);
  auto partition_rect_infos =
    build_linearized_rect_infos<DIM_input>(partition_rects, num_ranks, stream);

  // Legion's DeferredBuffer asserts when the typed view is reconstructed
  // from the underlying UntypedDeferredBuffer for a zero-element rect
  // (field_size == sizeof(FT).
  // Bump the per-request allocation to at least one element so the cast is
  // always valid. All downstream consumers are already guarded by the real
  // `local_index_count`, so the extra slot is never read.
  const size_t request_alloc_count = std::max<size_t>(local_index_count, 1);
  auto request_positions = create_buffer<uint64_t>(request_alloc_count, Memory::Kind::GPU_FB_MEM);
  auto target_ranks      = create_buffer<int>(request_alloc_count, Memory::Kind::GPU_FB_MEM);
  auto send_offsets_per_rank =
    create_buffer<unsigned long long>(num_ranks, Memory::Kind::Z_COPY_MEM);

  if (local_index_count > 0) {
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(target_ranks.ptr(0), -1, local_index_count * sizeof(int), stream));
  }

  auto plan = create_shuffle_information<DIM_input>(context,
                                                    index_provider,
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

  // The Step-3 offsets are flat positions inside the owner's partition rect,
  // so they fit in uint32 when every partition volume does. Every rank derives
  // the same decision from the AllGathered rects (no extra coordination).
  const bool use_uint32_offsets =
    partition_offsets_fit_uint32<DIM_input>(partition_rects, num_ranks);

  if (use_uint32_offsets) {
    local_gather_and_exchange<CODE, DIM_input, DIM_input, DIM_output, /*IS_GATHER=*/true, uint32_t>(
      context,
      index_provider,
      request_positions.ptr(0),
      plan,
      partition_rect_infos.ptr(0),
      input_view,
      output_view,
      sizeof(VAL),
      nccl_comm,
      stream);
  } else {
    local_gather_and_exchange<CODE, DIM_input, DIM_input, DIM_output, /*IS_GATHER=*/true, uint64_t>(
      context,
      index_provider,
      request_positions.ptr(0),
      plan,
      partition_rect_infos.ptr(0),
      input_view,
      output_view,
      sizeof(VAL),
      nccl_comm,
      stream);
  }
}

// ============================================================================
// Legate task dispatch
// ============================================================================

template <Type::Code CODE, int32_t DIM_input, int32_t DIM_output>
struct All2AllGatherGPUBody {
  using VAL       = type_of<CODE>;
  using INDEX_VAL = legate::Point<DIM_input>;

  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input_array,
                  const legate::PhysicalStore& output_array,
                  size_t max_staging_bytes,
                  int64_t narrays,
                  int64_t key_dim,
                  int64_t start_index,
                  const legate::DomainPoint& shape,
                  bool check_bounds)
  {
    const auto stream      = context.get_task_stream();
    const auto input_rect  = input_array.shape<DIM_input>();
    const auto output_rect = output_array.shape<DIM_output>();

    const auto input  = input_array.read_accessor<VAL, DIM_input>(input_rect);
    const auto output = output_array.read_write_accessor<VAL, DIM_output>(output_rect);

    auto* nccl_comm = context.communicators()[0].get<ncclComm_t*>();

    if (narrays == 0) {
      legate::PhysicalStore index_array{context.input(1)};
      const auto index_rect = index_array.shape<DIM_output>();
      const auto index_acc  = index_array.read_accessor<INDEX_VAL, DIM_output>(index_rect);
      StoreView<AccessMode::Read, INDEX_VAL, DIM_output> index_view(index_acc, index_rect);
      PointIndexLoaderProvider<DIM_input, DIM_output> provider{&index_view};
      const size_t local_index_count = index_rect.volume();

      global_all2all_gather_impl<CODE, DIM_input, DIM_output>(context,
                                                              provider,
                                                              input,
                                                              output,
                                                              input_rect,
                                                              output_rect,
                                                              local_index_count,
                                                              max_staging_bytes,
                                                              nccl_comm,
                                                              stream);
    } else {
      // Fused path: per-dim int64 index arrays at slots 1..narrays.
      // They are aligned with the result iteration domain (output_rect).
      const auto provider = build_fused_index_provider<DIM_input, DIM_output>(context,
                                                                              narrays,
                                                                              key_dim,
                                                                              start_index,
                                                                              shape,
                                                                              check_bounds,
                                                                              output_rect,
                                                                              nccl_comm,
                                                                              stream);

      const size_t local_index_count = output_rect.volume();

      global_all2all_gather_impl<CODE, DIM_input, DIM_output>(context,
                                                              provider,
                                                              input,
                                                              output,
                                                              input_rect,
                                                              output_rect,
                                                              local_index_count,
                                                              max_staging_bytes,
                                                              nccl_comm,
                                                              stream);
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <int DIM_input, int DIM_output>
struct All2AllGatherImpl_type {
  template <Type::Code CODE>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& output,
                  size_t max_staging_bytes,
                  int64_t narrays,
                  int64_t key_dim,
                  int64_t start_index,
                  const legate::DomainPoint& shape,
                  bool check_bounds) const
  {
    All2AllGatherGPUBody<CODE, DIM_input, DIM_output>()(context,
                                                        input,
                                                        output,
                                                        max_staging_bytes,
                                                        narrays,
                                                        key_dim,
                                                        start_index,
                                                        shape,
                                                        check_bounds);
  }
};

struct All2AllGatherImpl {
  template <int DIM_input, int DIM_output>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& output,
                  size_t max_staging_bytes,
                  int64_t narrays,
                  int64_t key_dim,
                  int64_t start_index,
                  const legate::DomainPoint& shape,
                  bool check_bounds) const
  {
    type_dispatch(input.code(),
                  All2AllGatherImpl_type<DIM_input, DIM_output>{},
                  context,
                  input,
                  output,
                  max_staging_bytes,
                  narrays,
                  key_dim,
                  start_index,
                  shape,
                  check_bounds);
  }
};

void all2all_gather_gpu(TaskContext& context)
{
  const auto input = context.input(0);
  auto output      = context.output(0);

  const auto domain         = context.get_launch_domain();
  const size_t num_ranks    = domain.get_volume();
  const bool is_index_space = !context.is_single_task() && context.communicators().size() > 0;

  assert(is_index_space || num_ranks == 1);

  if (is_index_space) {
    const auto dim_input      = std::max(input.dim(), 1);
    const auto dim_output     = std::max(output.dim(), 1);
    const auto staging_factor = context.scalar(0).value<double>();
    const auto global_index   = context.scalar(1).value<uint64_t>();
    const auto narrays        = context.scalar(2).value<int64_t>();
    const auto fused          = detail::read_index_scalar_args(context, narrays);
    const auto elem_size      = input.type().size();
    const auto max_staging_bytes =
      detail::compute_max_staging_bytes(staging_factor, global_index, elem_size, num_ranks);

    legate::double_dispatch(dim_input,
                            dim_output,
                            All2AllGatherImpl{},
                            context,
                            input,
                            output,
                            max_staging_bytes,
                            narrays,
                            fused.key_dim,
                            fused.start_index,
                            fused.shape,
                            fused.check_bounds);
  } else {
    const auto stream  = context.get_task_stream();
    const auto narrays = context.scalar(2).value<int64_t>();
    if (narrays == 0) {
      // Legacy path: input(1) is a materialized Point<DIM_input> store.
      const auto index_array = context.input(1);
      type_dispatch(
        input.type().code(),
        GatherTypeDispatch<decltype(DEFAULT_POLICY.on(stream))>{DEFAULT_POLICY.on(stream)},
        output,
        input,
        index_array);
    } else {
      // Fused path: input(1..narrays) are per-dim int64 index arrays. All data
      // is local on this rank, so skip NCCL and run the same local zip+gather
      // as CUPYNUMERIC_ZIPGATHER (reconstructs the Point<DIM_input> on the fly).
      const auto fused = detail::read_index_scalar_args(context, narrays);
      std::vector<legate::PhysicalStore> index_arrays;
      index_arrays.reserve(narrays);
      for (int64_t i = 0; i < narrays; ++i) {
        index_arrays.emplace_back(context.input(static_cast<uint32_t>(1 + i)));
      }
      ZipGatherArgs args{output,
                         input,
                         std::move(index_arrays),
                         fused.key_dim,
                         fused.start_index,
                         fused.shape,
                         fused.check_bounds};
      detail::launch_local_zipgather_gpu(context, args);
    }
  }
}

const auto cupynumeric_reg_task_ = []() -> char {
  All2AllTask::register_variants();
  return 0;
}();

}  // namespace

/*static*/ void All2AllTask::gpu_variant(TaskContext context) { all2all_gather_gpu(context); }

}  // namespace cupynumeric
