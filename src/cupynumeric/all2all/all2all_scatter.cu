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

#include "cupynumeric/all2all/all2all_scatter.h"
#include "cupynumeric/all2all/all2all.cuh"
#include "cupynumeric/index/scatter_template.inl"

namespace cupynumeric {

// Distributed scatter via all-to-all shuffle (NCCL)
//
// Implements `output[index[p]] = source[p]` where `output` is the array
// partitioned across N ranks. Shares the 5-step shuffle pipeline with the
// gather counterpart (All2AllTask); only Step 4's NCCL direction (requester
// -> owner) and Step 5's indirect write site (owner side) differ. See
// all2all.cuh for the step-by-step algorithm.
//
// Duplicate target indices (two local index points p1 != p2 with
// index[p1] == index[p2]) race on the owning rank's output cell; the
// surviving value is undefined, matching the single-node ScatterTask.

using namespace legate;

namespace {

using namespace cupynumeric::detail;

// ============================================================================
// Orchestrator - Runs steps 1-5 for scatter
// ============================================================================

// Template parameters:
//   DIM_partition - dimensionality of the partitioned output array (== DIM
//                   of points stored in the index array).
//   DIM_local     - dimensionality of the locally-iterated source / index
//                   arrays (which are aligned with each other).
//
// Note that the shared helpers in all2all.cuh use the names
// `<DIM_input, DIM_output>` to mean `<dim of points in index, dim of index
// store>`. For scatter that is `<DIM_partition, DIM_local>`.
template <Type::Code CODE, int DIM_partition, int DIM_local>
void global_all2all_scatter(TaskContext& context,
                            AccessorRO<type_of<CODE>, DIM_local> source_acc,
                            AccessorRO<legate::Point<DIM_partition>, DIM_local> index_acc,
                            AccessorRW<type_of<CODE>, DIM_partition> output_acc,
                            const legate::Rect<DIM_local>& source_rect,
                            const legate::Rect<DIM_local>& index_rect,
                            const legate::Rect<DIM_partition>& output_rect,
                            size_t num_requests,
                            size_t max_staging_bytes,
                            ncclComm_t* nccl_comm,
                            cudaStream_t stream)
{
  using VAL           = type_of<CODE>;
  using INDEX_VAL     = legate::Point<DIM_partition>;
  const int num_ranks = static_cast<int>(context.get_launch_domain().get_volume());

  StoreView<AccessMode::Read, INDEX_VAL, DIM_local> index_view(index_acc, index_rect);
  StoreView<AccessMode::Read, VAL, DIM_local> source_view(source_acc, source_rect);
  StoreView<AccessMode::Write, VAL, DIM_partition> output_view(output_acc, output_rect);

  // Step 1: AllGather the partitioned-output rects so every rank knows which
  // rank owns which region of the global output array.
  auto partition_rects =
    allgather_partition_rects<DIM_partition>(context, output_rect, nccl_comm, stream);
  auto partition_rect_infos =
    build_linearized_rect_infos<DIM_partition>(partition_rects, num_ranks, stream);

  // Legion's DeferredBuffer asserts when the typed view is reconstructed
  // from the underlying UntypedDeferredBuffer for a zero-element rect
  // (field_size == sizeof(FT).
  // Bump the per-request allocation to at least one element so the cast is
  // always valid. All downstream consumers are already guarded by the real
  // `local_index_count`, so the extra slot is never read.
  const size_t request_alloc_count = std::max<size_t>(num_requests, 1);
  auto request_positions = create_buffer<uint64_t>(request_alloc_count, Memory::Kind::GPU_FB_MEM);
  auto target_ranks      = create_buffer<int>(request_alloc_count, Memory::Kind::GPU_FB_MEM);
  auto send_offsets_per_rank =
    create_buffer<unsigned long long>(num_ranks, Memory::Kind::Z_COPY_MEM);

  if (num_requests > 0) {
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(target_ranks.ptr(0), -1, num_requests * sizeof(int), stream));
  }

  // Step 2: classify index points by destination (output)
  // rank and NCCL-exchange counts. `request_positions` is filled
  // separately, in peer-major layout using `send_offsets_per_rank`.
  auto plan = create_shuffle_information<DIM_partition, DIM_local>(context,
                                                                   index_view,
                                                                   num_requests,
                                                                   partition_rects.ptr(0),
                                                                   target_ranks.ptr(0),
                                                                   send_offsets_per_rank.ptr(0),
                                                                   nccl_comm,
                                                                   stream);

  // K is derived from a single ncclAllReduce(max, uint64) over per-pair
  // counts, so all ranks issue the same number of grouped collectives.
  //
  // Data direction (set inside `local_gather_and_exchange` via
  // IS_GATHER=false): requester -> owner. The schedule is built with the
  // requester-side counts directly (h_send_counts = how many data points
  // I send to peer i).
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

  // Step 2e: pack request_positions in peer-major layout (see all2all.cuh).
  pack_request_positions(target_ranks.ptr(0),
                         num_requests,
                         send_offsets_per_rank.ptr(0),
                         num_ranks,
                         request_positions.ptr(0),
                         stream);

  // Step 3 + 4 + 5: Chunked linearize / offset-exchange / pack / data-
  // exchange / unpack. Every offset and staging buffer used here is
  // round-local and sized by `max_elems_per_peer`, so per-buffer FB memory
  // is bounded by `max_staging_bytes` (set via
  // CUPYNUMERIC_ALL2ALL_STAGING_FACTOR).
  local_gather_and_exchange<CODE,
                            DIM_partition,
                            DIM_local,
                            DIM_local,
                            DIM_partition,
                            /*IS_GATHER=*/false>(context,
                                                 index_view,
                                                 request_positions.ptr(0),
                                                 plan,
                                                 partition_rect_infos.ptr(0),
                                                 source_view,
                                                 output_view,
                                                 sizeof(VAL),
                                                 nccl_comm,
                                                 stream);
}

// ============================================================================
// Legate task dispatch
// ============================================================================

template <Type::Code CODE, int32_t DIM_partition, int32_t DIM_local>
struct All2AllScatterGPUBody {
  using VAL       = type_of<CODE>;
  using INDEX_VAL = legate::Point<DIM_partition>;

  void operator()(TaskContext& context,
                  const legate::PhysicalStore& source_array,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output_array,
                  size_t max_staging_bytes)
  {
    const auto stream      = context.get_task_stream();
    const auto source_rect = source_array.shape<DIM_local>();
    const auto index_rect  = index_array.shape<DIM_local>();
    const auto output_rect = output_array.shape<DIM_partition>();

    const auto source         = source_array.read_accessor<VAL, DIM_local>(source_rect);
    const auto index          = index_array.read_accessor<INDEX_VAL, DIM_local>(index_rect);
    const auto output         = output_array.read_write_accessor<VAL, DIM_partition>(output_rect);
    const size_t num_requests = source_rect.volume();

    global_all2all_scatter<CODE, DIM_partition, DIM_local>(
      context,
      source,
      index,
      output,
      source_rect,
      index_rect,
      output_rect,
      num_requests,
      max_staging_bytes,
      context.communicators()[0].get<ncclComm_t*>(),
      stream);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <int DIM_partition, int DIM_local>
struct All2AllScatterImpl_type {
  template <Type::Code CODE>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& source,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output,
                  size_t max_staging_bytes) const
  {
    All2AllScatterGPUBody<CODE, DIM_partition, DIM_local>()(
      context, source, index_array, output, max_staging_bytes);
  }
};

struct All2AllScatterImpl {
  template <int DIM_partition, int DIM_local>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& source,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output,
                  size_t max_staging_bytes) const
  {
    type_dispatch(source.code(),
                  All2AllScatterImpl_type<DIM_partition, DIM_local>{},
                  context,
                  source,
                  index_array,
                  output,
                  max_staging_bytes);
  }
};

void all2all_scatter_gpu(TaskContext& context)
{
  const auto source      = context.input(0);
  const auto index_array = context.input(1);
  auto output            = context.output(0);

  const auto domain         = context.get_launch_domain();
  const size_t num_ranks    = domain.get_volume();
  const bool is_index_space = !context.is_single_task() && context.communicators().size() > 0;

  assert(is_index_space || num_ranks == 1);

  if (is_index_space) {
    const auto dim_partition  = std::max(output.dim(), 1);
    const auto dim_local      = std::max(source.dim(), 1);
    const auto staging_factor = context.scalar(0).value<double>();
    const auto global_index   = context.scalar(1).value<uint64_t>();
    const auto num_ranks      = static_cast<int>(domain.get_volume());
    const auto elem_size      = source.type().size();
    const auto max_staging_bytes =
      detail::compute_max_staging_bytes(staging_factor, global_index, elem_size, num_ranks);

    legate::double_dispatch(dim_partition,
                            dim_local,
                            All2AllScatterImpl{},
                            context,
                            source,
                            index_array,
                            output,
                            max_staging_bytes);
  } else {
    // Single-rank fallback: dispatch to the same scatter kernels used by
    // CUPYNUMERIC_SCATTER. Python normally routes single-rank scatter to
    // CUPYNUMERIC_SCATTER directly; this branch is defensive.
    const auto stream = context.get_task_stream();

    type_dispatch(
      source.type().code(),
      ScatterTypeDispatch<decltype(DEFAULT_POLICY.on(stream))>{DEFAULT_POLICY.on(stream)},
      output,
      source,
      index_array);
  }
}

const auto cupynumeric_reg_task_ = []() -> char {
  All2AllScatterTask::register_variants();
  return 0;
}();

}  // namespace

/*static*/ void All2AllScatterTask::gpu_variant(TaskContext context)
{
  all2all_scatter_gpu(context);
}

}  // namespace cupynumeric
