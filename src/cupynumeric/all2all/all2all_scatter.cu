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

  auto request_positions = create_buffer<uint64_t>(num_requests, Memory::Kind::GPU_FB_MEM);
  auto target_ranks      = create_buffer<int>(num_requests, Memory::Kind::GPU_FB_MEM);
  auto send_offsets_per_rank =
    create_buffer<unsigned long long>(num_ranks, Memory::Kind::Z_COPY_MEM);

  if (num_requests > 0) {
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(target_ranks.ptr(0), -1, num_requests * sizeof(int), stream));
  }

  // Step 2: Classify index points by destination (output) rank, exchange counts,
  // and produce the request_positions permutation.
  auto plan = create_shuffle_information<DIM_partition, DIM_local>(context,
                                                                   index_view,
                                                                   num_requests,
                                                                   partition_rects.ptr(0),
                                                                   request_positions.ptr(0),
                                                                   target_ranks.ptr(0),
                                                                   send_offsets_per_rank.ptr(0),
                                                                   nccl_comm,
                                                                   stream);

  // Step 3: Linearize index points into flat offsets within each owner's
  // output rect and ship them requester -> owner. recv_flat_offsets contains,
  // for every value other ranks have asked us to write, the flat offset
  // within our local output partition.
  auto recv_flat_offsets =
    linearize_and_exchange_offsets<DIM_partition, DIM_local>(context,
                                                             index_view,
                                                             num_requests,
                                                             request_positions.ptr(0),
                                                             send_offsets_per_rank.ptr(0),
                                                             partition_rect_infos.ptr(0),
                                                             plan,
                                                             nccl_comm,
                                                             stream);

  // Step 4: Requester gathers values from local source at request_positions,
  // then ships them to the owner (requester -> owner).
  auto send_staging_buffer =
    create_buffer<int8_t>(num_requests * sizeof(VAL), Memory::Kind::GPU_FB_MEM);
  auto recv_data_buf =
    create_buffer<int8_t>(plan.total_incoming * sizeof(VAL), Memory::Kind::GPU_FB_MEM);

  pack_values_into_buffer<CODE, DIM_local>(source_view,
                                           request_positions.ptr(0),
                                           num_requests,
                                           sizeof(VAL),
                                           send_staging_buffer.ptr(0),
                                           stream);

  exchange_values_requester_to_owner(context,
                                     send_staging_buffer.ptr(0),
                                     recv_data_buf.ptr(0),
                                     sizeof(VAL),
                                     plan,
                                     num_ranks,
                                     nccl_comm,
                                     stream);

  // Step 5: Owner writes received values into the local output partition at
  // recv_flat_offsets. Duplicate target indices race; this matches the
  // single-GPU ScatterTask semantics.
  unpack_recv_into_output<CODE, DIM_partition>(
    output_view, recv_flat_offsets.ptr(0), plan.total_incoming, recv_data_buf.ptr(0), stream);
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
                  const legate::PhysicalStore& output_array)
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
                  const legate::PhysicalStore& output) const
  {
    All2AllScatterGPUBody<CODE, DIM_partition, DIM_local>()(context, source, index_array, output);
  }
};

struct All2AllScatterImpl {
  template <int DIM_partition, int DIM_local>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& source,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output) const
  {
    type_dispatch(source.code(),
                  All2AllScatterImpl_type<DIM_partition, DIM_local>{},
                  context,
                  source,
                  index_array,
                  output);
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
    const auto dim_partition = std::max(output.dim(), 1);
    const auto dim_local     = std::max(source.dim(), 1);

    legate::double_dispatch(
      dim_partition, dim_local, All2AllScatterImpl{}, context, source, index_array, output);
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
