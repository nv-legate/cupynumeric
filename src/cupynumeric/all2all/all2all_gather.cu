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

using namespace legate;

namespace {

using namespace cupynumeric::detail;

// ============================================================================
// Orchestrator - Runs steps 1-5 for gather
// ============================================================================

template <Type::Code CODE, int DIM_input, int DIM_output>
void global_all2all_gather(TaskContext& context,
                           AccessorRO<type_of<CODE>, DIM_input> input_acc,
                           AccessorRO<legate::Point<DIM_input>, DIM_output> index_acc,
                           AccessorRW<type_of<CODE>, DIM_output> output_acc,
                           const legate::Rect<DIM_input>& input_rect,
                           const legate::Rect<DIM_output>& index_rect,
                           const legate::Rect<DIM_output>& output_rect,
                           size_t local_index_count,
                           ncclComm_t* nccl_comm,
                           cudaStream_t stream)
{
  using VAL           = type_of<CODE>;
  using INDEX_VAL     = legate::Point<DIM_input>;
  const int num_ranks = static_cast<int>(context.get_launch_domain().get_volume());

  StoreView<AccessMode::Read, INDEX_VAL, DIM_output> index_view(index_acc, index_rect);
  StoreView<AccessMode::Read, VAL, DIM_input> input_view(input_acc, input_rect);
  StoreView<AccessMode::Write, VAL, DIM_output> output_view(output_acc, output_rect);

  // Step 1: AllGather the source partition rects.
  auto partition_rects =
    allgather_partition_rects<DIM_input>(context, input_rect, nccl_comm, stream);
  auto partition_rect_infos =
    build_linearized_rect_infos<DIM_input>(partition_rects, num_ranks, stream);

  auto request_positions = create_buffer<uint64_t>(local_index_count, Memory::Kind::GPU_FB_MEM);
  auto target_ranks      = create_buffer<int>(local_index_count, Memory::Kind::GPU_FB_MEM);
  auto send_offsets_per_rank =
    create_buffer<unsigned long long>(num_ranks, Memory::Kind::Z_COPY_MEM);
  auto recv_staging_buffer =
    create_buffer<int8_t>(local_index_count * sizeof(VAL), Memory::Kind::GPU_FB_MEM);

  if (local_index_count > 0) {
    // cudaMemsetAsync uses the low byte of `value` as the fill pattern; -1 (=0xFF)
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(target_ranks.ptr(0), -1, local_index_count * sizeof(int), stream));
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemsetAsync(recv_staging_buffer.ptr(0), 0, local_index_count * sizeof(VAL), stream));
  }

  // Step 2: Classify requests and exchange counts.
  auto plan = create_shuffle_information<DIM_input, DIM_output>(context,
                                                                index_view,
                                                                local_index_count,
                                                                partition_rects.ptr(0),
                                                                request_positions.ptr(0),
                                                                target_ranks.ptr(0),
                                                                send_offsets_per_rank.ptr(0),
                                                                nccl_comm,
                                                                stream);

  // Step 3: Linearize index points to flat offsets and exchange them
  // (requester to owner).
  auto recv_flat_offsets =
    linearize_and_exchange_offsets<DIM_input, DIM_output>(context,
                                                          index_view,
                                                          local_index_count,
                                                          request_positions.ptr(0),
                                                          send_offsets_per_rank.ptr(0),
                                                          partition_rect_infos.ptr(0),
                                                          plan,
                                                          nccl_comm,
                                                          stream);

  // Step 4: Owner gathers values from local source at recv_flat_offsets, then
  // ships them back to the requester (owner -> requester).
  auto send_staging_buffer =
    create_buffer<int8_t>(plan.total_incoming * sizeof(VAL), Memory::Kind::GPU_FB_MEM);

  pack_values_into_buffer<CODE, DIM_input>(input_view,
                                           recv_flat_offsets.ptr(0),
                                           plan.total_incoming,
                                           sizeof(VAL),
                                           send_staging_buffer.ptr(0),
                                           stream);

  exchange_values_owner_to_requester(context,
                                     send_staging_buffer.ptr(0),
                                     recv_staging_buffer.ptr(0),
                                     sizeof(VAL),
                                     plan,
                                     num_ranks,
                                     nccl_comm,
                                     stream);

  // Step 5: Requester writes received values into the local output store at
  // the request_positions captured during step 2.
  unpack_recv_into_output<CODE, DIM_output>(
    output_view, request_positions.ptr(0), local_index_count, recv_staging_buffer.ptr(0), stream);
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
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output_array)
  {
    const auto stream      = context.get_task_stream();
    const auto input_rect  = input_array.shape<DIM_input>();
    const auto index_rect  = index_array.shape<DIM_output>();
    const auto output_rect = output_array.shape<DIM_output>();

    const auto input               = input_array.read_accessor<VAL, DIM_input>(input_rect);
    const auto index               = index_array.read_accessor<INDEX_VAL, DIM_output>(index_rect);
    const auto output              = output_array.read_write_accessor<VAL, DIM_output>(output_rect);
    const size_t local_index_count = index_rect.volume();

    global_all2all_gather<CODE, DIM_input, DIM_output>(
      context,
      input,
      index,
      output,
      input_rect,
      index_rect,
      output_rect,
      local_index_count,
      context.communicators()[0].get<ncclComm_t*>(),
      stream);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <int DIM_input, int DIM_output>
struct All2AllGatherImpl_type {
  template <Type::Code CODE>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output) const
  {
    All2AllGatherGPUBody<CODE, DIM_input, DIM_output>()(context, input, index_array, output);
  }
};

struct All2AllGatherImpl {
  template <int DIM_input, int DIM_output>
  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output) const
  {
    type_dispatch(input.code(),
                  All2AllGatherImpl_type<DIM_input, DIM_output>{},
                  context,
                  input,
                  index_array,
                  output);
  }
};

void all2all_gather_gpu(TaskContext& context)
{
  const auto input       = context.input(0);
  const auto index_array = context.input(1);
  auto output            = context.output(0);

  const auto domain         = context.get_launch_domain();
  const size_t num_ranks    = domain.get_volume();
  const bool is_index_space = !context.is_single_task() && context.communicators().size() > 0;

  assert(is_index_space || num_ranks == 1);

  if (is_index_space) {
    const auto dim_input  = std::max(input.dim(), 1);
    const auto dim_output = std::max(output.dim(), 1);

    legate::double_dispatch(
      dim_input, dim_output, All2AllGatherImpl{}, context, input, index_array, output);
  } else {
    const auto stream = context.get_task_stream();

    type_dispatch(
      input.type().code(),
      GatherTypeDispatch<decltype(DEFAULT_POLICY.on(stream))>{DEFAULT_POLICY.on(stream)},
      output,
      input,
      index_array);
  }
}

const auto cupynumeric_reg_task_ = []() -> char {
  All2AllTask::register_variants();
  return 0;
}();

}  // namespace

/*static*/ void All2AllTask::gpu_variant(TaskContext context) { all2all_gather_gpu(context); }

}  // namespace cupynumeric
