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

// Useful for IDEs
#include "cupynumeric/index/zip.h"
#include "cupynumeric/pitches.h"

#if LEGATE_DEFINED(LEGATE_USE_CUDA) and LEGATE_DEFINED(LEGATE_NVCC)
#include "cupynumeric/cuda_help.h"
#endif

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, int DIM, int N>
struct ZipImplBody;

template <VariantKind KIND>
struct ZipImpl {
  TaskContext context;
  explicit ZipImpl(TaskContext context) : context(context) {}

  template <int DIM, int N>
  void operator()(ZipArgs& args) const
  {
    using VAL     = int64_t;
    auto out_rect = args.out.shape<DIM>();
    auto out      = args.out.write_accessor<Point<N>, DIM>(out_rect);
    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(out_rect);
    if (volume == 0) {
      return;
    }

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    bool dense = out.accessor.is_dense_row_major(out_rect);
#else
    bool dense = false;
#endif
    std::vector<AccessorRO<VAL, DIM>> index_arrays;
    for (uint32_t i = 0; i < args.inputs.size(); i++) {
      index_arrays.push_back(args.inputs[i].read_accessor<VAL, DIM>(args.inputs[i].shape<DIM>()));
      dense = dense && index_arrays[i].accessor.is_dense_row_major(out_rect);
    }

    ZipImplBody<KIND, DIM, N>{context}(out,
                                       index_arrays,
                                       out_rect,
                                       pitches,
                                       dense,
                                       args.check_bounds,
                                       args.key_dim,
                                       args.start_index,
                                       args.shape,
                                       std::make_index_sequence<N>());
  }
};

template <VariantKind KIND>
static void zip_template(TaskContext& context)
{
  // Here `N` is the number of dimensions of the input array and the number
  // of dimensions of the Point<N> field
  // key_dim - is the number of dimensions of the index arrays before
  // they were broadcasted to the shape of the input array (shape of
  // all index arrays should be the same))
  // start index - is the index from which first index array was passed
  // DIM - dimension of the output array
  //
  // for the example:
  // x.shape = (2,3,4,5)
  // ind1.shape = (6,7,8)
  // ind2.shape = (6,7,8)
  // y = x[:,ind1,ind2,:]
  // y.shape == (2,6,7,8,5)
  // out.shape == (2,6,7,8,5)
  // index_arrays = [ind1', ind2']
  // ind1' == ind1 promoted to (2,6,7,8,5)
  // ind2' == ind2 promoted to (2,6,7,8,5)
  // DIM = 5
  // N = 4
  // key_dim = 3
  // start_index = 1

  int64_t N           = context.scalar(0).value<int64_t>();
  int64_t key_dim     = context.scalar(1).value<int64_t>();
  int64_t start_index = context.scalar(2).value<int64_t>();
  auto shape          = context.scalar(3).value<DomainPoint>();
  bool check_bounds   = context.scalar(4).value<bool>();
  std::vector<legate::PhysicalStore> inputs;
  for (auto& input : context.inputs()) {
    inputs.emplace_back(input);
  }
  ZipArgs args{context.output(0), std::move(inputs), N, key_dim, start_index, shape, check_bounds};
  int dim = std::max(1, args.inputs[0].dim());
  double_dispatch(dim, N, ZipImpl<KIND>{context}, args);
}

#if LEGATE_DEFINED(LEGATE_USE_CUDA) and LEGATE_DEFINED(LEGATE_NVCC)
// Reduction kernel that flags whether any element of any per-dim index array
// falls outside its corresponding extent in `shape`. Each thread sweeps
// `iters` chunks of `volume` so we can launch with a capped grid size.
// Shared between ZipTask and ZipGatherTask GPU variants.
template <typename Output, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  index_bounds_check_kernel(Output out,
                            const Buffer<AccessorRO<int64_t, DIM>, 1> index_arrays,
                            const int64_t volume,
                            const int64_t iters,
                            const Rect<DIM> rect,
                            const Pitches<DIM - 1> pitches,
                            const int64_t narrays,
                            const int64_t start_index,
                            const DomainPoint shape)
{
  bool value = false;
  for (size_t i = 0; i < iters; ++i) {
    const auto idx = (i * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= volume) {
      break;
    }
    const auto p = pitches.unflatten(idx, rect.lo);
    for (size_t n = 0; n < narrays; ++n) {
      const int64_t extent = shape[start_index + n];
      const coord_t idx    = compute_idx_unchecked(index_arrays[n][p], extent);
      const bool oob       = (idx < 0 || idx >= extent);
      SumReduction<bool>::fold<true>(value, oob);
    }
  }
  reduce_output(out, value);
}

// Host-side wrapper used by both ZipTask and ZipGatherTask GPU variants.
// Returns `true` if any index in the supplied arrays falls outside its
// corresponding extent in `shape`.
template <int DIM>
[[nodiscard]] inline bool check_index_arrays_out_of_bounds(
  const Buffer<AccessorRO<int64_t, DIM>, 1>& index_arrays,
  const int64_t volume,
  const Rect<DIM>& rect,
  const Pitches<DIM - 1>& pitches,
  const int64_t narrays,
  const int64_t start_index,
  const DomainPoint& shape,
  cudaStream_t stream)
{
  const size_t blocks     = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const size_t shmem_size = THREADS_PER_BLOCK / 32 * sizeof(bool);

  DeviceScalarReductionBuffer<SumReduction<bool>> out_of_bounds(stream);
  if (blocks >= MAX_REDUCTION_CTAS) {
    const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
    index_bounds_check_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
      out_of_bounds, index_arrays, volume, iters, rect, pitches, narrays, start_index, shape);
  } else {
    index_bounds_check_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
      out_of_bounds, index_arrays, volume, 1, rect, pitches, narrays, start_index, shape);
  }
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  return out_of_bounds.read(stream);
}
#endif  // LEGATE_DEFINED(LEGATE_NVCC)

}  // namespace cupynumeric
