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

#include "cupynumeric/index/zip.h"
#include "cupynumeric/index/zip_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <int DIM, int N, size_t... Is>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel(const AccessorWO<Point<N>, DIM> out,
             const Buffer<AccessorRO<int64_t, DIM>, 1> index_arrays,
             const Rect<DIM> rect,
             const Pitches<DIM - 1> pitches,
             const size_t volume,
             const DomainPoint shape,
             std::index_sequence<Is...>)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto p = pitches.unflatten(idx, rect.lo);
  Point<N> new_point;
  for (size_t i = 0; i < N; i++) {
    new_point[i] = compute_idx_cuda(index_arrays[i][p], shape[i]);
  }
  out[p] = new_point;
}

template <int DIM, int N, size_t... Is>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel_dense(Point<N>* out,
                   const Buffer<const int64_t*, 1> index_arrays,
                   const Rect<DIM> rect,
                   const size_t volume,
                   const DomainPoint shape,
                   std::index_sequence<Is...>)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  Point<N> new_point;
  for (size_t i = 0; i < N; i++) {
    new_point[i] = compute_idx_cuda(index_arrays[i][idx], shape[i]);
  }
  out[idx] = new_point;
}

template <int DIM, int N>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zip_kernel(const AccessorWO<Point<N>, DIM> out,
             const Buffer<AccessorRO<int64_t, DIM>, 1> index_arrays,
             const Rect<DIM> rect,
             const Pitches<DIM - 1> pitches,
             const int64_t narrays,
             const size_t volume,
             const int64_t key_dim,
             const int64_t start_index,
             const DomainPoint shape)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  auto p = pitches.unflatten(idx, rect.lo);
  Point<N> new_point;
  for (size_t i = 0; i < start_index; i++) {
    new_point[i] = p[i];
  }
  for (size_t i = 0; i < narrays; i++) {
    new_point[start_index + i] = compute_idx_cuda(index_arrays[i][p], shape[start_index + i]);
  }
  for (size_t i = (start_index + narrays); i < N; i++) {
    int64_t j    = key_dim + i - narrays;
    new_point[i] = p[j];
  }
  out[p] = new_point;
}

template <typename Output, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  check_kernel(Output out,
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
  for (size_t i = 0; i < iters; i++) {
    const auto idx = (i * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= volume) {
      break;
    }
    auto p = pitches.unflatten(idx, rect.lo);
    for (size_t n = 0; n < narrays; n++) {
      const int64_t extent = shape[start_index + n];
      coord_t index = index_arrays[n][p] < 0 ? index_arrays[n][p] + extent : index_arrays[n][p];
      bool val      = (index < 0 || index >= extent);
      SumReduction<bool>::fold<true>(value, val);
    }  // for n
  }
  reduce_output(out, value);
}

template <int DIM, int N>
struct ZipImplBody<VariantKind::GPU, DIM, N> {
  using VAL = int64_t;

  void check_out_of_bounds(const Buffer<AccessorRO<int64_t, DIM>, 1>& index_arrays,
                           const int64_t volume,
                           const Rect<DIM>& rect,
                           const Pitches<DIM - 1>& pitches,
                           const int64_t narrays,
                           const int64_t start_index,
                           const DomainPoint& shape,
                           cudaStream_t stream) const
  {
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shmem_size   = THREADS_PER_BLOCK / 32 * sizeof(bool);
    DeviceScalarReductionBuffer<SumReduction<bool>> out_of_bounds(stream);
    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      check_kernel<<<MAX_REDUCTION_CTAS, THREADS_PER_BLOCK, shmem_size, stream>>>(
        out_of_bounds, index_arrays, volume, iters, rect, pitches, narrays, start_index, shape);
    } else {
      check_kernel<<<blocks, THREADS_PER_BLOCK, shmem_size, stream>>>(
        out_of_bounds, index_arrays, volume, 1, rect, pitches, narrays, start_index, shape);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

    bool res = out_of_bounds.read(stream);
    if (res) {
      throw legate::TaskException("index is out of bounds in index array");
    }
  }

  template <size_t... Is>
  void operator()(const AccessorWO<Point<N>, DIM>& out,
                  const std::vector<AccessorRO<VAL, DIM>>& index_arrays,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense,
                  const int64_t key_dim,
                  const int64_t start_index,
                  const DomainPoint& shape,
                  std::index_sequence<Is...>) const
  {
    auto stream         = get_cached_stream();
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto index_buf =
      create_buffer<AccessorRO<VAL, DIM>, 1>(index_arrays.size(), legate::Memory::Kind::Z_COPY_MEM);
    for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) {
      index_buf[idx] = index_arrays[idx];
    }
    check_out_of_bounds(
      index_buf, volume, rect, pitches, index_arrays.size(), start_index, shape, stream);

    if (index_arrays.size() == N) {
      if (dense) {
        auto index_buf_dense =
          create_buffer<const int64_t*, 1>(index_arrays.size(), legate::Memory::Kind::Z_COPY_MEM);
        for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) {
          index_buf_dense[idx] = index_arrays[idx].ptr(rect);
        }
        zip_kernel_dense<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          out.ptr(rect), index_buf_dense, rect, volume, shape, std::make_index_sequence<N>());
      } else {
        zip_kernel<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          out, index_buf, rect, pitches, volume, shape, std::make_index_sequence<N>());
      }
    } else {
#ifdef DEBUG_CUPYNUMERIC
      assert(index_arrays.size() < N);
#endif
      int num_arrays = index_arrays.size();
      zip_kernel<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        out, index_buf, rect, pitches, num_arrays, volume, key_dim, start_index, shape);
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void ZipTask::gpu_variant(TaskContext context)
{
  zip_template<VariantKind::GPU>(context);
}
}  // namespace cupynumeric
