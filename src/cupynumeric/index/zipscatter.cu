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

#include "cupynumeric/index/zipscatter.h"
#include "cupynumeric/index/zipscatter_template.inl"
#include "cupynumeric/index/zip_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

// Type-erased dense kernel. Source values are loaded by flat index from a
// contiguous `src_bytes` buffer, the destination point is computed by zipping
// the index arrays at the same flat index, and the write uses a typed
// load/store specialised on `elem_size` (see `copy_elements` in
// ``cuda_help.h``). Mirrors `zipgather_kernel_dense` with the read/write
// direction inverted.
template <int DIM, int N>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zipscatter_kernel_dense(char* out_bytes,
                          const Point<N> out_byte_strides,
                          const char* src_bytes,
                          const size_t elem_size,
                          const Buffer<const int64_t*, 1> index_arrays,
                          const Pitches<DIM - 1> src_pitches,
                          const Point<DIM> src_lo,
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

  Point<DIM> p{};
  if (narrays != N) {
    p = src_pitches.unflatten(idx, src_lo);
  }

  const auto new_point =
    build_source_point<DIM, N>(p,
                               DenseIndexLoader<DIM, Buffer<const int64_t*, 1>>{index_arrays},
                               idx,
                               narrays,
                               key_dim,
                               start_index,
                               shape,
                               ComputeIdxCudaFn{});
  copy_elements(
    out_bytes + new_point.dot(out_byte_strides), src_bytes + idx * elem_size, elem_size);
}

// Typed general (non-dense) kernel. Uses Legate accessors so LEGATE_BOUNDS_CHECKS
// builds validate indices and so we work correctly for non-affine layouts.
template <int DIM, int N, typename T>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  zipscatter_kernel(const AccessorWO<T, N> out,
                    const AccessorRO<T, DIM> src,
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

  const auto p         = pitches.unflatten(idx, rect.lo);
  const auto new_point = build_source_point<DIM, N>(
    p,
    SparseIndexLoader<DIM, Buffer<AccessorRO<int64_t, DIM>, 1>>{index_arrays},
    idx,
    narrays,
    key_dim,
    start_index,
    shape,
    ComputeIdxCudaFn{});
  out[new_point] = src[p];
}

template <int DIM, int N>
struct ZipScatterGeneralLauncher {
  size_t blocks;
  cudaStream_t stream;
  legate::PhysicalStore& out_store;
  legate::PhysicalStore& src_store;
  Buffer<AccessorRO<int64_t, DIM>, 1> index_buf;
  Rect<N> out_rect;
  Rect<DIM> src_rect;
  Pitches<DIM - 1> pitches;
  size_t narrays;
  size_t volume;
  int64_t key_dim;
  int64_t start_index;
  DomainPoint shape;

  template <Type::Code CODE>
  void operator()() const
  {
    using T        = type_of<CODE>;
    const auto out = out_store.write_accessor<T, N>(out_rect);
    const auto src = src_store.read_accessor<T, DIM>(src_rect);

    zipscatter_kernel<DIM, N, T><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      out, src, index_buf, src_rect, pitches, narrays, volume, key_dim, start_index, shape);
  }
};

template <int DIM, int N>
struct ZipScatterImplBody<VariantKind::GPU, DIM, N> {
  TaskContext context;
  explicit ZipScatterImplBody(TaskContext context) : context(context) {}

  using VAL = int64_t;

  void operator()(ZipScatterArgs& args) const
  {
    auto out_rect   = args.out.shape<N>();
    auto src_rect   = args.source.shape<DIM>();
    auto stream     = context.get_task_stream();
    auto out_acc    = args.out.write_accessor<char, N, false>();
    auto source_acc = args.source.read_accessor<char, DIM, false>();

    size_t out_bstrides[N];
    char* out_bytes = reinterpret_cast<char*>(out_acc.ptr(out_rect, out_bstrides));

    size_t src_bstrides[DIM];
    const char* src_bytes = reinterpret_cast<const char*>(source_acc.ptr(src_rect, src_bstrides));

    Pitches<DIM - 1> pitches;
    const size_t volume = pitches.flatten(src_rect);
    if (volume == 0) {
      return;
    }

    const size_t elem_size = args.source.type().size();
    LEGATE_ASSERT(elem_size == args.out.type().size());

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    bool out_dense     = is_dense_row_major<N>(out_bstrides, out_rect, elem_size);
    bool indices_dense = true;
    bool source_dense  = is_dense_row_major<DIM>(src_bstrides, src_rect, elem_size);
#else
    bool out_dense     = false;
    bool indices_dense = false;
    bool source_dense  = false;
#endif

    std::vector<AccessorRO<VAL, DIM>> index_arrays;
    for (uint32_t i = 0; i < args.inputs.size(); ++i) {
      auto input_rect = args.inputs[i].shape<DIM>();
      LEGATE_ASSERT(input_rect == src_rect);
      index_arrays.push_back(args.inputs[i].read_accessor<VAL, DIM>(input_rect));
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
      indices_dense = indices_dense && index_arrays.back().accessor.is_dense_row_major(input_rect);
#endif
    }

    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    const bool dense_path = out_dense && indices_dense && source_dense;

    Buffer<AccessorRO<VAL, DIM>, 1> index_buf;
    if (!dense_path || args.check_bounds) {
      index_buf =
        create_buffer<AccessorRO<VAL, DIM>, 1>(index_arrays.size(), Memory::Kind::GPU_FB_MEM);
      if (!index_arrays.empty()) {
        CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(index_buf.ptr(0),
                                               index_arrays.data(),
                                               sizeof(AccessorRO<VAL, DIM>) * index_arrays.size(),
                                               cudaMemcpyHostToDevice,
                                               stream));
      }
    }

    if (args.check_bounds) {
      // Identical contract as the gather pipeline: confirms each index is in
      // [0, shape[start_index + n]). `shape` here is the destination shape,
      // so out-of-bounds means an illegal write target.
      const bool oob = check_index_arrays_out_of_bounds<DIM>(index_buf,
                                                             volume,
                                                             src_rect,
                                                             pitches,
                                                             index_arrays.size(),
                                                             args.start_index,
                                                             args.shape,
                                                             stream);

      if (oob) {
        throw legate::TaskException("index is out of bounds in index array");
      }
    }

    if (dense_path) {
      // Pointer table for the dense kernel lives in zero-copy memory: small
      // (one pointer per index array), only read once on the GPU, and lets us
      // write directly from the host via subscript without a separate H2D copy.
      const auto index_buf_dense =
        create_buffer<const int64_t*, 1>(index_arrays.size(), Memory::Kind::Z_COPY_MEM);

      for (uint32_t idx = 0; idx < index_arrays.size(); ++idx) {
        index_buf_dense[idx] = index_arrays[idx].ptr(src_rect);
      }

      const auto out_byte_strides = Point<N>{out_bstrides};

      zipscatter_kernel_dense<DIM, N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(out_bytes,
                                                                                out_byte_strides,
                                                                                src_bytes,
                                                                                elem_size,
                                                                                index_buf_dense,
                                                                                pitches,
                                                                                src_rect.lo,
                                                                                index_arrays.size(),
                                                                                volume,
                                                                                args.key_dim,
                                                                                args.start_index,
                                                                                args.shape);
    } else {
      type_dispatch(args.source.type().code(),
                    ZipScatterGeneralLauncher<DIM, N>{blocks,
                                                      stream,
                                                      args.out,
                                                      args.source,
                                                      index_buf,
                                                      out_rect,
                                                      src_rect,
                                                      pitches,
                                                      index_arrays.size(),
                                                      volume,
                                                      args.key_dim,
                                                      args.start_index,
                                                      args.shape});
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

namespace detail {

void launch_local_zipscatter_gpu(TaskContext& context, ZipScatterArgs& args)
{
  // Mirror zipscatter_template's dispatch order: DIM = source/iteration dim,
  // N = indexed (out) dim.
  double_dispatch(std::max(1, args.source.dim()),
                  std::max(1, args.out.dim()),
                  ZipScatterDimDispatch<VariantKind::GPU>{context},
                  args);
}

}  // namespace detail

/*static*/ void ZipScatterTask::gpu_variant(TaskContext context)
{
  zipscatter_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
