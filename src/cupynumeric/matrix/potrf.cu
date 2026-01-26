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

#include "cupynumeric/matrix/potrf.h"
#include "cupynumeric/matrix/potrf_template.inl"

#include "cupynumeric/cuda_help.h"

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

namespace cupynumeric {

using namespace legate;

template <>
void CopyBlockImpl<VariantKind::GPU>::operator()(void* dst, const void* src, size_t size)
{
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, context.get_task_stream());
}

template <typename PotrfBufferSize, typename Potrf, typename VAL>
static inline void potrf_template(PotrfBufferSize potrfBufferSize,
                                  Potrf potrf,
                                  VAL* array,
                                  int32_t m,
                                  int32_t lda,
                                  bool lower,
                                  cudaStream_t stream)
{
  auto uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  auto cu_context = get_cusolver();
  CHECK_CUSOLVER(cusolverDnSetStream(cu_context, stream));

  int32_t bufferSize;
  CHECK_CUSOLVER(potrfBufferSize(cu_context, uplo, m, array, lda, &bufferSize));

  auto buffer = create_buffer<VAL>(bufferSize, Memory::Kind::GPU_FB_MEM);
  auto info   = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(potrf(cu_context, uplo, m, array, lda, buffer.ptr(0), bufferSize, info.ptr(0)));

  // TODO: We need a deferred exception to avoid this synchronization
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  if (info[0] != 0) {
    throw legate::TaskException("Matrix is not positive definite");
  }
}

template <typename PotrfBatched, typename VAL>
static inline void batched_potrf_template(PotrfBatched potrfBatched,
                                          VAL* array,
                                          int32_t m,
                                          int32_t lda,
                                          bool lower,
                                          cudaStream_t stream,
                                          int32_t num_blocks,
                                          int64_t block_stride)
{
  auto uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  auto handle = get_cusolver();
  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  // Create pointer arrays in zero-copy memory
  Buffer<VAL*> a_array = create_buffer<VAL*>(num_blocks, legate::Memory::Z_COPY_MEM);

  for (int32_t i = 0; i < num_blocks; ++i) {
    a_array[i] = const_cast<VAL*>(array) + i * block_stride;
  }

  auto info = create_buffer<int32_t>(num_blocks, Memory::Kind::Z_COPY_MEM);

  // Call batched POTRF
  CHECK_CUSOLVER(potrfBatched(handle, uplo, m, a_array.ptr(0), lda, info.ptr(0), num_blocks));

  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (int32_t i = 0; i < num_blocks; ++i) {
    if (info[i] != 0) {
      throw legate::TaskException("Matrix is not positive definite");
    }
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::FLOAT32>::operator()(
  float* array, int32_t n, int32_t lda, bool lower, int32_t num_blocks, int64_t block_stride)
{
  auto stream = context.get_task_stream();
  if (num_blocks > 1) {
    batched_potrf_template(
      cusolverDnSpotrfBatched, array, n, lda, lower, stream, num_blocks, block_stride);
  } else {
    potrf_template(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, array, n, lda, lower, stream);
  }
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::FLOAT64>::operator()(
  double* array, int32_t n, int32_t lda, bool lower, int32_t num_blocks, int64_t block_stride)
{
  auto stream = context.get_task_stream();
  if (num_blocks > 1) {
    batched_potrf_template(
      cusolverDnDpotrfBatched, array, n, lda, lower, stream, num_blocks, block_stride);
  } else {
    potrf_template(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, array, n, lda, lower, stream);
  }
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX64>::operator()(
  legate::Complex<float>* array,
  int32_t n,
  int32_t lda,
  bool lower,
  int32_t num_blocks,
  int64_t block_stride)
{
  auto stream = context.get_task_stream();
  if (num_blocks > 1) {
    batched_potrf_template(cusolverDnCpotrfBatched,
                           reinterpret_cast<cuComplex*>(array),
                           n,
                           lda,
                           lower,
                           stream,
                           num_blocks,
                           block_stride);
  } else {
    potrf_template(cusolverDnCpotrf_bufferSize,
                   cusolverDnCpotrf,
                   reinterpret_cast<cuComplex*>(array),
                   n,
                   lda,
                   lower,
                   stream);
  }
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX128>::operator()(
  legate::Complex<double>* array,
  int32_t n,
  int32_t lda,
  bool lower,
  int32_t num_blocks,
  int64_t block_stride)
{
  auto stream = context.get_task_stream();
  if (num_blocks > 1) {
    batched_potrf_template(cusolverDnZpotrfBatched,
                           reinterpret_cast<cuDoubleComplex*>(array),
                           n,
                           lda,
                           lower,
                           stream,
                           num_blocks,
                           block_stride);
  } else {
    potrf_template(cusolverDnZpotrf_bufferSize,
                   cusolverDnZpotrf,
                   reinterpret_cast<cuDoubleComplex*>(array),
                   n,
                   lda,
                   lower,
                   stream);
  }
}

template <typename VAL>
struct ZeroTriangleFunctor {
  VAL* data;
  int32_t n;
  bool lower;
  int64_t block_stride;

  ZeroTriangleFunctor(VAL* data, int32_t n, bool lower, int64_t block_stride)
    : data(data), n(n), lower(lower), block_stride(block_stride)
  {
  }

  __device__ void operator()(int64_t idx) const
  {
    // Decompose idx into batch_idx, row, col
    auto batch_idx    = idx / block_stride;
    auto idx_in_block = idx % block_stride;
    auto row          = idx_in_block % n;
    auto col          = idx_in_block / n;

    auto* block_data = data + batch_idx * block_stride;

    // Zero out based on triangle type
    if (lower) {
      // Zero out upper triangle (elements where row < col)
      if (row < col) {
        block_data[col * n + row] = VAL(0);
      }
    } else {
      // Zero out lower triangle (elements where row > col)
      if (row > col) {
        block_data[col * n + row] = VAL(0);
      }
    }
  }
};

template <Type::Code CODE>
struct BatchedTriluImplBody<VariantKind::GPU, CODE> {
  TaskContext context;
  explicit BatchedTriluImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(VAL* out, int32_t n, bool lower, int32_t num_blocks, int64_t block_stride) const
  {
    auto stream = context.get_task_stream();

    // Zero out the lower/upper diagonal using thrust
    int64_t total_elements = static_cast<int64_t>(num_blocks) * block_stride;

    auto exec_policy = thrust::cuda::par.on(stream);
    thrust::for_each(exec_policy,
                     thrust::make_counting_iterator<int64_t>(0),
                     thrust::make_counting_iterator<int64_t>(total_elements),
                     ZeroTriangleFunctor<VAL>(out, n, lower, block_stride));

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void PotrfTask::gpu_variant(TaskContext context)
{
  potrf_task_context_dispatch<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
