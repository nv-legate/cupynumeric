/* Copyright 2025 NVIDIA Corporation
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

#include "cupynumeric/matrix/potrs.h"
#include "cupynumeric/matrix/potrs_template.inl"

#include "cupynumeric/cuda_help.h"

#include <vector>

namespace cupynumeric {

using namespace legate;

template <typename Potrs, typename VAL>
static inline void potrs_template(Potrs potrs,
                                  const VAL* a,
                                  VAL* x,
                                  int32_t m,
                                  int32_t n,
                                  cudaStream_t stream,
                                  int32_t num_blocks,
                                  int64_t a_block_stride,
                                  int64_t x_block_stride,
                                  bool lower)
{
  auto handle = get_cusolver();
  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  auto uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  auto info = create_buffer<int32_t>(num_blocks, Memory::Kind::Z_COPY_MEM);

  // Batched solve - loop over batches using single API
  // Note: cuSOLVER batched potrs API only supports nrhs=1,
  // so we manually loop for n > 1
  for (int32_t i = 0; i < num_blocks; ++i) {
    CHECK_CUSOLVER(
      potrs(handle, uplo, m, n, a + i * a_block_stride, m, x + i * x_block_stride, m, info.ptr(i)));
  }

  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (int32_t i = 0; i < num_blocks; ++i) {
    if (info[i] != 0) {
      std::stringstream ss;
      ss << "Incorrect value in potrs() " << std::abs(info[i]) << "-th argument in batch " << i
         << ".";
      throw legate::TaskException(ss.str());
    }
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <typename PotrsBatched, typename VAL>
static inline void potrs_batched_template(PotrsBatched potrs_batched,
                                          const VAL* a,
                                          VAL* x,
                                          int32_t m,
                                          int32_t n,
                                          cudaStream_t stream,
                                          int32_t num_blocks,
                                          int64_t a_block_stride,
                                          int64_t x_block_stride,
                                          bool lower)
{
  auto handle = get_cusolver();
  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  auto uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  // Create pointer arrays in zero-copy memory
  Buffer<VAL*> a_array = create_buffer<VAL*>(num_blocks, legate::Memory::Z_COPY_MEM);
  Buffer<VAL*> x_array = create_buffer<VAL*>(num_blocks, legate::Memory::Z_COPY_MEM);

  for (int32_t i = 0; i < num_blocks; ++i) {
    a_array[i] = const_cast<VAL*>(a) + i * a_block_stride;
    x_array[i] = x + i * x_block_stride;
  }

  auto info = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  // Call batched POTRS (only supports nrhs=1)
  CHECK_CUSOLVER(potrs_batched(
    handle, uplo, m, n, a_array.ptr(0), m, x_array.ptr(0), m, info.ptr(0), num_blocks));

  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    std::stringstream ss;
    ss << "Incorrect value in batched potrs() " << std::abs(info[0]) << "-th argument.";
    throw legate::TaskException(ss.str());
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
}

template <>
struct PotrsImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit PotrsImplBody(TaskContext context) : context(context) {}

  void operator()(const float* a,
                  float* x,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t x_block_stride,
                  bool lower)
  {
    auto stream = context.get_task_stream();
    // Use batched API when num_blocks > 1 and nrhs == 1 (batched API limitation)
    // TODO: for large num_blocks > 1 and n > 1 we might want to try 2calls to trsm to see if it's
    // faster
    if (num_blocks > 1 && n == 1) {
      potrs_batched_template(cusolverDnSpotrsBatched,
                             a,
                             x,
                             m,
                             n,
                             stream,
                             num_blocks,
                             a_block_stride,
                             x_block_stride,
                             lower);
    } else {
      potrs_template(
        cusolverDnSpotrs, a, x, m, n, stream, num_blocks, a_block_stride, x_block_stride, lower);
    }
  }
};

template <>
struct PotrsImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit PotrsImplBody(TaskContext context) : context(context) {}

  void operator()(const double* a,
                  double* x,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t x_block_stride,
                  bool lower)
  {
    auto stream = context.get_task_stream();
    // Use batched API when num_blocks > 1 and nrhs == 1 (batched API limitation)
    // TODO: for large num_blocks > 1 and n > 1 we might want to try 2calls to trsm to see if it's
    // faster
    if (num_blocks > 1 && n == 1) {
      potrs_batched_template(cusolverDnDpotrsBatched,
                             a,
                             x,
                             m,
                             n,
                             stream,
                             num_blocks,
                             a_block_stride,
                             x_block_stride,
                             lower);
    } else {
      potrs_template(
        cusolverDnDpotrs, a, x, m, n, stream, num_blocks, a_block_stride, x_block_stride, lower);
    }
  }
};

template <>
struct PotrsImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit PotrsImplBody(TaskContext context) : context(context) {}

  void operator()(const legate::Complex<float>* a_,
                  legate::Complex<float>* x_,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t x_block_stride,
                  bool lower)
  {
    auto stream = context.get_task_stream();
    auto a      = reinterpret_cast<const cuComplex*>(a_);
    auto x      = reinterpret_cast<cuComplex*>(x_);

    // Use batched API when num_blocks > 1 and nrhs == 1 (batched API limitation)
    // TODO: for large num_blocks > 1 and n > 1 we might want to try 2calls to trsm to see if it's
    // faster
    if (num_blocks > 1 && n == 1) {
      potrs_batched_template(cusolverDnCpotrsBatched,
                             a,
                             x,
                             m,
                             n,
                             stream,
                             num_blocks,
                             a_block_stride,
                             x_block_stride,
                             lower);
    } else {
      potrs_template(
        cusolverDnCpotrs, a, x, m, n, stream, num_blocks, a_block_stride, x_block_stride, lower);
    }
  }
};

template <>
struct PotrsImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit PotrsImplBody(TaskContext context) : context(context) {}

  void operator()(const legate::Complex<double>* a_,
                  legate::Complex<double>* x_,
                  int32_t m,
                  int32_t n,
                  int32_t num_blocks,
                  int64_t a_block_stride,
                  int64_t x_block_stride,
                  bool lower)
  {
    auto stream = context.get_task_stream();
    auto a      = reinterpret_cast<const cuDoubleComplex*>(a_);
    auto x      = reinterpret_cast<cuDoubleComplex*>(x_);

    // Use batched API when num_blocks > 1 and nrhs == 1 (batched API limitation)
    // TODO: for large num_blocks > 1 and n > 1 we might want to try 2calls to trsm to see if it's
    // faster
    if (num_blocks > 1 && n == 1) {
      potrs_batched_template(cusolverDnZpotrsBatched,
                             a,
                             x,
                             m,
                             n,
                             stream,
                             num_blocks,
                             a_block_stride,
                             x_block_stride,
                             lower);
    } else {
      potrs_template(
        cusolverDnZpotrs, a, x, m, n, stream, num_blocks, a_block_stride, x_block_stride, lower);
    }
  }
};

/*static*/ void PotrsTask::gpu_variant(TaskContext context)
{
  potrs_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
