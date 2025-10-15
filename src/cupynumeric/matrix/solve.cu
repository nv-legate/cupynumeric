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

#include "cupynumeric/matrix/solve.h"
#include "cupynumeric/matrix/solve_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename GetrfBufferSize, typename Getrf, typename Getrs, typename VAL>
static inline void solve_template(GetrfBufferSize getrf_buffer_size,
                                  Getrf getrf,
                                  Getrs getrs,
                                  int32_t m,
                                  int32_t n,
                                  int32_t nrhs,
                                  const VAL* a,
                                  const VAL* b,
                                  VAL* x,
                                  cudaStream_t stream)
{
  const auto trans = CUBLAS_OP_N;

  auto handle = get_cusolver();

  // copy inputs for in-place compute
  auto a_copy = create_buffer<VAL>(int64_t(m) * n, Memory::Kind::GPU_FB_MEM);
  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(a_copy.ptr(0), a, sizeof(VAL) * m * n, cudaMemcpyDeviceToDevice, stream));
  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(x, b, sizeof(VAL) * m * nrhs, cudaMemcpyDeviceToDevice, stream));

  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  int32_t buffer_size;
  CHECK_CUSOLVER(getrf_buffer_size(handle, m, n, a_copy.ptr(0), m, &buffer_size));

  auto ipiv   = create_buffer<int32_t>(std::min(m, n), Memory::Kind::GPU_FB_MEM);
  auto buffer = create_buffer<VAL>(buffer_size, Memory::Kind::GPU_FB_MEM);
  auto info   = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(getrf(handle, m, n, a_copy.ptr(0), m, buffer.ptr(0), ipiv.ptr(0), info.ptr(0)));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    throw legate::TaskException(SolveTask::ERROR_MESSAGE);
  }

  CHECK_CUSOLVER(getrs(handle, trans, n, nrhs, a_copy.ptr(0), m, ipiv.ptr(0), x, n, info.ptr(0)));

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

#ifdef DEBUG_CUPYNUMERIC
  assert(info[0] == 0);
#endif
}

template <typename GetrfBatched, typename GetrsBatched, typename VAL>
static inline void solve_template_batched(GetrfBatched getrfbatched,
                                          GetrsBatched getrsbatched,
                                          int32_t batchsize,
                                          int32_t n,
                                          int32_t nrhs,
                                          const VAL* a,
                                          const VAL* b,
                                          VAL* x,
                                          cudaStream_t stream)
{
  auto cublas_handle = get_cublas();

  // copy inputs for in-place compute
  auto a_copy = create_buffer<VAL>(int64_t(batchsize) * n * n, Memory::Kind::GPU_FB_MEM);
  CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
    a_copy.ptr(0), a, sizeof(VAL) * batchsize * n * n, cudaMemcpyDeviceToDevice, stream));
  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(x, b, sizeof(VAL) * batchsize * n * nrhs, cudaMemcpyDeviceToDevice, stream));

  CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

  Buffer<VAL*> aArray = create_buffer<VAL*>(batchsize, legate::Memory::Z_COPY_MEM);
  Buffer<VAL*> bArray = create_buffer<VAL*>(batchsize, legate::Memory::Z_COPY_MEM);
  for (int i = 0; i < batchsize; ++i) {
    aArray[i] = a_copy.ptr(0) + i * n * n;
    bArray[i] = x + i * n * nrhs;
  }

  auto ipiv = create_buffer<int32_t>(int64_t(n) * batchsize, Memory::Kind::GPU_FB_MEM);

  auto info = create_buffer<int32_t>(batchsize, Memory::Kind::Z_COPY_MEM);
  CHECK_CUBLAS(
    getrfbatched(cublas_handle, n, aArray.ptr(0), n, ipiv.ptr(0), info.ptr(0), batchsize));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (int i = 0; i < batchsize; ++i) {
    if (info[i] != 0) {
      throw legate::TaskException(SolveTask::ERROR_MESSAGE);
    }
  }

  const auto trans = CUBLAS_OP_N;
  CHECK_CUBLAS(getrsbatched(cublas_handle,
                            trans,
                            n,
                            nrhs,
                            aArray.ptr(0),
                            n,
                            ipiv.ptr(0),
                            bArray.ptr(0),
                            n,
                            info.ptr(0),
                            batchsize));

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

#ifdef DEBUG_CUPYNUMERIC
  for (int i = 0; i < batchsize; ++i) {
    assert(info[i] == 0);
  }
#endif
}

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(
    int32_t batchsize, int32_t m, int32_t n, int32_t nrhs, const float* a, const float* b, float* x)
  {
    auto stream = context.get_task_stream();
    if (batchsize > 1) {
      solve_template_batched(
        cublasSgetrfBatched, cublasSgetrsBatched, batchsize, n, nrhs, a, b, x, stream);
    } else {
      solve_template(cusolverDnSgetrf_bufferSize,
                     cusolverDnSgetrf,
                     cusolverDnSgetrs,
                     m,
                     n,
                     nrhs,
                     a,
                     b,
                     x,
                     stream);
    }
  }
};

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t batchsize,
                  int32_t m,
                  int32_t n,
                  int32_t nrhs,
                  const double* a,
                  const double* b,
                  double* x)
  {
    auto stream = context.get_task_stream();
    if (batchsize > 1) {
      solve_template_batched(
        cublasDgetrfBatched, cublasDgetrsBatched, batchsize, n, nrhs, a, b, x, stream);
    } else {
      solve_template(cusolverDnDgetrf_bufferSize,
                     cusolverDnDgetrf,
                     cusolverDnDgetrs,
                     m,
                     n,
                     nrhs,
                     a,
                     b,
                     x,
                     stream);
    }
  }
};

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t batchsize,
                  int32_t m,
                  int32_t n,
                  int32_t nrhs,
                  const complex<float>* a,
                  const complex<float>* b,
                  complex<float>* x)
  {
    auto stream = context.get_task_stream();
    if (batchsize > 1) {
      solve_template_batched(cublasCgetrfBatched,
                             cublasCgetrsBatched,
                             batchsize,
                             n,
                             nrhs,
                             reinterpret_cast<const cuComplex*>(a),
                             reinterpret_cast<const cuComplex*>(b),
                             reinterpret_cast<cuComplex*>(x),
                             stream);
    } else {
      solve_template(cusolverDnCgetrf_bufferSize,
                     cusolverDnCgetrf,
                     cusolverDnCgetrs,
                     m,
                     n,
                     nrhs,
                     reinterpret_cast<const cuComplex*>(a),
                     reinterpret_cast<const cuComplex*>(b),
                     reinterpret_cast<cuComplex*>(x),
                     stream);
    }
  }
};

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t batchsize,
                  int32_t m,
                  int32_t n,
                  int32_t nrhs,
                  const complex<double>* a,
                  const complex<double>* b,
                  complex<double>* x)
  {
    auto stream = context.get_task_stream();
    if (batchsize > 1) {
      solve_template_batched(cublasZgetrfBatched,
                             cublasZgetrsBatched,
                             batchsize,
                             n,
                             nrhs,
                             reinterpret_cast<const cuDoubleComplex*>(a),
                             reinterpret_cast<const cuDoubleComplex*>(b),
                             reinterpret_cast<cuDoubleComplex*>(x),
                             stream);
    } else {
      solve_template(cusolverDnZgetrf_bufferSize,
                     cusolverDnZgetrf,
                     cusolverDnZgetrs,
                     m,
                     n,
                     nrhs,
                     reinterpret_cast<const cuDoubleComplex*>(a),
                     reinterpret_cast<const cuDoubleComplex*>(b),
                     reinterpret_cast<cuDoubleComplex*>(x),
                     stream);
    }
  }
};

/*static*/ void SolveTask::gpu_variant(TaskContext context)
{
  solve_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
