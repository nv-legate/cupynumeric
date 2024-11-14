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
                                  VAL* a,
                                  VAL* b)
{
  const auto trans = CUBLAS_OP_N;

  auto handle = get_cusolver();
  auto stream = get_cached_stream();
  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  int32_t buffer_size;
  CHECK_CUSOLVER(getrf_buffer_size(handle, m, n, a, m, &buffer_size));

  auto ipiv   = create_buffer<int32_t>(std::min(m, n), Memory::Kind::GPU_FB_MEM);
  auto buffer = create_buffer<VAL>(buffer_size, Memory::Kind::GPU_FB_MEM);
  auto info   = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(getrf(handle, m, n, a, m, buffer.ptr(0), ipiv.ptr(0), info.ptr(0)));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    throw legate::TaskException(SolveTask::ERROR_MESSAGE);
  }

  CHECK_CUSOLVER(getrs(handle, trans, n, nrhs, a, m, ipiv.ptr(0), b, n, info.ptr(0)));

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

#ifdef DEBUG_CUPYNUMERIC
  assert(info[0] == 0);
#endif
}

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, float* a, float* b)
  {
    solve_template(
      cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs, m, n, nrhs, a, b);
  }
};

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, double* a, double* b)
  {
    solve_template(
      cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs, m, n, nrhs, a, b);
  }
};

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<float>* a, complex<float>* b)
  {
    solve_template(cusolverDnCgetrf_bufferSize,
                   cusolverDnCgetrf,
                   cusolverDnCgetrs,
                   m,
                   n,
                   nrhs,
                   reinterpret_cast<cuComplex*>(a),
                   reinterpret_cast<cuComplex*>(b));
  }
};

template <>
struct SolveImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<double>* a, complex<double>* b)
  {
    solve_template(cusolverDnZgetrf_bufferSize,
                   cusolverDnZgetrf,
                   cusolverDnZgetrs,
                   m,
                   n,
                   nrhs,
                   reinterpret_cast<cuDoubleComplex*>(a),
                   reinterpret_cast<cuDoubleComplex*>(b));
  }
};

/*static*/ void SolveTask::gpu_variant(TaskContext context)
{
  solve_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
