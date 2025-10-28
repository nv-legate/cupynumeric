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

#include "cupynumeric/matrix/svd.h"
#include "cupynumeric/matrix/svd_template.inl"

#include "cupynumeric/cuda_help.h"
#include <vector>
namespace cupynumeric {

using namespace legate;

template <typename VAL, typename DataType>
static inline void svd_template(DataType valTypeC,
                                DataType valTypeR,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                bool full_matrices,
                                const void* a,
                                void* u,
                                void* s,
                                void* vh,
                                cudaStream_t stream)
{
  auto handle = get_cusolver();

  auto a_copy = create_buffer<VAL>(m * n, Memory::Kind::GPU_FB_MEM);
  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(a_copy.ptr(0), a, m * n * sizeof(VAL), cudaMemcpyDeviceToDevice, stream));

  // a[m][n], u[m][m] s[k] vh[n][n]
  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  size_t lwork_device, lwork_host;
  CHECK_CUSOLVER(cusolverDnXgesvd_bufferSize(handle,
                                             nullptr,
                                             full_matrices ? 'A' : 'S',
                                             'A',
                                             m,
                                             n,
                                             valTypeC,
                                             reinterpret_cast<void*>(a_copy.ptr(0)),
                                             m,
                                             valTypeR,
                                             s,
                                             valTypeC,
                                             u,
                                             m,
                                             valTypeC,
                                             vh,
                                             n,
                                             valTypeC,
                                             &lwork_device,
                                             &lwork_host));

  auto buffer = create_buffer<char>(lwork_device, Memory::Kind::GPU_FB_MEM);
  std::vector<char> buffer_host(std::max(1ul, lwork_host));
  auto info = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(cusolverDnXgesvd(handle,
                                  nullptr,
                                  full_matrices ? 'A' : 'S',
                                  'A',
                                  m,
                                  n,
                                  valTypeC,
                                  reinterpret_cast<void*>(a_copy.ptr(0)),
                                  m,
                                  valTypeR,
                                  s,
                                  valTypeC,
                                  u,
                                  m,
                                  valTypeC,
                                  vh,
                                  n,
                                  valTypeC,
                                  buffer.ptr(0),
                                  lwork_device,
                                  buffer_host.data(),
                                  lwork_host,
                                  info.ptr(0)));

  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    throw legate::TaskException(SvdTask::ERROR_MESSAGE);
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

#ifdef DEBUG_CUPYNUMERIC
  assert(info[0] == 0);
#endif
}

template <>
struct SvdImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int64_t m,
                  int64_t n,
                  int64_t k,
                  bool full_matrices,
                  const float* a,
                  float* u,
                  float* s,
                  float* vh)
  {
    auto stream = context.get_task_stream();
    svd_template<float>(CUDA_R_32F, CUDA_R_32F, m, n, k, full_matrices, a, u, s, vh, stream);
  }
};

template <>
struct SvdImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int64_t m,
                  int64_t n,
                  int64_t k,
                  bool full_matrices,
                  const double* a,
                  double* u,
                  double* s,
                  double* vh)
  {
    auto stream = context.get_task_stream();
    svd_template<double>(CUDA_R_64F, CUDA_R_64F, m, n, k, full_matrices, a, u, s, vh, stream);
  }
};

template <>
struct SvdImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int64_t m,
                  int64_t n,
                  int64_t k,
                  bool full_matrices,
                  const legate::Complex<float>* a,
                  legate::Complex<float>* u,
                  float* s,
                  legate::Complex<float>* vh)
  {
    auto stream = context.get_task_stream();
    svd_template<legate::Complex<float>>(CUDA_C_32F,
                                         CUDA_R_32F,
                                         m,
                                         n,
                                         k,
                                         full_matrices,
                                         reinterpret_cast<const cuComplex*>(a),
                                         reinterpret_cast<cuComplex*>(u),
                                         s,
                                         reinterpret_cast<cuComplex*>(vh),
                                         stream);
  }
};

template <>
struct SvdImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int64_t m,
                  int64_t n,
                  int64_t k,
                  bool full_matrices,
                  const legate::Complex<double>* a,
                  legate::Complex<double>* u,
                  double* s,
                  legate::Complex<double>* vh)
  {
    auto stream = context.get_task_stream();
    svd_template<legate::Complex<double>>(CUDA_C_64F,
                                          CUDA_R_64F,
                                          m,
                                          n,
                                          k,
                                          full_matrices,
                                          reinterpret_cast<const cuDoubleComplex*>(a),
                                          reinterpret_cast<cuDoubleComplex*>(u),
                                          s,
                                          reinterpret_cast<cuDoubleComplex*>(vh),
                                          stream);
  }
};

/*static*/ void SvdTask::gpu_variant(TaskContext context)
{
  svd_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
