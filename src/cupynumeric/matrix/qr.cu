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

#include "cupynumeric/matrix/qr.h"
#include "cupynumeric/matrix/qr_template.inl"

#include "cupynumeric/cuda_help.h"
#include <vector>
namespace cupynumeric {

using namespace legate;

template <typename GeqrfBufferSize,
          typename OrgqrBufferSize,
          typename Geqrf,
          typename Orgqr,
          typename VAL>
static inline void qr_template(GeqrfBufferSize geqrf_buffer_size,
                               OrgqrBufferSize orgqr_buffer_size,
                               Geqrf geqrf,
                               Orgqr orgqr,
                               int32_t m,
                               int32_t n,
                               int32_t k,
                               const VAL* a,
                               VAL* q,
                               VAL* r)
{
  auto handle = get_cusolver();
  auto stream = get_cached_stream();

  // m>=n : a[m][n], q[m][n] r[n][n]
  // m<n  : a[m][n], q[m][m] r[m][n]

  VAL* q_tmp = q;
  // if m < n:  q is not large enough to make compute inplace -> make tmp buffer
  if (m < n) {
    auto q_copy = create_buffer<VAL>(m * n, Memory::Kind::GPU_FB_MEM);
    q_tmp       = q_copy.ptr(0);
  }

  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(q_tmp, a, sizeof(VAL) * m * n, cudaMemcpyDeviceToDevice, stream));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  auto tau  = create_buffer<VAL>(k, Memory::Kind::GPU_FB_MEM);
  auto info = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  // compute and alloc buffer for geqrf
  int32_t lwork_geqrf, lwork_orgqr;
  CHECK_CUSOLVER(geqrf_buffer_size(handle, m, n, q_tmp, m, &lwork_geqrf));
  CHECK_CUSOLVER(orgqr_buffer_size(handle, m, n, k, q_tmp, m, tau.ptr(0), &lwork_orgqr));
  int32_t lwork_total = std::max(lwork_geqrf, lwork_orgqr);

  auto buffer = create_buffer<VAL>(lwork_total, Memory::Kind::GPU_FB_MEM);

  CHECK_CUSOLVER(
    geqrf(handle, m, n, q_tmp, m, tau.ptr(0), buffer.ptr(0), lwork_total, info.ptr(0)));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    throw legate::TaskException(QrTask::ERROR_MESSAGE);
  }

  // extract R from upper triangular of geqrf result
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(r, 0, k * n * sizeof(VAL), stream));
  for (int i = 0; i < k; ++i) {
    int elements = i + 1;
    if (i == k - 1 && n > k) {
      elements = k * (n - k + 1);
    }
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      r + i * k, q_tmp + i * m, sizeof(VAL) * elements, cudaMemcpyDeviceToDevice, stream));
  }

  // assemble Q
  CHECK_CUSOLVER(
    orgqr(handle, m, k, k, q_tmp, m, tau.ptr(0), buffer.ptr(0), lwork_total, info.ptr(0)));
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    throw legate::TaskException(QrTask::ERROR_MESSAGE);
  }

  // if we used a tmp storage we still need to copy back Q
  if (q_tmp != q) {
    assert(n > m);
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemcpyAsync(q, q_tmp, sizeof(VAL) * m * m, cudaMemcpyDeviceToDevice, stream));
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

#ifdef DEBUG_CUPYNUMERIC
  assert(info[0] == 0);
#endif
}

template <>
struct QrImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  void operator()(int32_t m, int32_t n, int32_t k, const float* a, float* q, float* r)
  {
    qr_template(cusolverDnSgeqrf_bufferSize,
                cusolverDnSorgqr_bufferSize,
                cusolverDnSgeqrf,
                cusolverDnSorgqr,
                m,
                n,
                k,
                a,
                q,
                r);
  }
};

template <>
struct QrImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  void operator()(int32_t m, int32_t n, int32_t k, const double* a, double* q, double* r)
  {
    qr_template(cusolverDnDgeqrf_bufferSize,
                cusolverDnDorgqr_bufferSize,
                cusolverDnDgeqrf,
                cusolverDnDorgqr,
                m,
                n,
                k,
                a,
                q,
                r);
  }
};

template <>
struct QrImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  void operator()(
    int32_t m, int32_t n, int32_t k, const complex<float>* a, complex<float>* q, complex<float>* r)
  {
    qr_template(cusolverDnCgeqrf_bufferSize,
                cusolverDnCungqr_bufferSize,
                cusolverDnCgeqrf,
                cusolverDnCungqr,
                m,
                n,
                k,
                reinterpret_cast<const cuComplex*>(a),
                reinterpret_cast<cuComplex*>(q),
                reinterpret_cast<cuComplex*>(r));
  }
};

template <>
struct QrImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  void operator()(int32_t m,
                  int32_t n,
                  int32_t k,
                  const complex<double>* a,
                  complex<double>* q,
                  complex<double>* r)
  {
    qr_template(cusolverDnZgeqrf_bufferSize,
                cusolverDnZungqr_bufferSize,
                cusolverDnZgeqrf,
                cusolverDnZungqr,
                m,
                n,
                k,
                reinterpret_cast<const cuDoubleComplex*>(a),
                reinterpret_cast<cuDoubleComplex*>(q),
                reinterpret_cast<cuDoubleComplex*>(r));
  }
};

/*static*/ void QrTask::gpu_variant(TaskContext context) { qr_template<VariantKind::GPU>(context); }

}  // namespace cupynumeric
