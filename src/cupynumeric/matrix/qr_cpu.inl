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

#include "cupynumeric/utilities/blas_lapack.h"
#include <cstring>

namespace cupynumeric {

using namespace legate;

template <typename Geqrf, typename Orgqr, typename VAL>
static inline void qr_template(
  Geqrf geqrf, Orgqr orgqr, int32_t m, int32_t n, int32_t k, const VAL* a, VAL* q, VAL* r)
{
  int32_t info = 0;

  // m>=n : a[m][n], q[m][n] r[n][n]
  // m<n  : a[m][n], q[m][m] r[m][n]

  VAL* q_tmp = q;
  // if m < n:  q is not large enough to make compute inplace -> make tmp buffer
  if (m < n) {
    auto q_copy = create_buffer<VAL>(m * n);
    q_tmp       = q_copy.ptr(0);
  }

  std::memcpy(q_tmp, a, m * n * sizeof(VAL));

  // compute and alloc buffer for geqrf
  int32_t lwork = n;
  auto buffer   = create_buffer<VAL>(lwork);
  auto tau      = create_buffer<VAL>(k);

  geqrf(&m, &n, q_tmp, &m, tau.ptr(0), buffer.ptr(0), &lwork, &info);

  if (info != 0) {
    throw legate::TaskException(QrTask::ERROR_MESSAGE);
  }

  // extract R from upper triangular of getrf result
  std::memset(r, 0, k * n * sizeof(VAL));
  for (int i = 0; i < k; ++i) {
    int elements = i + 1;
    if (i == k - 1 && n > k) {
      elements = k * (n - k + 1);
    }
    std::memcpy(r + i * k, q_tmp + i * m, sizeof(VAL) * elements);
  }

  // assemble Q
  orgqr(&m, &k, &k, q_tmp, &m, tau.ptr(0), buffer.ptr(0), &lwork, &info);
  if (info != 0) {
    throw legate::TaskException(QrTask::ERROR_MESSAGE);
  }

  // if we used a tmp storage we still need to copy back Q
  if (q_tmp != q) {
    assert(n > m);
    std::memcpy(q, q_tmp, sizeof(VAL) * m * m);
  }
}

template <VariantKind KIND>
struct QrImplBody<KIND, Type::Code::FLOAT32> {
  TaskContext context;
  explicit QrImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m, int32_t n, int32_t k, const float* a, float* q, float* r)
  {
    qr_template(sgeqrf_, sorgqr_, m, n, k, a, q, r);
  }
};

template <VariantKind KIND>
struct QrImplBody<KIND, Type::Code::FLOAT64> {
  TaskContext context;
  explicit QrImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m, int32_t n, int32_t k, const double* a, double* q, double* r)
  {
    qr_template(dgeqrf_, dorgqr_, m, n, k, a, q, r);
  }
};

template <VariantKind KIND>
struct QrImplBody<KIND, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit QrImplBody(TaskContext context) : context(context) {}

  void operator()(
    int32_t m, int32_t n, int32_t k, const complex<float>* a, complex<float>* q, complex<float>* r)
  {
    qr_template(cgeqrf_,
                cungqr_,
                m,
                n,
                k,
                reinterpret_cast<const __complex__ float*>(a),
                reinterpret_cast<__complex__ float*>(q),
                reinterpret_cast<__complex__ float*>(r));
  }
};

template <VariantKind KIND>
struct QrImplBody<KIND, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit QrImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m,
                  int32_t n,
                  int32_t k,
                  const complex<double>* a,
                  complex<double>* q,
                  complex<double>* r)
  {
    qr_template(zgeqrf_,
                zungqr_,
                m,
                n,
                k,
                reinterpret_cast<const __complex__ double*>(a),
                reinterpret_cast<__complex__ double*>(q),
                reinterpret_cast<__complex__ double*>(r));
  }
};

}  // namespace cupynumeric
