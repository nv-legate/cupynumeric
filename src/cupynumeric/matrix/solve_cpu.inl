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

#include <cstring>
#include "cupynumeric/utilities/blas_lapack.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::FLOAT32> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(
    int32_t batchsize, int32_t m, int32_t n, int32_t nrhs, const float* a, const float* b, float* x)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    auto a_copy = create_buffer<float>(m * n);
    std::memcpy(x, b, batchsize * m * nrhs * sizeof(float));

    int32_t info = 0;
    for (int i = 0; i < batchsize; ++i) {
      std::memcpy(a_copy.ptr(0), a + i * m * n, m * n * sizeof(float));
      sgesv_(&n, &nrhs, a_copy.ptr(0), &m, ipiv.ptr(0), x + i * m * nrhs, &n, &info);
      if (info != 0) {
        throw legate::TaskException(SolveTask::ERROR_MESSAGE);
      }
    }
  }
};

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::FLOAT64> {
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
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    auto a_copy = create_buffer<double>(m * n);
    std::memcpy(x, b, batchsize * m * nrhs * sizeof(double));

    int32_t info = 0;
    for (int i = 0; i < batchsize; ++i) {
      std::memcpy(a_copy.ptr(0), a + i * m * n, m * n * sizeof(double));
      dgesv_(&n, &nrhs, a_copy.ptr(0), &m, ipiv.ptr(0), x + i * m * nrhs, &n, &info);
      if (info != 0) {
        throw legate::TaskException(SolveTask::ERROR_MESSAGE);
      }
    }
  }
};

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t batchsize,
                  int32_t m,
                  int32_t n,
                  int32_t nrhs,
                  const legate::Complex<float>* a_,
                  const legate::Complex<float>* b_,
                  legate::Complex<float>* x_)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    auto a_copy = create_buffer<legate::Complex<float>>(m * n);
    std::memcpy(x_, b_, batchsize * m * nrhs * sizeof(legate::Complex<float>));

    auto a = reinterpret_cast<__complex__ float*>(a_copy.ptr(0));
    auto b = reinterpret_cast<__complex__ float*>(x_);

    int32_t info = 0;
    for (int i = 0; i < batchsize; ++i) {
      std::memcpy(a_copy.ptr(0), a_ + i * m * n, m * n * sizeof(legate::Complex<float>));
      cgesv_(&n, &nrhs, a, &m, ipiv.ptr(0), b + i * m * nrhs, &n, &info);
      if (info != 0) {
        throw legate::TaskException(SolveTask::ERROR_MESSAGE);
      }
    }
  }
};

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit SolveImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t batchsize,
                  int32_t m,
                  int32_t n,
                  int32_t nrhs,
                  const legate::Complex<double>* a_,
                  const legate::Complex<double>* b_,
                  legate::Complex<double>* x_)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    auto a_copy = create_buffer<legate::Complex<double>>(m * n);
    std::memcpy(x_, b_, batchsize * m * nrhs * sizeof(legate::Complex<double>));

    auto a = reinterpret_cast<__complex__ double*>(a_copy.ptr(0));
    auto b = reinterpret_cast<__complex__ double*>(x_);

    int32_t info = 0;
    for (int i = 0; i < batchsize; ++i) {
      std::memcpy(a_copy.ptr(0), a_ + i * m * n, m * n * sizeof(legate::Complex<double>));
      zgesv_(&n, &nrhs, a, &m, ipiv.ptr(0), b + i * m * nrhs, &n, &info);
      if (info != 0) {
        throw legate::TaskException(SolveTask::ERROR_MESSAGE);
      }
    }
  }
};

}  // namespace cupynumeric
