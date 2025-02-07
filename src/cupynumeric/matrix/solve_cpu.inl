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

#include <cblas.h>
#include <lapack.h>

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::FLOAT32> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, float* a, float* b)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    int32_t info = 0;
    LAPACK_sgesv(&n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) {
      throw legate::TaskException(SolveTask::ERROR_MESSAGE);
    }
  }
};

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::FLOAT64> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, double* a, double* b)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    int32_t info = 0;
    LAPACK_dgesv(&n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) {
      throw legate::TaskException(SolveTask::ERROR_MESSAGE);
    }
  }
};

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::COMPLEX64> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<float>* a_, complex<float>* b_)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    auto a = reinterpret_cast<__complex__ float*>(a_);
    auto b = reinterpret_cast<__complex__ float*>(b_);

    int32_t info = 0;
    LAPACK_cgesv(&n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) {
      throw legate::TaskException(SolveTask::ERROR_MESSAGE);
    }
  }
};

template <VariantKind KIND>
struct SolveImplBody<KIND, Type::Code::COMPLEX128> {
  void operator()(int32_t m, int32_t n, int32_t nrhs, complex<double>* a_, complex<double>* b_)
  {
    auto ipiv = create_buffer<int32_t>(std::min(m, n));

    auto a = reinterpret_cast<__complex__ double*>(a_);
    auto b = reinterpret_cast<__complex__ double*>(b_);

    int32_t info = 0;
    LAPACK_zgesv(&n, &nrhs, a, &m, ipiv.ptr(0), b, &n, &info);

    if (info != 0) {
      throw legate::TaskException(SolveTask::ERROR_MESSAGE);
    }
  }
};

}  // namespace cupynumeric
