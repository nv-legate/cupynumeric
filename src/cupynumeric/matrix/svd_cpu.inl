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

template <VariantKind KIND>
struct SvdImplBody<KIND, Type::Code::FLOAT32> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m,
                  int32_t n,
                  int32_t k,
                  bool full_matrices,
                  const float* a,
                  float* u,
                  float* s,
                  float* vh)
  {
    auto a_copy = create_buffer<float>(m * n);
    std::copy(a, a + (n * m), a_copy.ptr(0));

    int32_t info  = 0;
    float wkopt   = 0;
    int32_t lwork = -1;
    sgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            a_copy.ptr(0),
            &m,
            s,
            u,
            &m,
            vh,
            &n,
            &wkopt,
            &lwork,
            &info);
    lwork = (int)wkopt;

    std::vector<float> work_tmp(lwork);
    sgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            a_copy.ptr(0),
            &m,
            s,
            u,
            &m,
            vh,
            &n,
            work_tmp.data(),
            &lwork,
            &info);

    if (info != 0) {
      throw legate::TaskException(SvdTask::ERROR_MESSAGE);
    }
  }
};

template <VariantKind KIND>
struct SvdImplBody<KIND, Type::Code::FLOAT64> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m,
                  int32_t n,
                  int32_t k,
                  bool full_matrices,
                  const double* a,
                  double* u,
                  double* s,
                  double* vh)
  {
    auto a_copy = create_buffer<double>(m * n);
    std::copy(a, a + (n * m), a_copy.ptr(0));

    int32_t info  = 0;
    double wkopt  = 0;
    int32_t lwork = -1;

    dgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            a_copy.ptr(0),
            &m,
            s,
            u,
            &m,
            vh,
            &n,
            &wkopt,
            &lwork,
            &info);
    lwork = (int)wkopt;

    std::vector<double> work_tmp(lwork);
    dgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            a_copy.ptr(0),
            &m,
            s,
            u,
            &m,
            vh,
            &n,
            work_tmp.data(),
            &lwork,
            &info);

    if (info != 0) {
      throw legate::TaskException(SvdTask::ERROR_MESSAGE);
    }
  }
};

template <VariantKind KIND>
struct SvdImplBody<KIND, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m,
                  int32_t n,
                  int32_t k,
                  bool full_matrices,
                  const complex<float>* a,
                  complex<float>* u,
                  float* s,
                  complex<float>* vh)
  {
    auto a_copy = create_buffer<complex<float>>(m * n);
    std::copy(a, a + (n * m), a_copy.ptr(0));

    int32_t info            = 0;
    int32_t lwork           = -1;
    __complex__ float wkopt = 0;
    std::vector<float> rwork(5 * k);

    cgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            reinterpret_cast<__complex__ float*>(a_copy.ptr(0)),
            &m,
            s,
            reinterpret_cast<__complex__ float*>(u),
            &m,
            reinterpret_cast<__complex__ float*>(vh),
            &n,
            &wkopt,
            &lwork,
            rwork.data(),
            &info);

    lwork = (int)(*((float*)&(wkopt)));

    std::vector<__complex__ float> work_tmp(lwork);
    cgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            reinterpret_cast<__complex__ float*>(a_copy.ptr(0)),
            &m,
            s,
            reinterpret_cast<__complex__ float*>(u),
            &m,
            reinterpret_cast<__complex__ float*>(vh),
            &n,
            work_tmp.data(),
            &lwork,
            rwork.data(),
            &info);

    if (info != 0) {
      throw legate::TaskException(SvdTask::ERROR_MESSAGE);
    }
  }
};

template <VariantKind KIND>
struct SvdImplBody<KIND, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit SvdImplBody(TaskContext context) : context(context) {}

  void operator()(int32_t m,
                  int32_t n,
                  int32_t k,
                  bool full_matrices,
                  const complex<double>* a,
                  complex<double>* u,
                  double* s,
                  complex<double>* vh)
  {
    auto a_copy = create_buffer<complex<double>>(m * n);
    std::copy(a, a + (n * m), a_copy.ptr(0));

    int32_t info             = 0;
    int32_t lwork            = -1;
    __complex__ double wkopt = 0;
    std::vector<double> rwork(5 * k);
    zgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            reinterpret_cast<__complex__ double*>(a_copy.ptr(0)),
            &m,
            s,
            reinterpret_cast<__complex__ double*>(u),
            &m,
            reinterpret_cast<__complex__ double*>(vh),
            &n,
            &wkopt,
            &lwork,
            rwork.data(),
            &info);

    lwork = (int)(*((double*)&(wkopt)));

    std::vector<__complex__ double> work_tmp(lwork);
    zgesvd_(full_matrices ? "A" : "S",
            "A",
            &m,
            &n,
            reinterpret_cast<__complex__ double*>(a_copy.ptr(0)),
            &m,
            s,
            reinterpret_cast<__complex__ double*>(u),
            &m,
            reinterpret_cast<__complex__ double*>(vh),
            &n,
            work_tmp.data(),
            &lwork,
            rwork.data(),
            &info);

    if (info != 0) {
      throw legate::TaskException(SvdTask::ERROR_MESSAGE);
    }
  }
};

}  // namespace cupynumeric
