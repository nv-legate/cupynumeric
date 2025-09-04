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

namespace {

template <typename T>
void remove_diag_imag(complex<T>* a_inout, size_t m)
{
  for (int i = 0; i < m; ++i) {
    a_inout[i * m + i].imag(T(0));
  }
}
}  // namespace

template <VariantKind KIND>
struct SyevImplBody<KIND, Type::Code::FLOAT32> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const float* a,
                  float* ew,
                  float* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<float>(m * m);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      int32_t info  = 0;
      float wkopt   = 0;
      int32_t lwork = -1;
      ssyev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             a_copy.ptr(0),
             &m,
             ew,
             &wkopt,
             &lwork,
             &info);
      lwork = (int)wkopt;

      std::vector<float> work_tmp(lwork);
      ssyev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             a_copy.ptr(0),
             &m,
             ew,
             work_tmp.data(),
             &lwork,
             &info);

      if (info != 0) {
        throw legate::TaskException(SyevTask::ERROR_MESSAGE);
      }

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        std::copy(a_copy.ptr(0), a_copy.ptr(0) + (m * m), ev);
        ev += batch_stride_ev;
      }
    }
  }
};

template <VariantKind KIND>
struct SyevImplBody<KIND, Type::Code::FLOAT64> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const double* a,
                  double* ew,
                  double* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<double>(m * m);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      int32_t info  = 0;
      double wkopt  = 0;
      int32_t lwork = -1;

      dsyev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             a_copy.ptr(0),
             &m,
             ew,
             &wkopt,
             &lwork,
             &info);
      lwork = (int)wkopt;

      std::vector<double> work_tmp(lwork);
      dsyev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             a_copy.ptr(0),
             &m,
             ew,
             work_tmp.data(),
             &lwork,
             &info);

      if (info != 0) {
        throw legate::TaskException(SyevTask::ERROR_MESSAGE);
      }

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        std::copy(a_copy.ptr(0), a_copy.ptr(0) + (m * m), ev);
        ev += batch_stride_ev;
      }
    }
  }
};

template <VariantKind KIND>
struct SyevImplBody<KIND, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const complex<float>* a,
                  float* ew,
                  complex<float>* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<complex<float>>(m * m);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      remove_diag_imag(a_copy.ptr(0), m);

      int32_t info            = 0;
      int32_t lwork           = -1;
      __complex__ float wkopt = 0;
      std::vector<float> rwork(3 * m - 2);

      cheev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             reinterpret_cast<__complex__ float*>(a_copy.ptr(0)),
             &m,
             ew,
             &wkopt,
             &lwork,
             rwork.data(),
             &info);

      lwork = __real__ wkopt;

      std::vector<__complex__ float> work_tmp(lwork);
      cheev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             reinterpret_cast<__complex__ float*>(a_copy.ptr(0)),
             &m,
             ew,
             work_tmp.data(),
             &lwork,
             rwork.data(),
             &info);

      if (info != 0) {
        throw legate::TaskException(SyevTask::ERROR_MESSAGE);
      }

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        std::copy(a_copy.ptr(0), a_copy.ptr(0) + (m * m), ev);
        ev += batch_stride_ev;
      }
    }
  }
};

template <VariantKind KIND>
struct SyevImplBody<KIND, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const complex<double>* a,
                  double* ew,
                  complex<double>* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<complex<double>>(m * m);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      remove_diag_imag(a_copy.ptr(0), m);

      int32_t info             = 0;
      int32_t lwork            = -1;
      __complex__ double wkopt = 0;
      std::vector<double> rwork(3 * m - 2);
      zheev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             reinterpret_cast<__complex__ double*>(a_copy.ptr(0)),
             &m,
             ew,
             &wkopt,
             &lwork,
             rwork.data(),
             &info);

      lwork = __real__ wkopt;

      std::vector<__complex__ double> work_tmp(lwork);
      zheev_(compute_evs ? "V" : "N",
             uplo_l ? "L" : "U",
             &m,
             reinterpret_cast<__complex__ double*>(a_copy.ptr(0)),
             &m,
             ew,
             work_tmp.data(),
             &lwork,
             rwork.data(),
             &info);

      if (info != 0) {
        throw legate::TaskException(SyevTask::ERROR_MESSAGE);
      }

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        std::copy(a_copy.ptr(0), a_copy.ptr(0) + (m * m), ev);
        ev += batch_stride_ev;
      }
    }
  }
};

}  // namespace cupynumeric
