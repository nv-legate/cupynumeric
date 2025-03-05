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
#include <cstring>

namespace cupynumeric {

using namespace legate;

namespace {

template <typename T>
void assemble_complex(complex<T>* ew, complex<T>* ev, T* ew_r, T* ew_i, T* ev_r, size_t m)
{
  bool skip_next_ev = false;
  for (int i = 0; i < m; ++i) {
    ew[i] = complex<T>(ew_r[i], ew_i[i]);
    if (ev != nullptr) {
      if (skip_next_ev) {
        skip_next_ev = false;
      } else {
        T* src1          = &ev_r[i * m];
        complex<T>* dst1 = &ev[i * m];
        if (ew_i[i] != T(0)) {
          // define next 2 EVs
          T* src2          = src1 + m;
          complex<T>* dst2 = dst1 + m;
          for (int k = 0; k < m; ++k) {
            dst1[k] = complex<T>(src1[k], src2[k]);
            dst2[k] = complex<T>(src1[k], T(-1) * src2[k]);
          }
          skip_next_ev = true;
        } else {
          for (int k = 0; k < m; ++k) {
            dst1[k] = complex<T>(src1[k], T(0));
          }
        }
      }
    }
  }
}
}  // namespace

template <VariantKind KIND>
struct GeevImplBody<KIND, Type::Code::FLOAT32> {
  void operator()(int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const float* a,
                  complex<float>* ew,
                  complex<float>* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<float>(m * m);

    // for real input --> create real buffer and assemble afterwards
    auto ev_tmp       = create_buffer<float>(m * m);
    float* ev_tmp_prt = ev_tmp.ptr(0);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      std::vector<float> ew_r(m);
      std::vector<float> ew_i(m);

      int32_t info  = 0;
      float wkopt   = 0;
      int32_t lwork = -1;
      LAPACK_sgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   a_copy.ptr(0),
                   &m,
                   ew_r.data(),
                   ew_i.data(),
                   nullptr,
                   &m,
                   ev_tmp_prt,
                   &m,
                   &wkopt,
                   &lwork,
                   &info);
      lwork = (int)wkopt;

      std::vector<float> work_tmp(lwork);
      LAPACK_sgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   a_copy.ptr(0),
                   &m,
                   ew_r.data(),
                   ew_i.data(),
                   nullptr,
                   &m,
                   ev_tmp_prt,
                   &m,
                   work_tmp.data(),
                   &lwork,
                   &info);

      if (info != 0) {
        throw legate::TaskException(GeevTask::ERROR_MESSAGE);
      }

      assemble_complex<float>(ew, ev, ew_r.data(), ew_i.data(), ev_tmp_prt, m);

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        ev += batch_stride_ev;
      }
    }
  }
};

template <VariantKind KIND>
struct GeevImplBody<KIND, Type::Code::FLOAT64> {
  void operator()(int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const double* a,
                  complex<double>* ew,
                  complex<double>* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<double>(m * m);

    // for real input --> create real buffer and assemble afterwards
    auto ev_tmp        = create_buffer<double>(m * m);
    double* ev_tmp_prt = ev_tmp.ptr(0);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      std::vector<double> ew_r(m);
      std::vector<double> ew_i(m);

      int32_t info  = 0;
      double wkopt  = 0;
      int32_t lwork = -1;

      LAPACK_dgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   a_copy.ptr(0),
                   &m,
                   ew_r.data(),
                   ew_i.data(),
                   nullptr,
                   &m,
                   ev_tmp_prt,
                   &m,
                   &wkopt,
                   &lwork,
                   &info);
      lwork = (int)wkopt;

      std::vector<double> work_tmp(lwork);
      LAPACK_dgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   a_copy.ptr(0),
                   &m,
                   ew_r.data(),
                   ew_i.data(),
                   nullptr,
                   &m,
                   ev_tmp_prt,
                   &m,
                   work_tmp.data(),
                   &lwork,
                   &info);

      if (info != 0) {
        throw legate::TaskException(GeevTask::ERROR_MESSAGE);
      }

      assemble_complex<double>(ew, ev, ew_r.data(), ew_i.data(), ev_tmp_prt, m);

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        ev += batch_stride_ev;
      }
    }
  }
};

template <VariantKind KIND>
struct GeevImplBody<KIND, Type::Code::COMPLEX64> {
  void operator()(int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const complex<float>* a,
                  complex<float>* ew,
                  complex<float>* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<complex<float>>(m * m);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      int32_t info            = 0;
      int32_t lwork           = -1;
      __complex__ float wkopt = 0;
      std::vector<float> rwork(2 * m);

      LAPACK_cgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   reinterpret_cast<__complex__ float*>(a_copy.ptr(0)),
                   &m,
                   reinterpret_cast<__complex__ float*>(ew),
                   nullptr,
                   &m,
                   reinterpret_cast<__complex__ float*>(ev),
                   &m,
                   &wkopt,
                   &lwork,
                   rwork.data(),
                   &info);

      lwork = __real__ wkopt;

      std::vector<__complex__ float> work_tmp(lwork);
      LAPACK_cgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   reinterpret_cast<__complex__ float*>(a_copy.ptr(0)),
                   &m,
                   reinterpret_cast<__complex__ float*>(ew),
                   nullptr,
                   &m,
                   reinterpret_cast<__complex__ float*>(ev),
                   &m,
                   work_tmp.data(),
                   &lwork,
                   rwork.data(),
                   &info);

      if (info != 0) {
        throw legate::TaskException(GeevTask::ERROR_MESSAGE);
      }

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        ev += batch_stride_ev;
      }
    }
  }
};

template <VariantKind KIND>
struct GeevImplBody<KIND, Type::Code::COMPLEX128> {
  void operator()(int32_t m,
                  int32_t num_batches,
                  int32_t batch_stride_ew,
                  int32_t batch_stride_ev,
                  const complex<double>* a,
                  complex<double>* ew,
                  complex<double>* ev)
  {
    bool compute_evs = ev != nullptr;
    auto a_copy      = create_buffer<complex<double>>(m * m);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      std::copy(a, a + (m * m), a_copy.ptr(0));

      int32_t info             = 0;
      int32_t lwork            = -1;
      __complex__ double wkopt = 0;
      std::vector<double> rwork(2 * m);
      LAPACK_zgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   reinterpret_cast<__complex__ double*>(a_copy.ptr(0)),
                   &m,
                   reinterpret_cast<__complex__ double*>(ew),
                   nullptr,
                   &m,
                   reinterpret_cast<__complex__ double*>(ev),
                   &m,
                   &wkopt,
                   &lwork,
                   rwork.data(),
                   &info);

      lwork = __real__ wkopt;

      std::vector<__complex__ double> work_tmp(lwork);
      LAPACK_zgeev("N",
                   compute_evs ? "V" : "N",
                   &m,
                   reinterpret_cast<__complex__ double*>(a_copy.ptr(0)),
                   &m,
                   reinterpret_cast<__complex__ double*>(ew),
                   nullptr,
                   &m,
                   reinterpret_cast<__complex__ double*>(ev),
                   &m,
                   work_tmp.data(),
                   &lwork,
                   rwork.data(),
                   &info);

      if (info != 0) {
        throw legate::TaskException(GeevTask::ERROR_MESSAGE);
      }

      a += batch_stride_ev;
      ew += batch_stride_ew;
      if (compute_evs) {
        ev += batch_stride_ev;
      }
    }
  }
};

}  // namespace cupynumeric