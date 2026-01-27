/* Copyright 2025 NVIDIA Corporation
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

#include <cstdint>
#include <cstddef>

#if CUPYNUMERIC_BLAS_VENDOR_OPENBLAS
extern "C" {

void openblas_set_num_threads(int);

}  // extern "C"
#elif CUPYNUMERIC_BLAS_VENDOR_APPLE  // macos
#if __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>

#define CUPYNUMERIC_BLAS_VENDOR_APPLE_HAVE_THREAD_API 1
#else
#define CUPYNUMERIC_BLAS_VENDOR_APPLE_HAVE_THREAD_API 0
#endif
#endif

namespace cupynumeric {

#if CUPYNUMERIC_BLAS_VENDOR_OPENBLAS

inline void blas_set_num_threads(std::int32_t threads) { openblas_set_num_threads(threads); }

#elif CUPYNUMERIC_BLAS_VENDOR_APPLE

inline void blas_set_num_threads(std::int32_t threads [[maybe_unused]])
{
#if CUPYNUMERIC_BLAS_VENDOR_APPLE_HAVE_THREAD_API
  // Choices are:
  //
  // - SINGLE_THREADED: use only 1 thread
  // - MULTI_THREADED: Mr. Tim Apple decides how many threads he shall grant you.
  //
  // So, not much to do here
  static_cast<void>(::BLASSetThreading(threads <= 1
                                         ? ::BLAS_THREADING::BLAS_THREADING_SINGLE_THREADED
                                         : ::BLAS_THREADING::BLAS_THREADING_MULTI_THREADED));
#endif
}

#else

inline void blas_set_num_threads(std::int32_t) {}

#endif

}  // namespace cupynumeric

// NOLINTBEGIN
extern "C" {

enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };

void cblas_sgemm(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB,
                 const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const int lda,
                 const float* B,
                 const int ldb,
                 const float beta,
                 float* C,
                 const int ldc);
void cblas_dgemm(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB,
                 const int M,
                 const int N,
                 const int K,
                 const double alpha,
                 const double* A,
                 const int lda,
                 const double* B,
                 const int ldb,
                 const double beta,
                 double* C,
                 const int ldc);
void cblas_cgemm(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB,
                 const int M,
                 const int N,
                 const int K,
                 const void* alpha,
                 const void* A,
                 const int lda,
                 const void* B,
                 const int ldb,
                 const void* beta,
                 void* C,
                 const int ldc);
void cblas_zgemm(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB,
                 const int M,
                 const int N,
                 const int K,
                 const void* alpha,
                 const void* A,
                 const int lda,
                 const void* B,
                 const int ldb,
                 const void* beta,
                 void* C,
                 const int ldc);

void cblas_strsm(CBLAS_LAYOUT layout,
                 CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag,
                 const int M,
                 const int N,
                 const float alpha,
                 const float* A,
                 const int lda,
                 float* B,
                 const int ldb);
void cblas_dtrsm(CBLAS_LAYOUT layout,
                 CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag,
                 const int M,
                 const int N,
                 const double alpha,
                 const double* A,
                 const int lda,
                 double* B,
                 const int ldb);
void cblas_ctrsm(CBLAS_LAYOUT layout,
                 CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag,
                 const int M,
                 const int N,
                 const void* alpha,
                 const void* A,
                 const int lda,
                 void* B,
                 const int ldb);
void cblas_ztrsm(CBLAS_LAYOUT layout,
                 CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag,
                 const int M,
                 const int N,
                 const void* alpha,
                 const void* A,
                 const int lda,
                 void* B,
                 const int ldb);

void cblas_ssyrk(CBLAS_LAYOUT layout,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const int lda,
                 const float beta,
                 float* C,
                 const int ldc);
void cblas_dsyrk(CBLAS_LAYOUT layout,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans,
                 const int N,
                 const int K,
                 const double alpha,
                 const double* A,
                 const int lda,
                 const double beta,
                 double* C,
                 const int ldc);
void cblas_cherk(CBLAS_LAYOUT layout,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans,
                 const int N,
                 const int K,
                 const float alpha,
                 const void* A,
                 const int lda,
                 const float beta,
                 void* C,
                 const int ldc);
void cblas_zherk(CBLAS_LAYOUT layout,
                 CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans,
                 const int N,
                 const int K,
                 const double alpha,
                 const void* A,
                 const int lda,
                 const double beta,
                 void* C,
                 const int ldc);

void cblas_sgemv(const CBLAS_LAYOUT layout,
                 const CBLAS_TRANSPOSE TransA,
                 const int M,
                 const int N,
                 const float alpha,
                 const float* A,
                 const int lda,
                 const float* X,
                 const int incX,
                 const float beta,
                 float* Y,
                 const int incY);
void cblas_dgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 const int M,
                 const int N,
                 const double alpha,
                 const double* A,
                 const int lda,
                 const double* X,
                 const int incX,
                 const double beta,
                 double* Y,
                 const int incY);
void cblas_cgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 const int M,
                 const int N,
                 const void* alpha,
                 const void* A,
                 const int lda,
                 const void* X,
                 const int incX,
                 const void* beta,
                 void* Y,
                 const int incY);
void cblas_zgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA,
                 const int M,
                 const int N,
                 const void* alpha,
                 const void* A,
                 const int lda,
                 const void* X,
                 const int incX,
                 const void* beta,
                 void* Y,
                 const int incY);

int sgesv_(int32_t* n,
           int32_t* nrhs,
           float* a,
           int32_t* lda,
           int32_t* ipiv,
           float* b,
           int32_t* ldb,
           int32_t* info);
int dgesv_(int32_t* n,
           int32_t* nrhs,
           double* a,
           int32_t* lda,
           int32_t* ipiv,
           double* b,
           int32_t* ldb,
           int32_t* info);
int cgesv_(int32_t* n,
           int32_t* nrhs,
           _Complex float* a,
           int32_t* lda,
           int32_t* ipiv,
           _Complex float* b,
           int32_t* ldb,
           int32_t* info);
int zgesv_(int32_t* n,
           int32_t* nrhs,
           _Complex double* a,
           int32_t* lda,
           int32_t* ipiv,
           _Complex double* b,
           int32_t* ldb,
           int32_t* info);

int sgesvd_(const char* jobu,
            const char* jobvt,
            int32_t* m,
            int32_t* n,
            float* a,
            int32_t* lda,
            float* s,
            float* u,
            int32_t* ldu,
            float* vt,
            int32_t* ldvt,
            float* work,
            int32_t* lwork,
            int32_t* info);
int dgesvd_(const char* jobu,
            const char* jobvt,
            int32_t* m,
            int32_t* n,
            double* a,
            int32_t* lda,
            double* s,
            double* u,
            int32_t* ldu,
            double* vt,
            int32_t* ldvt,
            double* work,
            int32_t* lwork,
            int32_t* info);
int cgesvd_(const char* jobu,
            const char* jobvt,
            int32_t* m,
            int32_t* n,
            _Complex float* a,
            int32_t* lda,
            float* s,
            _Complex float* u,
            int32_t* ldu,
            _Complex float* vt,
            int32_t* ldvt,
            _Complex float* work,
            int32_t* lwork,
            float* rwork,
            int32_t* info);
int zgesvd_(const char* jobu,
            const char* jobvt,
            int32_t* m,
            int32_t* n,
            _Complex double* a,
            int32_t* lda,
            double* s,
            _Complex double* u,
            int32_t* ldu,
            _Complex double* vt,
            int32_t* ldvt,
            _Complex double* work,
            int32_t* lwork,
            double* rwork,
            int32_t* info);

int ssyev_(const char* jobz,
           const char* uplo,
           int32_t* n,
           float* a,
           int32_t* lda,
           float* w,
           float* work,
           int32_t* lwork,
           int32_t* info);
int dsyev_(const char* jobz,
           const char* uplo,
           int32_t* n,
           double* a,
           int32_t* lda,
           double* w,
           double* work,
           int32_t* lwork,
           int32_t* info);

int cheev_(const char* jobz,
           const char* uplo,
           int32_t* n,
           _Complex float* a,
           int32_t* lda,
           float* w,
           _Complex float* work,
           int32_t* lwork,
           float* rwork,
           int32_t* info);
int zheev_(const char* jobz,
           const char* uplo,
           int32_t* n,
           _Complex double* a,
           int32_t* lda,
           double* w,
           _Complex double* work,
           int32_t* lwork,
           double* rwork,
           int32_t* info);

int spotrf_(const char* uplo, int32_t* n, float* a, int32_t* lda, int32_t* info);
int dpotrf_(const char* uplo, int32_t* n, double* a, int32_t* lda, int32_t* info);
int cpotrf_(const char* uplo, int32_t* n, _Complex float* a, int32_t* lda, int32_t* info);
int zpotrf_(const char* uplo, int32_t* n, _Complex double* a, int32_t* lda, int32_t* info);

int spotrs_(const char* uplo,
            int32_t* n,
            int32_t* nrhs,
            float* a,
            int32_t* lda,
            float* b,
            int32_t* ldb,
            int32_t* info);
int dpotrs_(const char* uplo,
            int32_t* n,
            int32_t* nrhs,
            double* a,
            int32_t* lda,
            double* b,
            int32_t* ldb,
            int32_t* info);
int cpotrs_(const char* uplo,
            int32_t* n,
            int32_t* nrhs,
            _Complex float* a,
            int32_t* lda,
            _Complex float* b,
            int32_t* ldb,
            int32_t* info);
int zpotrs_(const char* uplo,
            int32_t* n,
            int32_t* nrhs,
            _Complex double* a,
            int32_t* lda,
            _Complex double* b,
            int32_t* ldb,
            int32_t* info);

int sgeev_(const char* jobvl,
           const char* jobvr,
           int32_t* n,
           float* a,
           int32_t* lda,
           float* wr,
           float* wi,
           float* vl,
           int32_t* ldvl,
           float* vr,
           int32_t* ldvr,
           float* work,
           int32_t* lwork,
           int32_t* info);
int dgeev_(const char* jobvl,
           const char* jobvr,
           int32_t* n,
           double* a,
           int32_t* lda,
           double* wr,
           double* wi,
           double* vl,
           int32_t* ldvl,
           double* vr,
           int32_t* ldvr,
           double* work,
           int32_t* lwork,
           int32_t* info);
int cgeev_(const char* jobvl,
           const char* jobvr,
           int32_t* n,
           _Complex float* a,
           int32_t* lda,
           _Complex float* w,
           _Complex float* vl,
           int32_t* ldvl,
           _Complex float* vr,
           int32_t* ldvr,
           _Complex float* work,
           int32_t* lwork,
           float* rwork,
           int32_t* info);
int zgeev_(const char* jobvl,
           const char* jobvr,
           int32_t* n,
           _Complex double* a,
           int32_t* lda,
           _Complex double* w,
           _Complex double* vl,
           int32_t* ldvl,
           _Complex double* vr,
           int32_t* ldvr,
           _Complex double* work,
           int32_t* lwork,
           double* rwork,
           int32_t* info);

int sgeqrf_(int32_t* m,
            int32_t* n,
            float* a,
            int32_t* lda,
            float* tau,
            float* work,
            int32_t* lwork,
            int32_t* info);
int sorgqr_(int32_t* m,
            int32_t* n,
            int32_t* k,
            float* a,
            int32_t* lda,
            float* tau,
            float* work,
            int32_t* lwork,
            int32_t* info);
int dgeqrf_(int32_t* m,
            int32_t* n,
            double* a,
            int32_t* lda,
            double* tau,
            double* work,
            int32_t* lwork,
            int32_t* info);
int dorgqr_(int32_t* m,
            int32_t* n,
            int32_t* k,
            double* a,
            int32_t* lda,
            double* tau,
            double* work,
            int32_t* lwork,
            int32_t* info);
int cgeqrf_(int32_t* m,
            int32_t* n,
            _Complex float* a,
            int32_t* lda,
            _Complex float* tau,
            _Complex float* work,
            int32_t* lwork,
            int32_t* info);
int cungqr_(int32_t* m,
            int32_t* n,
            int32_t* k,
            _Complex float* a,
            int32_t* lda,
            _Complex float* tau,
            _Complex float* work,
            int32_t* lwork,
            int32_t* info);
int zgeqrf_(int32_t* m,
            int32_t* n,
            _Complex double* a,
            int32_t* lda,
            _Complex double* tau,
            _Complex double* work,
            int32_t* lwork,
            int32_t* info);
int zungqr_(int32_t* m,
            int32_t* n,
            int32_t* k,
            _Complex double* a,
            int32_t* lda,
            _Complex double* tau,
            _Complex double* work,
            int32_t* lwork,
            int32_t* info);

int sgemm_(const char* transa,
           const char* transb,
           int32_t* m,
           int32_t* n,
           int32_t* k,
           float* alpha,
           float* a,
           int32_t* lda,
           float* b,
           int32_t* ldb,
           float* beta,
           float* c__,
           int32_t* ldc);
int dgemm_(const char* transa,
           const char* transb,
           int32_t* m,
           int32_t* n,
           int32_t* k,
           double* alpha,
           double* a,
           int32_t* lda,
           double* b,
           int32_t* ldb,
           double* beta,
           double* c__,
           int32_t* ldc);
int cgemm_(const char* transa,
           const char* transb,
           int32_t* m,
           int32_t* n,
           int32_t* k,
           _Complex float* alpha,
           _Complex float* a,
           int32_t* lda,
           _Complex float* b,
           int32_t* ldb,
           _Complex float* beta,
           _Complex float* c__,
           int32_t* ldc);
int zgemm_(const char* transa,
           const char* transb,
           int32_t* m,
           int32_t* n,
           int32_t* k,
           _Complex double* alpha,
           _Complex double* a,
           int32_t* lda,
           _Complex double* b,
           int32_t* ldb,
           _Complex double* beta,
           _Complex double* c__,
           int32_t* ldc);

int ssyrk_(const char* uplo,
           const char* trans,
           int32_t* n,
           int32_t* k,
           float* alpha,
           float* a,
           int32_t* lda,
           float* beta,
           float* c__,
           int32_t* ldc);
int dsyrk_(const char* uplo,
           const char* trans,
           int32_t* n,
           int32_t* k,
           double* alpha,
           double* a,
           int32_t* lda,
           double* beta,
           double* c__,
           int32_t* ldc);
int cherk_(const char* uplo,
           const char* trans,
           int32_t* n,
           int32_t* k,
           float* alpha,
           _Complex float* a,
           int32_t* lda,
           float* beta,
           _Complex float* c__,
           int32_t* ldc);
int zherk_(const char* uplo,
           const char* trans,
           int32_t* n,
           int32_t* k,
           double* alpha,
           _Complex double* a,
           int32_t* lda,
           double* beta,
           _Complex double* c__,
           int32_t* ldc);

int strsm_(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           int32_t* m,
           int32_t* n,
           float* alpha,
           float* a,
           int32_t* lda,
           float* b,
           int32_t* ldb);
int dtrsm_(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           int32_t* m,
           int32_t* n,
           double* alpha,
           double* a,
           int32_t* lda,
           double* b,
           int32_t* ldb);
int ctrsm_(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           int32_t* m,
           int32_t* n,
           _Complex float* alpha,
           _Complex float* a,
           int32_t* lda,
           _Complex float* b,
           int32_t* ldb);
int ztrsm_(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           int32_t* m,
           int32_t* n,
           _Complex double* alpha,
           _Complex double* a,
           int32_t* lda,
           _Complex double* b,
           int32_t* ldb);

int sgemv_(const char* trans,
           int32_t* m,
           int32_t* n,
           float* alpha,
           float* a,
           int32_t* lda,
           float* x,
           int32_t* incx,
           float* beta,
           float* y,
           int32_t* incy);
int dgemv_(const char* trans,
           int32_t* m,
           int32_t* n,
           double* alpha,
           double* a,
           int32_t* lda,
           double* x,
           int32_t* incx,
           double* beta,
           double* y,
           int32_t* incy);
int cgemv_(const char* trans,
           int32_t* m,
           int32_t* n,
           _Complex float* alpha,
           _Complex float* a,
           int32_t* lda,
           _Complex float* x,
           int32_t* incx,
           _Complex float* beta,
           _Complex float* y,
           int32_t* incy);
int zgemv_(const char* trans,
           int32_t* m,
           int32_t* n,
           _Complex double* alpha,
           _Complex double* a,
           int32_t* lda,
           _Complex double* x,
           int32_t* incx,
           _Complex double* beta,
           _Complex double* y,
           int32_t* incy);

}  // extern "C"
// NOLINTEND
