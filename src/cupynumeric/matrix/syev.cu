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

#include "cupynumeric/matrix/syev.h"
#include "cupynumeric/matrix/syev_template.inl"
#include "cupynumeric/utilities/thrust_util.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "cupynumeric/cuda_help.h"
#include <vector>

namespace cupynumeric {

using namespace legate;

template <typename VAL_COMPLEX>
struct removeDiagImag {
  VAL_COMPLEX* a_inout_;
  const int64_t m_;

  removeDiagImag(VAL_COMPLEX* a_inout, int64_t m) : a_inout_(a_inout), m_(m) {}

  __CUDA_HD__ void operator()(const int64_t& idx) const
  {
    int64_t idx_diag     = idx * m_ + idx % m_;
    a_inout_[idx_diag].y = 0.0;
  }
};

template <typename VAL_COMPLEX>
void remove_diag_imag(VAL_COMPLEX* a_inout, int64_t m, cudaStream_t stream, int64_t num_batches = 1)
{
  thrust::for_each(DEFAULT_POLICY.on(stream),
                   thrust::make_counting_iterator<int64_t>(0),
                   thrust::make_counting_iterator<int64_t>(m * num_batches),
                   removeDiagImag<VAL_COMPLEX>(a_inout, m));
}

template <typename VAL, typename DataType>
static inline void syev_batched_template(DataType valTypeR,
                                         DataType valTypeA,
                                         bool uplo_l,
                                         int64_t m,
                                         const void* a,
                                         void* ew,
                                         void* ev,
                                         int64_t num_batches,
                                         cudaStream_t stream)
{
  auto handle       = get_cusolver();
  auto syev_handles = get_cusolver_extra_symbols();

  assert(syev_handles->has_syev_batched);

  bool compute_evs = ev != nullptr;

  auto a_copy = create_buffer<VAL>(compute_evs ? 0 : num_batches * m * m, Memory::Kind::GPU_FB_MEM);
  void* a_copy_ptr = compute_evs ? ev : reinterpret_cast<void*>(a_copy.ptr(0));

  CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
    a_copy_ptr, a, num_batches * m * m * sizeof(VAL), cudaMemcpyDeviceToDevice, stream));

  if constexpr (std::is_same_v<VAL, complex<float>>) {
    remove_diag_imag(reinterpret_cast<cuComplex*>(a_copy_ptr), m, stream, num_batches);
  } else if constexpr (std::is_same_v<VAL, complex<double>>) {
    remove_diag_imag(reinterpret_cast<cuDoubleComplex*>(a_copy_ptr), m, stream, num_batches);
  }

  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  size_t lwork_device, lwork_host;
  CHECK_CUSOLVER(syev_handles->cusolver_syev_batched_bufferSize(
    handle,
    nullptr,
    compute_evs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
    uplo_l ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
    m,
    valTypeA,
    a_copy_ptr,
    m,
    valTypeR,
    ew,
    valTypeA,
    &lwork_device,
    &lwork_host,
    num_batches));

  auto buffer = create_buffer<char>(lwork_device, Memory::Kind::GPU_FB_MEM);
  std::vector<char> buffer_host(std::max(1ul, lwork_host));
  auto info = create_buffer<int32_t>(num_batches, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(syev_handles->cusolver_syev_batched(
    handle,
    nullptr,
    compute_evs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
    uplo_l ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
    m,
    valTypeA,
    a_copy_ptr,
    m,
    valTypeR,
    ew,
    valTypeA,
    buffer.ptr(0),
    lwork_device,
    buffer_host.data(),
    lwork_host,
    info.ptr(0),
    num_batches));

  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (int i = 0; i < num_batches; i++) {
    if (info[i] != 0) {
      throw legate::TaskException(SyevTask::ERROR_MESSAGE);
    }
  }
}

template <typename VAL, typename DataType>
static inline void syevd_template(DataType valTypeR,
                                  DataType valTypeA,
                                  bool uplo_l,
                                  int64_t m,
                                  const void* a,
                                  void* ew,
                                  void* ev,
                                  cudaStream_t stream)
{
  auto handle = get_cusolver();

  bool compute_evs = ev != nullptr;

  auto a_copy      = create_buffer<VAL>(compute_evs ? 0 : m * m, Memory::Kind::GPU_FB_MEM);
  void* a_copy_ptr = compute_evs ? ev : reinterpret_cast<void*>(a_copy.ptr(0));

  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(a_copy_ptr, a, m * m * sizeof(VAL), cudaMemcpyDeviceToDevice, stream));

  if constexpr (std::is_same_v<VAL, complex<float>>) {
    remove_diag_imag(reinterpret_cast<cuComplex*>(a_copy_ptr), m, stream);
  } else if constexpr (std::is_same_v<VAL, complex<double>>) {
    remove_diag_imag(reinterpret_cast<cuDoubleComplex*>(a_copy_ptr), m, stream);
  }

  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  size_t lwork_device, lwork_host;
  CHECK_CUSOLVER(
    cusolverDnXsyevd_bufferSize(handle,
                                nullptr,
                                compute_evs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
                                uplo_l ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                m,
                                valTypeA,
                                a_copy_ptr,
                                m,
                                valTypeR,
                                ew,
                                valTypeA,
                                &lwork_device,
                                &lwork_host));

  auto buffer = create_buffer<char>(lwork_device, Memory::Kind::GPU_FB_MEM);
  std::vector<char> buffer_host(std::max(1ul, lwork_host));
  auto info = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(
    cusolverDnXsyevd(handle,
                     nullptr,
                     compute_evs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
                     uplo_l ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                     m,
                     valTypeA,
                     a_copy_ptr,
                     m,
                     valTypeR,
                     ew,
                     valTypeA,
                     buffer.ptr(0),
                     lwork_device,
                     buffer_host.data(),
                     lwork_host,
                     info.ptr(0)));

  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] != 0) {
    throw legate::TaskException(SyevTask::ERROR_MESSAGE);
  }
}

template <>
struct SyevImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const float* a,
                  float* ew,
                  float* ev)
  {
    auto stream      = context.get_task_stream();
    bool compute_evs = ev != nullptr;

    if (num_batches > 1 && get_cusolver_extra_symbols()->has_syev_batched) {
      syev_batched_template<float>(
        CUDA_R_32F, CUDA_R_32F, uplo_l, m, a, ew, compute_evs ? ev : nullptr, num_batches, stream);
    } else {
      for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        syevd_template<float>(CUDA_R_32F,
                              CUDA_R_32F,
                              uplo_l,
                              m,
                              a + batch_idx * batch_stride_ev,
                              ew + batch_idx * batch_stride_ew,
                              compute_evs ? (ev + batch_idx * batch_stride_ev) : nullptr,
                              stream);
      }
    }
  }
};

template <>
struct SyevImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const double* a,
                  double* ew,
                  double* ev)
  {
    auto stream      = context.get_task_stream();
    bool compute_evs = ev != nullptr;

    if (num_batches > 1 && get_cusolver_extra_symbols()->has_syev_batched) {
      syev_batched_template<double>(
        CUDA_R_64F, CUDA_R_64F, uplo_l, m, a, ew, compute_evs ? ev : nullptr, num_batches, stream);
    } else {
      for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        syevd_template<double>(CUDA_R_64F,
                               CUDA_R_64F,
                               uplo_l,
                               m,
                               a + batch_idx * batch_stride_ev,
                               ew + batch_idx * batch_stride_ew,
                               compute_evs ? (ev + batch_idx * batch_stride_ev) : nullptr,
                               stream);
      }
    }
  }
};

template <>
struct SyevImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const complex<float>* a,
                  float* ew,
                  complex<float>* ev)
  {
    auto stream      = context.get_task_stream();
    bool compute_evs = ev != nullptr;

    if (num_batches > 1 && get_cusolver_extra_symbols()->has_syev_batched) {
      syev_batched_template<complex<float>>(
        CUDA_R_32F,
        CUDA_C_32F,
        uplo_l,
        m,
        reinterpret_cast<const cuComplex*>(a),
        ew,
        compute_evs ? reinterpret_cast<cuComplex*>(ev) : nullptr,
        num_batches,
        stream);
    } else {
      for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        syevd_template<complex<float>>(
          CUDA_R_32F,
          CUDA_C_32F,
          uplo_l,
          m,
          reinterpret_cast<const cuComplex*>(a + batch_idx * batch_stride_ev),
          ew + batch_idx * batch_stride_ew,
          compute_evs ? reinterpret_cast<cuComplex*>(ev + batch_idx * batch_stride_ev) : nullptr,
          stream);
      }
    }
  }
};

template <>
struct SyevImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit SyevImplBody(TaskContext context) : context(context) {}

  void operator()(bool uplo_l,
                  int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const complex<double>* a,
                  double* ew,
                  complex<double>* ev)
  {
    auto stream      = context.get_task_stream();
    bool compute_evs = ev != nullptr;

    if (num_batches > 1 && get_cusolver_extra_symbols()->has_syev_batched) {
      syev_batched_template<complex<double>>(
        CUDA_R_64F,
        CUDA_C_64F,
        uplo_l,
        m,
        reinterpret_cast<const cuDoubleComplex*>(a),
        ew,
        compute_evs ? reinterpret_cast<cuDoubleComplex*>(ev) : nullptr,
        num_batches,
        stream);
    } else {
      for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        syevd_template<complex<double>>(
          CUDA_R_64F,
          CUDA_C_64F,
          uplo_l,
          m,
          reinterpret_cast<const cuDoubleComplex*>(a + batch_idx * batch_stride_ev),
          ew + batch_idx * batch_stride_ew,
          compute_evs ? reinterpret_cast<cuDoubleComplex*>(ev + batch_idx * batch_stride_ev)
                      : nullptr,
          stream);
      }
    }
  }
};

/*static*/ void SyevTask::gpu_variant(TaskContext context)
{
  syev_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
