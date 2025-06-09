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

#include "cupynumeric/matrix/geev.h"
#include "cupynumeric/matrix/geev_template.inl"
#include "cupynumeric/utilities/thrust_util.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "cupynumeric/cuda_help.h"
#include <vector>

namespace cupynumeric {

using namespace legate;

template <typename VAL, typename VAL_COMPLEX>
struct assembleEvs : public thrust::unary_function<VAL_COMPLEX, int64_t> {
  const VAL_COMPLEX* ew_in_;
  const VAL* ev_in_;
  const int64_t m_;

  assembleEvs(VAL_COMPLEX* ew_in, VAL* ev_in, int64_t m) : ew_in_(ew_in), ev_in_(ev_in), m_(m) {}

  __CUDA_HD__ VAL_COMPLEX operator()(const int64_t& idx) const
  {
    int64_t col_idx = idx / m_;
    auto ew_i       = ew_in_[col_idx].y;
    // if img == 0 -> ev = ev[idx]
    // if img positive -> ev = ev[idx] + i*ev[idx+1]
    // if img negative -> ev = ev[idx-1] - i*ev[idx]
    const int64_t real_idx = idx - ((ew_i < 0) ? m_ : 0);
    const int64_t img_idx  = idx + ((ew_i > 0) ? m_ : 0);
    VAL factor             = ((ew_i > 0) ? VAL(1.0) : ((ew_i < 0) ? VAL(-1.0) : VAL(0.0)));
    VAL_COMPLEX result;
    result.x = ev_in_[real_idx];
    result.y = factor * ev_in_[img_idx];
    return result;
  }
};

template <typename VAL, typename VAL_COMPLEX>
void assemble_complex_evs(VAL_COMPLEX* ev_out, VAL_COMPLEX* ew_in, VAL* ev_in, int64_t m)
{
  auto stream = get_cached_stream();
  thrust::transform(DEFAULT_POLICY.on(stream),
                    thrust::make_counting_iterator<int64_t>(0),
                    thrust::make_counting_iterator<int64_t>(m * m),
                    ev_out,
                    assembleEvs(ew_in, ev_in, m));
}

template <typename VAL, typename DataType>
static inline void geev_template(
  DataType valTypeC, DataType valTypeA, int64_t m, const void* a, void* ew, void* ev)
{
  auto handle       = get_cusolver();
  auto stream       = get_cached_stream();
  auto geev_handles = get_cusolver_extra_symbols();

  bool compute_evs = ev != nullptr;

  auto a_copy = create_buffer<VAL>(m * m, Memory::Kind::GPU_FB_MEM);

  CUPYNUMERIC_CHECK_CUDA(
    cudaMemcpyAsync(a_copy.ptr(0), a, m * m * sizeof(VAL), cudaMemcpyDeviceToDevice, stream));

  CHECK_CUSOLVER(cusolverDnSetStream(handle, stream));

  size_t lwork_device, lwork_host;
  CHECK_CUSOLVER(geev_handles->cusolver_geev_bufferSize(
    handle,
    nullptr,
    CUSOLVER_EIG_MODE_NOVECTOR,
    compute_evs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
    m,
    valTypeA,
    reinterpret_cast<void*>(a_copy.ptr(0)),
    m,
    valTypeC,
    ew,
    valTypeA,
    nullptr,  // left EVs
    m,
    valTypeA,
    ev,
    m,
    valTypeA,
    &lwork_device,
    &lwork_host));

  auto buffer = create_buffer<char>(lwork_device, Memory::Kind::GPU_FB_MEM);
  std::vector<char> buffer_host(std::max(1ul, lwork_host));
  auto info = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(
    geev_handles->cusolver_geev(handle,
                                nullptr,
                                CUSOLVER_EIG_MODE_NOVECTOR,
                                compute_evs ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR,
                                m,
                                valTypeA,
                                reinterpret_cast<void*>(a_copy.ptr(0)),
                                m,
                                valTypeC,
                                ew,
                                valTypeA,
                                nullptr,  // left EVs
                                m,
                                valTypeA,
                                ev,
                                m,
                                valTypeA,
                                buffer.ptr(0),
                                lwork_device,
                                buffer_host.data(),
                                lwork_host,
                                info.ptr(0)));

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  if (info[0] != 0) {
    throw legate::TaskException(GeevTask::ERROR_MESSAGE);
  }
}

template <>
struct GeevImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  void operator()(int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const float* a,
                  complex<float>* ew,
                  complex<float>* ev)
  {
    bool compute_evs = ev != nullptr;

    // for real input --> create real buffer and assemble afterwards
    auto ev_tmp = create_buffer<float>(compute_evs ? m * m : 0, Memory::Kind::GPU_FB_MEM);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      geev_template<float>(CUDA_C_32F,
                           CUDA_R_32F,
                           m,
                           a + batch_idx * batch_stride_ev,
                           reinterpret_cast<cuComplex*>(ew + batch_idx * batch_stride_ew),
                           compute_evs ? reinterpret_cast<void*>(ev_tmp.ptr(0)) : nullptr);

      if (compute_evs) {
        assemble_complex_evs(reinterpret_cast<cuComplex*>(ev + batch_idx * batch_stride_ev),
                             reinterpret_cast<cuComplex*>(ew + batch_idx * batch_stride_ew),
                             ev_tmp.ptr(0),
                             m);
      }
    }
  }
};

template <>
struct GeevImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  void operator()(int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const double* a,
                  complex<double>* ew,
                  complex<double>* ev)
  {
    bool compute_evs = ev != nullptr;

    // for real input --> create real buffer and assemble afterwards
    auto ev_tmp = create_buffer<double>(compute_evs ? m * m : 0, Memory::Kind::GPU_FB_MEM);

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      geev_template<double>(CUDA_C_64F,
                            CUDA_R_64F,
                            m,
                            a + batch_idx * batch_stride_ev,
                            reinterpret_cast<cuDoubleComplex*>(ew + batch_idx * batch_stride_ew),
                            compute_evs ? reinterpret_cast<void*>(ev_tmp.ptr(0)) : nullptr);

      if (compute_evs) {
        assemble_complex_evs(reinterpret_cast<cuDoubleComplex*>(ev + batch_idx * batch_stride_ev),
                             reinterpret_cast<cuDoubleComplex*>(ew + batch_idx * batch_stride_ew),
                             ev_tmp.ptr(0),
                             m);
      }
    }
  }
};

template <>
struct GeevImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  void operator()(int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const complex<float>* a,
                  complex<float>* ew,
                  complex<float>* ev)
  {
    bool compute_evs = ev != nullptr;

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      geev_template<complex<float>>(
        CUDA_C_32F,
        CUDA_C_32F,
        m,
        reinterpret_cast<const cuComplex*>(a + batch_idx * batch_stride_ev),
        reinterpret_cast<cuComplex*>(ew + batch_idx * batch_stride_ew),
        compute_evs ? reinterpret_cast<cuComplex*>(ev + batch_idx * batch_stride_ev) : nullptr);
    }
  }
};

template <>
struct GeevImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  void operator()(int64_t m,
                  int64_t num_batches,
                  int64_t batch_stride_ew,
                  int64_t batch_stride_ev,
                  const complex<double>* a,
                  complex<double>* ew,
                  complex<double>* ev)
  {
    bool compute_evs = ev != nullptr;

    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      geev_template<complex<double>>(
        CUDA_C_64F,
        CUDA_C_64F,
        m,
        reinterpret_cast<const cuDoubleComplex*>(a + batch_idx * batch_stride_ev),
        reinterpret_cast<cuDoubleComplex*>(ew + batch_idx * batch_stride_ew),
        compute_evs ? reinterpret_cast<cuDoubleComplex*>(ev + batch_idx * batch_stride_ev)
                    : nullptr);
    }
  }
};

/*static*/ void GeevTask::gpu_variant(TaskContext context)
{
  geev_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
