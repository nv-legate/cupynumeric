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

#include "cupynumeric/matrix/mp_qr.h"
#include "cupynumeric/matrix/mp_qr_template.inl"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <typename VAL, typename comm_t>
static inline void mp_qr_template(comm_t comm,
                                  int nprow,
                                  int npcol,
                                  int64_t m,
                                  int64_t n,
                                  int64_t mb,
                                  int64_t nb,
                                  VAL* a_array,
                                  VAL* q_array,
                                  size_t a_volume,
                                  size_t llda,
                                  size_t lldq,
                                  int rank,
                                  cudaStream_t ctx_stream)
{
  auto handle = get_cusolvermp(ctx_stream, nprow, npcol);

  // synchronize all previous copies on default stream
  // cusolverMP has its unmodifiable stream to continue with
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(ctx_stream));

  // must use cusolvermp mandated stream
  cudaStream_t stream;
  CHECK_CUSOLVER(cusolverMpGetStream(handle, &stream));

  cusolverMpGrid_t grid = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateDeviceGrid(
    handle, &grid, comm, nprow, npcol, CUSOLVERMP_GRID_MAPPING_COL_MAJOR));

  cusolverMpMatrixDescriptor_t a_desc = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateMatrixDesc(
    &a_desc, grid, cudaTypeToDataType<VAL>::type, m, n, mb, nb, 0, 0, llda));  // mirror
                                                                               // mp_solver

  cusolverMpMatrixDescriptor_t q_desc = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateMatrixDesc(
    &q_desc, grid, cudaTypeToDataType<VAL>::type, m, n, mb, nb, 0, 0, llda));  // mirror
                                                                               // mp_solver
  // local-tau array:
  //
  size_t tau_size = a_volume > 0 ? a_volume / llda : 1;
  auto d_tau      = create_buffer<VAL>(a_volume, Memory::Kind::GPU_FB_MEM);
  auto* d_tau_ptr = d_tau.ptr(0);
  // 0-initialize d_tau:
  thrust::transform(
    DEFAULT_POLICY.on(stream), d_tau_ptr, d_tau_ptr + tau_size, d_tau_ptr, [] __CUDA_HD__(VAL _) {
      return VAL{0};
    });

  constexpr int64_t IA{1};
  constexpr int64_t JA{1};

  // workspace sizes
  size_t workspaceInBytesOnDevice_geqrf = 0;
  size_t workspaceInBytesOnHost_geqrf   = 0;
  size_t workspaceInBytesOnDevice_ormqr = 0;
  size_t workspaceInBytesOnHost_ormqr   = 0;

  // ======================= GEQRF scratch space ====================
  //
  CHECK_CUSOLVER(cusolverMpGeqrf_bufferSize(handle,
                                            m,
                                            n,
                                            a_array,
                                            IA,
                                            JA,
                                            a_desc,
                                            cudaTypeToDataType<VAL>::type,
                                            &workspaceInBytesOnDevice_geqrf,
                                            &workspaceInBytesOnHost_geqrf));
  // ensure non-empty buffers
  workspaceInBytesOnDevice_geqrf = std::max(workspaceInBytesOnDevice_geqrf, 1ul);
  workspaceInBytesOnHost_geqrf   = std::max(workspaceInBytesOnHost_geqrf, 1ul);

  // ======================= ORMQR scratch space ====================
  //
  CHECK_CUSOLVER(cusolverMpOrmqr_bufferSize(handle,
                                            CUBLAS_SIDE_LEFT,
                                            CUBLAS_OP_N,
                                            m,
                                            n,
                                            n,
                                            a_array,
                                            IA,
                                            JA,
                                            a_desc,
                                            d_tau_ptr,
                                            q_array,
                                            IA,
                                            JA,
                                            q_desc,
                                            cudaTypeToDataType<VAL>::type,
                                            &workspaceInBytesOnDevice_ormqr,
                                            &workspaceInBytesOnHost_ormqr));

  // Use maximum workspace size between GEQRF and ORMQR
  size_t workspaceInBytesOnDevice =
    std::max(workspaceInBytesOnDevice_geqrf, workspaceInBytesOnDevice_ormqr);
  size_t workspaceInBytesOnHost =
    std::max(workspaceInBytesOnHost_geqrf, workspaceInBytesOnHost_ormqr);

  auto device_buffer = create_buffer<int8_t>(workspaceInBytesOnDevice, Memory::Kind::GPU_FB_MEM);
  auto host_buffer   = create_buffer<int8_t>(workspaceInBytesOnHost, Memory::Kind::Z_COPY_MEM);
  auto info          = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  // initialize status buffer to zero
  info.ptr(0)[0] = 0;
  CHECK_CUSOLVER(cusolverMpGeqrf(handle,
                                 m,
                                 n,
                                 a_array,
                                 IA,
                                 JA,
                                 a_desc,
                                 d_tau_ptr,
                                 cudaTypeToDataType<VAL>::type,
                                 device_buffer.ptr(0),
                                 workspaceInBytesOnDevice,
                                 host_buffer.ptr(0),
                                 workspaceInBytesOnHost,
                                 info.ptr(0)));

  // From mp_solver.cu: TODO: We need a deferred exception to avoid this synchronization
  //
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  // (long-term goal: use Legate::style formatting API / string interpolation)
  if (info[0] < 0) {
    std::stringstream ss;
    ss << "Incorrect value in cusolverMpGeqrf() " << std::abs(info[0]) << "-th argument.";
    throw legate::TaskException(ss.str());
  }

  // ======================= Obtaining q_array from d_tau ====================
  //
  // re-initialize status buffer to zero
  info[0] = 0;
  CHECK_CUSOLVER(cusolverMpOrmqr(handle,
                                 CUBLAS_SIDE_LEFT,
                                 CUBLAS_OP_N,
                                 m,
                                 n,
                                 n,
                                 a_array,
                                 IA,
                                 JA,
                                 a_desc,
                                 d_tau_ptr,
                                 q_array,  // assume Q preset to I and block-cyclic partitioned;
                                 IA,
                                 JA,
                                 q_desc,
                                 cudaTypeToDataType<VAL>::type,
                                 device_buffer.ptr(0),
                                 workspaceInBytesOnDevice,
                                 host_buffer.ptr(0),
                                 workspaceInBytesOnHost,
                                 info.ptr(0)));

  // From mp_solver.cu: TODO: We need a deferred exception to avoid this synchronization
  //
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));

  if (info[0] < 0) {
    std::stringstream ss;
    ss << "Incorrect value in cusolverMpOrmqr() " << std::abs(info[0]) << "-th argument.";
    throw legate::TaskException(ss.str());
  }

  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  CHECK_CUSOLVER(cusolverMpDestroyMatrixDesc(a_desc));
  CHECK_CUSOLVER(cusolverMpDestroyMatrixDesc(q_desc));
  CHECK_CUSOLVER(cusolverMpDestroyGrid(grid));
}

template <>
struct MpQRImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit MpQRImplBody(TaskContext context) : context(context) {}

  template <typename comm_t>
  void operator()(comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t m,
                  int64_t n,
                  int64_t mb,
                  int64_t nb,
                  float* a_array,
                  float* q_array,
                  size_t a_volume,
                  size_t llda,
                  size_t lldq,
                  int rank) const
  {
    auto stream = context.get_task_stream();
    mp_qr_template(
      comm, nprow, npcol, m, n, mb, nb, a_array, q_array, a_volume, llda, lldq, rank, stream);
  }
};

template <>
struct MpQRImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit MpQRImplBody(TaskContext context) : context(context) {}

  template <typename comm_t>
  void operator()(comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t m,
                  int64_t n,
                  int64_t mb,
                  int64_t nb,
                  double* a_array,
                  double* q_array,
                  size_t a_volume,
                  size_t llda,
                  size_t lldq,
                  int rank) const
  {
    auto stream = context.get_task_stream();
    mp_qr_template(
      comm, nprow, npcol, m, n, mb, nb, a_array, q_array, a_volume, llda, lldq, rank, stream);
  }
};

template <>
struct MpQRImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit MpQRImplBody(TaskContext context) : context(context) {}

  template <typename comm_t>
  void operator()(comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t m,
                  int64_t n,
                  int64_t mb,
                  int64_t nb,
                  complex<float>* a_array,
                  complex<float>* q_array,
                  size_t a_volume,
                  size_t llda,
                  size_t lldq,
                  int rank) const
  {
    auto stream = context.get_task_stream();
    mp_qr_template(comm,
                   nprow,
                   npcol,
                   m,
                   n,
                   mb,
                   nb,
                   reinterpret_cast<cuComplex*>(a_array),
                   reinterpret_cast<cuComplex*>(q_array),
                   a_volume,
                   llda,
                   lldq,
                   rank,
                   stream);
  }
};

template <>
struct MpQRImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit MpQRImplBody(TaskContext context) : context(context) {}

  template <typename comm_t>
  void operator()(comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t m,
                  int64_t n,
                  int64_t mb,
                  int64_t nb,
                  complex<double>* a_array,
                  complex<double>* q_array,
                  size_t a_volume,
                  size_t llda,
                  size_t lldq,
                  int rank) const
  {
    auto stream = context.get_task_stream();
    mp_qr_template(comm,
                   nprow,
                   npcol,
                   m,
                   n,
                   mb,
                   nb,
                   reinterpret_cast<cuDoubleComplex*>(a_array),
                   reinterpret_cast<cuDoubleComplex*>(q_array),
                   a_volume,
                   llda,
                   lldq,
                   rank,
                   stream);
  }
};

/*static*/ void MpQRTask::gpu_variant(TaskContext context)
{
  mp_qr_template<VariantKind::GPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  MpQRTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
