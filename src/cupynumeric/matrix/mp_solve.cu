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

#include "cupynumeric/matrix/mp_solve.h"
#include "cupynumeric/matrix/mp_solve_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <typename VAL>
static inline void mp_solve_template(cal_comm_t comm,
                                     int nprow,
                                     int npcol,
                                     int64_t n,
                                     int64_t nrhs,
                                     int64_t nb,
                                     VAL* a_array,
                                     int64_t llda,
                                     VAL* b_array,
                                     int64_t lldb,
                                     cudaStream_t ctx_stream)
{
  const auto trans = CUBLAS_OP_N;

  auto handle = get_cusolvermp(ctx_stream);

  // synchronize all previous copies on default stream
  // cusolverMP has its unmodifiable stream to continue with
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(ctx_stream));

  cudaStream_t stream;
  CHECK_CUSOLVER(cusolverMpGetStream(handle, &stream));

  cusolverMpGrid_t grid = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateDeviceGrid(
    handle, &grid, comm, nprow, npcol, CUSOLVERMP_GRID_MAPPING_COL_MAJOR));

  cusolverMpMatrixDescriptor_t a_desc = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateMatrixDesc(
    &a_desc, grid, cudaTypeToDataType<VAL>::type, n, n, nb, nb, 0, 0, llda));

  cusolverMpMatrixDescriptor_t b_desc = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateMatrixDesc(
    &b_desc, grid, cudaTypeToDataType<VAL>::type, n, nrhs, nb, nb, 0, 0, lldb));

  size_t getrf_device_buffer_size = 0;
  size_t getrf_host_buffer_size   = 0;
  CHECK_CUSOLVER(cusolverMpGetrf_bufferSize(handle,
                                            n,
                                            n,
                                            a_array,
                                            1,
                                            1,
                                            a_desc,
                                            nullptr,
                                            cudaTypeToDataType<VAL>::type,
                                            &getrf_device_buffer_size,
                                            &getrf_host_buffer_size));

  size_t getrs_device_buffer_size = 0;
  size_t getrs_host_buffer_size   = 0;
  CHECK_CUSOLVER(cusolverMpGetrs_bufferSize(handle,
                                            trans,
                                            n,
                                            nrhs,
                                            a_array,
                                            1,
                                            1,
                                            a_desc,
                                            nullptr,
                                            b_array,
                                            1,
                                            1,
                                            b_desc,
                                            cudaTypeToDataType<VAL>::type,
                                            &getrs_device_buffer_size,
                                            &getrs_host_buffer_size));

  // ensure non-empty buffers
  size_t device_buffer_size =
    std::max(std::max(getrf_device_buffer_size, getrs_device_buffer_size), 1ul);
  size_t host_buffer_size = std::max(std::max(getrf_host_buffer_size, getrs_host_buffer_size), 1ul);

  auto device_buffer = create_buffer<int8_t>(device_buffer_size, Memory::Kind::GPU_FB_MEM);
  auto host_buffer   = create_buffer<int8_t>(host_buffer_size, Memory::Kind::Z_COPY_MEM);
  auto info          = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  // initialize to zero
  info[0] = 0;

  CHECK_CUSOLVER(cusolverMpGetrf(handle,
                                 n,
                                 n,
                                 a_array,
                                 1,
                                 1,
                                 a_desc,
                                 nullptr,
                                 cudaTypeToDataType<VAL>::type,
                                 device_buffer.ptr(0),
                                 device_buffer_size,
                                 host_buffer.ptr(0),
                                 host_buffer_size,
                                 info.ptr(0)));

  if (info[0] != 0) {
    throw legate::TaskException("Matrix is singular");
  }

  CHECK_CUSOLVER(cusolverMpGetrs(handle,
                                 trans,
                                 n,
                                 nrhs,
                                 a_array,
                                 1,
                                 1,
                                 a_desc,
                                 nullptr,
                                 b_array,
                                 1,
                                 1,
                                 b_desc,
                                 cudaTypeToDataType<VAL>::type,
                                 device_buffer.ptr(0),
                                 device_buffer_size,
                                 host_buffer.ptr(0),
                                 host_buffer_size,
                                 info.ptr(0)));

  // TODO: We need a deferred exception to avoid this synchronization
  CHECK_CAL(cal_stream_sync(comm, stream));
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  CHECK_CUSOLVER(cusolverMpDestroyMatrixDesc(a_desc));
  CHECK_CUSOLVER(cusolverMpDestroyMatrixDesc(b_desc));
  CHECK_CUSOLVER(cusolverMpDestroyGrid(grid));

  // FIXME: this should be synchronized with all participating tasks in order to quit gracefully
  if (info[0] != 0) {
    throw legate::TaskException("Matrix is singular");
  }
}

template <>
struct MpSolveImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit MpSolveImplBody(TaskContext context) : context(context) {}

  void operator()(cal_comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t n,
                  int64_t nrhs,
                  int64_t nb,
                  float* a_array,
                  int64_t llda,
                  float* b_array,
                  int64_t lldb)
  {
    auto stream = context.get_task_stream();
    mp_solve_template(comm, nprow, npcol, n, nrhs, nb, a_array, llda, b_array, lldb, stream);
  }
};

template <>
struct MpSolveImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit MpSolveImplBody(TaskContext context) : context(context) {}

  void operator()(cal_comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t n,
                  int64_t nrhs,
                  int64_t nb,
                  double* a_array,
                  int64_t llda,
                  double* b_array,
                  int64_t lldb)
  {
    auto stream = context.get_task_stream();
    mp_solve_template(comm, nprow, npcol, n, nrhs, nb, a_array, llda, b_array, lldb, stream);
  }
};

template <>
struct MpSolveImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit MpSolveImplBody(TaskContext context) : context(context) {}

  void operator()(cal_comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t n,
                  int64_t nrhs,
                  int64_t nb,
                  complex<float>* a_array,
                  int64_t llda,
                  complex<float>* b_array,
                  int64_t lldb)
  {
    auto stream = context.get_task_stream();
    mp_solve_template(comm,
                      nprow,
                      npcol,
                      n,
                      nrhs,
                      nb,
                      reinterpret_cast<cuComplex*>(a_array),
                      llda,
                      reinterpret_cast<cuComplex*>(b_array),
                      lldb,
                      stream);
  }
};

template <>
struct MpSolveImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit MpSolveImplBody(TaskContext context) : context(context) {}

  void operator()(cal_comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t n,
                  int64_t nrhs,
                  int64_t nb,
                  complex<double>* a_array,
                  int64_t llda,
                  complex<double>* b_array,
                  int64_t lldb)
  {
    auto stream = context.get_task_stream();
    mp_solve_template(comm,
                      nprow,
                      npcol,
                      n,
                      nrhs,
                      nb,
                      reinterpret_cast<cuDoubleComplex*>(a_array),
                      llda,
                      reinterpret_cast<cuDoubleComplex*>(b_array),
                      lldb,
                      stream);
  }
};

/*static*/ void MpSolveTask::gpu_variant(TaskContext context)
{
  mp_solve_template<VariantKind::GPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  MpSolveTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
