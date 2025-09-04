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

#include "cupynumeric/matrix/mp_potrf.h"
#include "cupynumeric/matrix/mp_potrf_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace Legion;
using namespace legate;

template <typename VAL>
static inline void mp_potrf_template(cal_comm_t comm,
                                     int nprow,
                                     int npcol,
                                     int64_t n,
                                     int64_t nb,
                                     VAL* array,
                                     int64_t lld,
                                     cudaStream_t ctx_stream)
{
  const auto uplo = CUBLAS_FILL_MODE_LOWER;

  auto handle = get_cusolvermp(ctx_stream);

  // synchronize all previous copies on default stream
  // cusolverMP has its unmodifiable stream to continue with
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(ctx_stream));

  cudaStream_t stream;
  CHECK_CUSOLVER(cusolverMpGetStream(handle, &stream));

  cusolverMpGrid_t grid = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateDeviceGrid(
    handle, &grid, comm, nprow, npcol, CUSOLVERMP_GRID_MAPPING_COL_MAJOR));

  cusolverMpMatrixDescriptor_t desc = nullptr;
  CHECK_CUSOLVER(cusolverMpCreateMatrixDesc(
    &desc, grid, cudaTypeToDataType<VAL>::type, n, n, nb, nb, 0, 0, lld));

  size_t device_buffer_size = 0;
  size_t host_buffer_size   = 0;
  CHECK_CUSOLVER(cusolverMpPotrf_bufferSize(handle,
                                            uplo,
                                            n,
                                            array,
                                            1,
                                            1,
                                            desc,
                                            cudaTypeToDataType<VAL>::type,
                                            &device_buffer_size,
                                            &host_buffer_size));

  // ensure non-empty buffers
  device_buffer_size = std::max(device_buffer_size, 1ul);
  host_buffer_size   = std::max(host_buffer_size, 1ul);

  auto device_buffer = create_buffer<int8_t>(device_buffer_size, Memory::Kind::GPU_FB_MEM);
  auto host_buffer   = create_buffer<int8_t>(host_buffer_size, Memory::Kind::Z_COPY_MEM);
  auto info          = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  // initialize to zero
  info[0] = 0;

  CHECK_CUSOLVER(cusolverMpPotrf(handle,
                                 uplo,
                                 n,
                                 array,
                                 1,
                                 1,
                                 desc,
                                 cudaTypeToDataType<VAL>::type,
                                 device_buffer.ptr(0),
                                 device_buffer_size,
                                 host_buffer.ptr(0),
                                 host_buffer_size,
                                 info.ptr(0)));

  // TODO: We need a deferred exception to avoid this synchronization
  CHECK_CAL(cal_stream_sync(comm, stream));
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  CHECK_CUSOLVER(cusolverMpDestroyMatrixDesc(desc));
  CHECK_CUSOLVER(cusolverMpDestroyGrid(grid));

  if (info[0] != 0) {
    throw legate::TaskException("Matrix is not positive definite");
  }
}

template <>
struct MpPotrfImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit MpPotrfImplBody(TaskContext context) : context(context) {}

  void operator()(
    cal_comm_t comm, int nprow, int npcol, int64_t n, int64_t nb, float* array, int64_t lld)
  {
    auto stream = context.get_task_stream();
    mp_potrf_template(comm, nprow, npcol, n, nb, array, lld, stream);
  }
};

template <>
struct MpPotrfImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit MpPotrfImplBody(TaskContext context) : context(context) {}

  void operator()(
    cal_comm_t comm, int nprow, int npcol, int64_t n, int64_t nb, double* array, int64_t lld)
  {
    auto stream = context.get_task_stream();
    mp_potrf_template(comm, nprow, npcol, n, nb, array, lld, stream);
  }
};

template <>
struct MpPotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit MpPotrfImplBody(TaskContext context) : context(context) {}

  void operator()(cal_comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t n,
                  int64_t nb,
                  complex<float>* array,
                  int64_t lld)
  {
    auto stream = context.get_task_stream();
    mp_potrf_template(comm, nprow, npcol, n, nb, reinterpret_cast<cuComplex*>(array), lld, stream);
  }
};

template <>
struct MpPotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit MpPotrfImplBody(TaskContext context) : context(context) {}

  void operator()(cal_comm_t comm,
                  int nprow,
                  int npcol,
                  int64_t n,
                  int64_t nb,
                  complex<double>* array,
                  int64_t lld)
  {
    auto stream = context.get_task_stream();
    mp_potrf_template(
      comm, nprow, npcol, n, nb, reinterpret_cast<cuDoubleComplex*>(array), lld, stream);
  }
};

/*static*/ void MpPotrfTask::gpu_variant(TaskContext context)
{
  mp_potrf_template<VariantKind::GPU>(context);
}

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  MpPotrfTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
