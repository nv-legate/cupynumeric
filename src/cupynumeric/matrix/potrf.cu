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

#include "cupynumeric/matrix/potrf.h"
#include "cupynumeric/matrix/potrf_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

using namespace legate;

template <typename PotrfBufferSize, typename Potrf, typename VAL>
static inline void potrf_template(PotrfBufferSize potrfBufferSize,
                                  Potrf potrf,
                                  VAL* array,
                                  int32_t m,
                                  int32_t n,
                                  cudaStream_t stream)
{
  auto uplo = CUBLAS_FILL_MODE_LOWER;

  auto cu_context = get_cusolver();
  CHECK_CUSOLVER(cusolverDnSetStream(cu_context, stream));

  int32_t bufferSize;
  CHECK_CUSOLVER(potrfBufferSize(cu_context, uplo, n, array, m, &bufferSize));

  auto buffer = create_buffer<VAL>(bufferSize, Memory::Kind::GPU_FB_MEM);
  auto info   = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(potrf(cu_context, uplo, n, array, m, buffer.ptr(0), bufferSize, info.ptr(0)));

  // TODO: We need a deferred exception to avoid this synchronization
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
  CUPYNUMERIC_CHECK_CUDA_STREAM(stream);

  if (info[0] != 0) {
    throw legate::TaskException("Matrix is not positive definite");
  }
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::FLOAT32>::operator()(float* array,
                                                                      int32_t m,
                                                                      int32_t n)
{
  auto stream = context.get_task_stream();
  potrf_template(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, array, m, n, stream);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::FLOAT64>::operator()(double* array,
                                                                      int32_t m,
                                                                      int32_t n)
{
  auto stream = context.get_task_stream();
  potrf_template(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, array, m, n, stream);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX64>::operator()(
  legate::Complex<float>* array, int32_t m, int32_t n)
{
  auto stream = context.get_task_stream();
  potrf_template(cusolverDnCpotrf_bufferSize,
                 cusolverDnCpotrf,
                 reinterpret_cast<cuComplex*>(array),
                 m,
                 n,
                 stream);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX128>::operator()(
  legate::Complex<double>* array, int32_t m, int32_t n)
{
  auto stream = context.get_task_stream();
  potrf_template(cusolverDnZpotrf_bufferSize,
                 cusolverDnZpotrf,
                 reinterpret_cast<cuDoubleComplex*>(array),
                 m,
                 n,
                 stream);
}

/*static*/ void PotrfTask::gpu_variant(TaskContext context)
{
  potrf_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
