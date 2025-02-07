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
static inline void potrf_template(
  PotrfBufferSize potrfBufferSize, Potrf potrf, VAL* array, int32_t m, int32_t n)
{
  auto uplo = CUBLAS_FILL_MODE_LOWER;

  auto context = get_cusolver();
  auto stream  = get_cached_stream();
  CHECK_CUSOLVER(cusolverDnSetStream(context, stream));

  int32_t bufferSize;
  CHECK_CUSOLVER(potrfBufferSize(context, uplo, n, array, m, &bufferSize));

  auto buffer = create_buffer<VAL>(bufferSize, Memory::Kind::GPU_FB_MEM);
  auto info   = create_buffer<int32_t>(1, Memory::Kind::Z_COPY_MEM);

  CHECK_CUSOLVER(potrf(context, uplo, n, array, m, buffer.ptr(0), bufferSize, info.ptr(0)));

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
  potrf_template(cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, array, m, n);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::FLOAT64>::operator()(double* array,
                                                                      int32_t m,
                                                                      int32_t n)
{
  potrf_template(cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, array, m, n);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX64>::operator()(complex<float>* array,
                                                                        int32_t m,
                                                                        int32_t n)
{
  potrf_template(
    cusolverDnCpotrf_bufferSize, cusolverDnCpotrf, reinterpret_cast<cuComplex*>(array), m, n);
}

template <>
void PotrfImplBody<VariantKind::GPU, Type::Code::COMPLEX128>::operator()(complex<double>* array,
                                                                         int32_t m,
                                                                         int32_t n)
{
  potrf_template(
    cusolverDnZpotrf_bufferSize, cusolverDnZpotrf, reinterpret_cast<cuDoubleComplex*>(array), m, n);
}

/*static*/ void PotrfTask::gpu_variant(TaskContext context)
{
  potrf_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
