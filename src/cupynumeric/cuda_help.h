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

#include "legate.h"

#if !LEGATE_DEFINED(LEGATE_NVCC)
#error "This header can only be included from .cu files"
#endif

#include "cupynumeric/arg.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
#include <cusolverMp.h>
#endif
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cutensor.h>
#include <nccl.h>

#define THREADS_PER_BLOCK 128
#define MIN_CTAS_PER_SM 4
#define MAX_REDUCTION_CTAS 1024
#define COOPERATIVE_THREADS 256
#define COOPERATIVE_CTAS_PER_SM 4

namespace cupynumeric {

namespace detail {

__host__ inline const char* cufft_error_string(cufftResult result)
{
  switch (result) {
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
#ifdef CUFFT_INCOMPLETE_PARAMETER_LIST
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
#ifdef CUFFT_PARSE_ERROR
    case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
#endif
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
#ifdef CUFFT_MISSING_DEPENDENCY
    case CUFFT_MISSING_DEPENDENCY: return "CUFFT_MISSING_DEPENDENCY";
#endif
#ifdef CUFFT_NVRTC_FAILURE
    case CUFFT_NVRTC_FAILURE: return "CUFFT_NVRTC_FAILURE";
#endif
#ifdef CUFFT_NVJITLINK_FAILURE
    case CUFFT_NVJITLINK_FAILURE: return "CUFFT_NVJITLINK_FAILURE";
#endif
#ifdef CUFFT_NVSHMEM_FAILURE
    case CUFFT_NVSHMEM_FAILURE: return "CUFFT_NVSHMEM_FAILURE";
#endif
    default: return "<unknown>";
  }
}

__host__ inline const char* cusolver_error_string(cusolverStatus_t status)
{
  switch (status) {
    case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
#ifdef CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED
    case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
#endif
#ifdef CUSOLVER_STATUS_IRS_PARAMS_INVALID
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
#endif
#ifdef CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
#endif
#ifdef CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
#endif
#ifdef CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
#endif
#ifdef CUSOLVER_STATUS_IRS_INTERNAL_ERROR
    case CUSOLVER_STATUS_IRS_INTERNAL_ERROR: return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
#endif
#ifdef CUSOLVER_STATUS_IRS_NOT_SUPPORTED
    case CUSOLVER_STATUS_IRS_NOT_SUPPORTED: return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
#endif
#ifdef CUSOLVER_STATUS_IRS_OUT_OF_RANGE
    case CUSOLVER_STATUS_IRS_OUT_OF_RANGE: return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
#endif
#ifdef CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES
    case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
      return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
#endif
#ifdef CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED
    case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
#endif
#ifdef CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED
    case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED: return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
#endif
#ifdef CUSOLVER_STATUS_IRS_MATRIX_SINGULAR
    case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR: return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
#endif
#ifdef CUSOLVER_STATUS_INVALID_WORKSPACE
    case CUSOLVER_STATUS_INVALID_WORKSPACE: return "CUSOLVER_STATUS_INVALID_WORKSPACE";
#endif
    default: return "<unknown>";
  }
}

__host__ inline const char* cublas_error_string(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "<unknown>";
  }
}

}  // namespace detail

__host__ inline void check_cuda(cudaError_t error, const char* file, int line)
{
  if (error != cudaSuccess) {
    LEGATE_ABORT("Internal CUDA failure with error ",
                 cudaGetErrorString(error),
                 " (",
                 cudaGetErrorName(error),
                 ", code ",
                 error,
                 ") in file ",
                 file,
                 " at line ",
                 line);
  }
}

__host__ inline void check_cublas(cublasStatus_t status, const char* file, int line)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    LEGATE_ABORT("Internal cuBLAS failure with error ",
                 detail::cublas_error_string(status),
                 " (code ",
                 status,
                 ") in file ",
                 file,
                 " at line ",
                 line);
  }
}

__host__ inline void check_cufft(cufftResult result, const char* file, int line)
{
  if (result != CUFFT_SUCCESS) {
    LEGATE_ABORT("Internal cuFFT failure with error ",
                 detail::cufft_error_string(result),
                 " (code ",
                 result,
                 ") in file ",
                 file,
                 " at line ",
                 line);
  }
}

__host__ inline void check_cusolver(cusolverStatus_t status, const char* file, int line)
{
  if (status != CUSOLVER_STATUS_SUCCESS) {
    LEGATE_ABORT("Internal cuSOLVER failure with error ",
                 detail::cusolver_error_string(status),
                 " (code ",
                 status,
                 ") in file ",
                 file,
                 " at line ",
                 line);
  }
}

__host__ inline void check_cutensor(cutensorStatus_t result, const char* file, int line)
{
  if (result != CUTENSOR_STATUS_SUCCESS) {
    LEGATE_ABORT("Internal cuTENSOR failure with error ",
                 cutensorGetErrorString(result),
                 " (code ",
                 result,
                 ") in file ",
                 file,
                 " at line ",
                 line);
  }
}

__host__ inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    LEGATE_ABORT("Internal NCCL failure with error ",
                 ncclGetErrorString(error),
                 " (code ",
                 error,
                 ") in file ",
                 file,
                 " at line ",
                 line);
  }
}

}  // namespace cupynumeric

#define CHECK_CUBLAS(expr)                                     \
  do {                                                         \
    cublasStatus_t __result__ = (expr);                        \
    cupynumeric::check_cublas(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUFFT(expr)                                     \
  do {                                                        \
    cufftResult __result__ = (expr);                          \
    cupynumeric::check_cufft(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUSOLVER(expr)                                     \
  do {                                                           \
    cusolverStatus_t __result__ = (expr);                        \
    cupynumeric::check_cusolver(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_CUTENSOR(expr)                                     \
  do {                                                           \
    cutensorStatus_t __result__ = (expr);                        \
    cupynumeric::check_cutensor(__result__, __FILE__, __LINE__); \
  } while (false)

#define CHECK_NCCL(...)                                      \
  do {                                                       \
    ncclResult_t __result__ = (__VA_ARGS__);                 \
    cupynumeric::check_nccl(__result__, __FILE__, __LINE__); \
  } while (false)

#define CUPYNUMERIC_CHECK_CUDA(...)                          \
  do {                                                       \
    cudaError_t __result__ = (__VA_ARGS__);                  \
    cupynumeric::check_cuda(__result__, __FILE__, __LINE__); \
  } while (false)

#ifdef DEBUG_CUPYNUMERIC
#define CUPYNUMERIC_CHECK_CUDA_STREAM(stream)              \
  do {                                                     \
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream)); \
    CUPYNUMERIC_CHECK_CUDA(cudaPeekAtLastError());         \
  } while (false)
#else
#define CUPYNUMERIC_CHECK_CUDA_STREAM(stream)      \
  do {                                             \
    CUPYNUMERIC_CHECK_CUDA(cudaPeekAtLastError()); \
  } while (false)
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

// Must go here since it depends on CUPYNUMERIC_CHECK_CUDA(), which is defined in this header...
#include "cupynumeric/device_scalar_reduction_buffer.h"

namespace cupynumeric {

template <typename T>
struct cudaTypeToDataType;

template <>
struct cudaTypeToDataType<float> {
  static constexpr cudaDataType type = CUDA_R_32F;
};

template <>
struct cudaTypeToDataType<double> {
  static constexpr cudaDataType type = CUDA_R_64F;
};

template <>
struct cudaTypeToDataType<cuComplex> {
  static constexpr cudaDataType type = CUDA_C_32F;
};

template <>
struct cudaTypeToDataType<cuDoubleComplex> {
  static constexpr cudaDataType type = CUDA_C_64F;
};

__device__ inline size_t global_tid_1d()
{
  return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// Type-erased copy of `elem_size` bytes from `src` to `dst` using a single
// typed load+store for the power-of-2 widths covering every NumPy scalar
// dtype (1/2/4/8/16 bytes).  Any other width falls back to byte-wise
// `memcpy`.  Intended for type-erased dense kernels (e.g. zipgather /
// zipscatter) that want to avoid generating one specialisation per element
// type while still emitting wide loads/stores for the common sizes.
__device__ __forceinline__ void copy_elements(char* dst, const char* src, size_t elem_size)
{
  switch (elem_size) {
    case 1: *reinterpret_cast<uint8_t*>(dst) = *reinterpret_cast<const uint8_t*>(src); break;
    case 2: *reinterpret_cast<uint16_t*>(dst) = *reinterpret_cast<const uint16_t*>(src); break;
    case 4: *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src); break;
    case 8: *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src); break;
    case 16: *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src); break;
    default: memcpy(dst, src, elem_size); break;
  }
}

struct cufftPlan {
  cufftHandle handle;
  size_t workarea_size;
};

class cufftContext {
 public:
  cufftContext(cufftPlan* plan);
  ~cufftContext();

 public:
  cufftContext(const cufftContext&)            = delete;
  cufftContext& operator=(const cufftContext&) = delete;

 public:
  cufftContext(cufftContext&&)            = default;
  cufftContext& operator=(cufftContext&&) = default;

 public:
  cufftHandle& handle();
  size_t workareaSize();
  void setCallback(cufftXtCallbackType type, void* callback, void* data);

 private:
  cufftPlan* plan_{nullptr};
  std::vector<cufftXtCallbackType> callback_types_{};
};

struct cufftPlanParams {
  int rank;
  long long int n[LEGION_MAX_DIM]       = {0};
  long long int inembed[LEGION_MAX_DIM] = {0};
  long long int istride;
  long long int idist;
  long long int onembed[LEGION_MAX_DIM] = {0};
  long long int ostride;
  long long int odist;
  long long int batch;

  cufftPlanParams(const Legion::DomainPoint& size);
  cufftPlanParams(int rank,
                  long long int* n,
                  long long int* inembed,
                  long long int istride,
                  long long int idist,
                  long long int* onembed,
                  long long int ostride,
                  long long int odist,
                  long long int batch);

  bool operator==(const cufftPlanParams& other) const;
  std::string to_string() const;
};

typedef cusolverStatus_t (*cusolverDnXgeev_bufferSize_handle)(cusolverDnHandle_t handle,
                                                              cusolverDnParams_t params,
                                                              cusolverEigMode_t jobvl,
                                                              cusolverEigMode_t jobvr,
                                                              int64_t n,
                                                              cudaDataType dataTypeA,
                                                              const void* A,
                                                              int64_t lda,
                                                              cudaDataType dataTypeW,
                                                              const void* W,
                                                              cudaDataType dataTypeVL,
                                                              const void* VL,
                                                              int64_t ldvl,
                                                              cudaDataType dataTypeVR,
                                                              const void* VR,
                                                              int64_t ldvr,
                                                              cudaDataType computeType,
                                                              size_t* workspaceInBytesOnDevice,
                                                              size_t* workspaceInBytesOnHost);

typedef cusolverStatus_t (*cusolverDnXgeev_handle)(cusolverDnHandle_t handle,
                                                   cusolverDnParams_t params,
                                                   cusolverEigMode_t jobvl,
                                                   cusolverEigMode_t jobvr,
                                                   int64_t n,
                                                   cudaDataType dataTypeA,
                                                   void* A,
                                                   int64_t lda,
                                                   cudaDataType dataTypeW,
                                                   void* W,
                                                   cudaDataType dataTypeVL,
                                                   void* VL,
                                                   int64_t ldvl,
                                                   cudaDataType dataTypeVR,
                                                   void* VR,
                                                   int64_t ldvr,
                                                   cudaDataType computeType,
                                                   void* bufferOnDevice,
                                                   size_t workspaceInBytesOnDevice,
                                                   void* bufferOnHost,
                                                   size_t workspaceInBytesOnHost,
                                                   int* info);

typedef cusolverStatus_t (*cusolverDnXsyevBatched_bufferSize_handle)(
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  cudaDataType dataTypeA,
  const void* A,
  int64_t lda,
  cudaDataType dataTypeW,
  const void* W,
  cudaDataType computeType,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  int64_t batchSize);

typedef cusolverStatus_t (*cusolverDnXsyevBatched_handle)(cusolverDnHandle_t handle,
                                                          cusolverDnParams_t params,
                                                          cusolverEigMode_t jobz,
                                                          cublasFillMode_t uplo,
                                                          int64_t n,
                                                          cudaDataType dataTypeA,
                                                          void* A,
                                                          int64_t lda,
                                                          cudaDataType dataTypeW,
                                                          void* W,
                                                          cudaDataType computeType,
                                                          void* bufferOnDevice,
                                                          size_t workspaceInBytesOnDevice,
                                                          void* bufferOnHost,
                                                          size_t workspaceInBytesOnHost,
                                                          int* info,
                                                          int64_t batchSize);

struct CuSolverExtraSymbols {
 private:
  void* cusolver_lib;

 public:
  // geev support (since 12.6)
  cusolverDnXgeev_bufferSize_handle cusolver_geev_bufferSize;
  cusolverDnXgeev_handle cusolver_geev;
  bool has_geev;

  cusolverDnXsyevBatched_bufferSize_handle cusolver_syev_batched_bufferSize;
  cusolverDnXsyevBatched_handle cusolver_syev_batched;
  bool has_syev_batched;

  CuSolverExtraSymbols();
  ~CuSolverExtraSymbols();

  // Prevent copying and overwriting
  CuSolverExtraSymbols(const CuSolverExtraSymbols& rhs)            = delete;
  CuSolverExtraSymbols& operator=(const CuSolverExtraSymbols& rhs) = delete;

  void finalize();
};

// Defined in cudalibs.cu

int get_device_ordinal();
const cudaDeviceProp& get_device_properties();
cublasHandle_t get_cublas();
cusolverDnHandle_t get_cusolver();
CuSolverExtraSymbols* get_cusolver_extra_symbols();
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
cusolverMpHandle_t get_cusolvermp(cudaStream_t stream, int nprow, int npcol);
#endif
[[nodiscard]] const cutensorHandle_t& get_cutensor();
cufftContext get_cufft_plan(cufftType type, const cufftPlanParams& params, cudaStream_t stream);

template <typename T>
__device__ __forceinline__ T shuffle(unsigned mask, T var, int laneMask, int width)
{
  // return __shfl_xor_sync(0xffffffff, value, i, 32);
  int array[(sizeof(T) + sizeof(int) - 1) / sizeof(int)];
  memcpy(array, &var, sizeof(T));
  for (int& value : array) {
    const int tmp = __shfl_xor_sync(mask, value, laneMask, width);
    value         = tmp;
  }
  memcpy(&var, array, sizeof(T));
  return var;
}

template <typename T>
struct HasNativeShuffle {
  static constexpr bool value = true;
};

template <typename T>
struct HasNativeShuffle<legate::Complex<T>> {
  static constexpr bool value = false;
};

template <typename T>
struct HasNativeShuffle<Argval<T>> {
  static constexpr bool value = false;
};

template <typename T, typename REDUCTION>
__device__ __forceinline__ void reduce_output(DeviceScalarReductionBuffer<REDUCTION> result,
                                              T value)
{
  __shared__ T trampoline[THREADS_PER_BLOCK / 32];
  // Reduce across the warp
  const int laneid = threadIdx.x & 0x1f;
  const int warpid = threadIdx.x >> 5;
  for (int i = 16; i >= 1; i /= 2) {
    T shuffle_value;
    if constexpr (HasNativeShuffle<T>::value) {
      shuffle_value = __shfl_xor_sync(0xffffffff, value, i, 32);
    } else {
      shuffle_value = shuffle(0xffffffff, value, i, 32);
    }
    REDUCTION::template fold<true /*exclusive*/>(value, shuffle_value);
  }
  // Write warp values into shared memory
  if ((laneid == 0) && (warpid > 0)) {
    trampoline[warpid] = value;
  }
  __syncthreads();
  // Output reduction
  if (threadIdx.x == 0) {
    for (int i = 1; i < (THREADS_PER_BLOCK / 32); i++) {
      REDUCTION::template fold<true /*exclusive*/>(value, trampoline[i]);
    }
    result.reduce<false /*EXCLUSIVE*/>(value);
    // Make sure the result is visible externally
    __threadfence_system();
  }
}

template <typename T>
__device__ __forceinline__ T load_streaming(const T* ptr)
{
  return *ptr;
}

// Specializations to use PTX cache qualifiers to prevent
// loads from interfering with cache state
// Use .cs qualifier to mark the line as evict-first
template <>
__device__ __forceinline__ uint16_t load_streaming<uint16_t>(const uint16_t* ptr)
{
  uint16_t value;
  asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ uint32_t load_streaming<uint32_t>(const uint32_t* ptr)
{
  uint32_t value;
  asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ uint64_t load_streaming<uint64_t>(const uint64_t* ptr)
{
  uint64_t value;
  asm volatile("ld.global.cs.u64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ int16_t load_streaming<int16_t>(const int16_t* ptr)
{
  int16_t value;
  asm volatile("ld.global.cs.s16 %0, [%1];" : "=h"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ int32_t load_streaming<int32_t>(const int32_t* ptr)
{
  int32_t value;
  asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ int64_t load_streaming<int64_t>(const int64_t* ptr)
{
  int64_t value;
  asm volatile("ld.global.cs.s64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
  return value;
}

// No half because inline ptx is dumb about the type

template <>
__device__ __forceinline__ float load_streaming<float>(const float* ptr)
{
  float value;
  asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(value) : "l"(ptr) : "memory");
  return value;
}

template <>
__device__ __forceinline__ double load_streaming<double>(const double* ptr)
{
  double value;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(value) : "l"(ptr) : "memory");
  return value;
}

template <typename T>
__device__ __forceinline__ void store_streaming(T* ptr, T value)
{
  *ptr = value;
}

// Specializations to use PTX cache qualifiers to avoid
// invalidating read data from caches as we are writing
// Use .cs qualifier to evict first at all levels
template <>
__device__ __forceinline__ void store_streaming<uint16_t>(uint16_t* ptr, uint16_t value)
{
  asm volatile("st.global.cs.u16 [%0], %1;" : : "l"(ptr), "h"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<uint32_t>(uint32_t* ptr, uint32_t value)
{
  asm volatile("st.global.cs.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<uint64_t>(uint64_t* ptr, uint64_t value)
{
  asm volatile("st.global.cs.u64 [%0], %1;" : : "l"(ptr), "l"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<int16_t>(int16_t* ptr, int16_t value)
{
  asm volatile("st.global.cs.s16 [%0], %1;" : : "l"(ptr), "h"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<int32_t>(int32_t* ptr, int32_t value)
{
  asm volatile("st.global.cs.s32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<int64_t>(int64_t* ptr, int64_t value)
{
  asm volatile("st.global.cs.s64 [%0], %1;" : : "l"(ptr), "l"(value) : "memory");
}

// No half because inline ptx is dumb about the type

template <>
__device__ __forceinline__ void store_streaming<float>(float* ptr, float value)
{
  asm volatile("st.global.cs.f32 [%0], %1;" : : "l"(ptr), "f"(value) : "memory");
}

template <>
__device__ __forceinline__ void store_streaming<double>(double* ptr, double value)
{
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(ptr), "d"(value) : "memory");
}

}  // namespace cupynumeric
