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

#include "cuda_help.h"

#include <map>

namespace cupynumeric {

struct cufftPlanCache;

struct CUDALibraries {
 public:
  CUDALibraries();
  ~CUDALibraries();

 private:
  // Prevent copying and overwriting
  CUDALibraries(const CUDALibraries& rhs)            = delete;
  CUDALibraries& operator=(const CUDALibraries& rhs) = delete;

 public:
  void finalize();
  int get_device_ordinal();
  const cudaDeviceProp& get_device_properties();
  cublasHandle_t get_cublas();
  cusolverDnHandle_t get_cusolver();
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
  cusolverMpHandle_t get_cusolvermp(cudaStream_t stream, int nprow, int npcol);
#endif
  [[nodiscard]] const cutensorHandle_t& get_cutensor();
  cufftContext get_cufft_plan(cufftType type, const cufftPlanParams& params, cudaStream_t stream);

 private:
  void finalize_cublas();
  void finalize_cusolver();
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
  void finalize_cusolvermp();
#endif
  void finalize_cutensor();

 private:
  bool finalized_;
  std::optional<int> ordinal_{};
  std::unique_ptr<cudaDeviceProp> device_prop_{};
  cublasContext* cublas_;
  cusolverDnContext* cusolver_;
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
  // per-grid handle map;
  // this was recommended by Marcin Rogowski (cusolvermp team)
  // following cusolvermp upgrade to 0.7.x
  // to avoid re-using a handle for a different grid
  // which triggers a bug in cusolvermp
  // see https://github.com/nv-legate/cupynumeric.internal/pull/672
  //
  std::map<std::pair<int, int>, cusolverMpHandle_t> cusolvermp_handle_map_;
#endif
  std::optional<cutensorHandle_t> cutensor_{};
  std::map<cufftType, cufftPlanCache*> plan_caches_;
};

}  // namespace cupynumeric
