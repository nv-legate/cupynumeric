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

#include "generator.h"

#include "cupynumeric/cuda_help.h"

namespace randutilimpl {
static constexpr int blocksPerMultiProcessor = 2;    // TODO: refine => number of blocks per mp
static constexpr int blockDimX               = 256;  // TODO: refine ?

extern __shared__ char local_shared[];

template <typename gen_t>
__global__ void __launch_bounds__(blockDimX, blocksPerMultiProcessor)
  initgenerators(gen_t* gens, uint64_t seed, uint64_t generatorID)
{
  static_assert((sizeof(gen_t) & 3) == 0, "alignment issues");
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  // TODO: preload to shared
  gen_t gen = gens[id];
  // get offset from ID
  unsigned baseOffset = __brev((unsigned)id);
  uint64_t offset     = ((uint64_t)baseOffset) << 32;
  // init the generator
  curand_init(seed, generatorID, offset, &gen);
  // store back
  gens[id] = gen;
}

template <typename gen_t, typename func_t, typename out_t>
__global__ void __launch_bounds__(blockDimX, blocksPerMultiProcessor)
  gpu_draw(int ngenerators, gen_t* gens, func_t func, size_t N, out_t* draws)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  assert(id < ngenerators);
  // TODO: improve load
  gen_t gen = gens[id];
  for (size_t k = id; k < N; k += blockDim.x * gridDim.x) {
    draws[k] = func(gen);
  }
  // save state
  gens[id] = gen;
}

template <typename gen_t>
struct inner_generator<gen_t, randutilimpl::execlocation::DEVICE> : basegenerator {
  uint64_t seed;
  uint64_t generatorID;
  cudaStream_t stream;

  int multiProcessorCount = 0;
  bool asyncsupported     = true;
  int ngenerators;
  gen_t* generators = nullptr;

  inner_generator(uint64_t seed, uint64_t generatorID, cudaStream_t stream)
    : seed(seed), generatorID(generatorID), stream(stream)
  {
    int deviceId;
    CUPYNUMERIC_CHECK_CUDA(::cudaGetDevice(&deviceId));
    CUPYNUMERIC_CHECK_CUDA(
      ::cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));
    // get number of generators
    ngenerators = blockDimX * multiProcessorCount * blocksPerMultiProcessor;
#ifdef DEBUG_CUPYNUMERIC
    assert(ngenerators > 0);
#endif

    // allocate buffer for generators state
    int driverVersion, runtimeVersion;
    CUPYNUMERIC_CHECK_CUDA(::cudaDriverGetVersion(&driverVersion));
    CUPYNUMERIC_CHECK_CUDA(::cudaRuntimeGetVersion(&runtimeVersion));
    asyncsupported = ((driverVersion >= 10020) && (runtimeVersion >= 10020));
    if (asyncsupported) {
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
      CUPYNUMERIC_CHECK_CUDA(::cudaMallocAsync(&generators, ngenerators * sizeof(gen_t), stream));
#else
      CUPYNUMERIC_CHECK_CUDA(::cudaMalloc(&generators, ngenerators * sizeof(gen_t)));
#endif
    } else {
      CUPYNUMERIC_CHECK_CUDA(::cudaMalloc(&generators, ngenerators * sizeof(gen_t)));
    }

    // initialize generators
    initgenerators<<<blocksPerMultiProcessor * multiProcessorCount, blockDimX, 0, stream>>>(
      generators, seed, generatorID);
    CUPYNUMERIC_CHECK_CUDA(::cudaPeekAtLastError());
  }

  virtual void destroy() override
  {
    CUPYNUMERIC_CHECK_CUDA(::cudaStreamSynchronize(stream));
    if (asyncsupported) {
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
      CUPYNUMERIC_CHECK_CUDA(::cudaFreeAsync(generators, stream));
#else
      CUPYNUMERIC_CHECK_CUDA(::cudaFree(generators));
#endif
    } else {
      CUPYNUMERIC_CHECK_CUDA(::cudaFree(generators));
    }

    generators = nullptr;
  }

  virtual execlocation location() override { return randutilimpl::execlocation::DEVICE; }

  virtual int generatorTypeId() override { return generatorid<gen_t>::rng_type; }

  virtual ~inner_generator()
  {
    // TODO: warning message if generator has not been destroyed
  }

  template <typename func_t, typename out_t>
  curandStatus_t draw(func_t func, size_t N, out_t* out)
  {
    if (generators == nullptr) {  // destroyed was called
      return CURAND_STATUS_NOT_INITIALIZED;
    }
    gpu_draw<gen_t, func_t, out_t><<<multiProcessorCount * blocksPerMultiProcessor, blockDimX>>>(
      ngenerators, generators, func, N, out);
    return ::cudaPeekAtLastError() == cudaSuccess ? CURAND_STATUS_SUCCESS
                                                  : CURAND_STATUS_INTERNAL_ERROR;
  }
};

// partially specialize dispatcher to enable DEVICE implementation generation
template <typename func_t, typename out_t>
struct dispatcher<randutilimpl::execlocation::DEVICE, func_t, out_t> {
  static curandStatus_t run(randutilimpl::basegenerator* gen, func_t func, size_t N, out_t* out)
  {
    return inner_dispatch_sample<randutilimpl::execlocation::DEVICE, func_t, out_t>(
      gen, func, N, out);
  }
};

}  // namespace randutilimpl
