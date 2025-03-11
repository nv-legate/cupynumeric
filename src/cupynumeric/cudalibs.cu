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

#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/random/bitgenerator.h"

#include "cudalibs.h"

#include <dlfcn.h>
#include <stdio.h>

using namespace legate;

namespace cupynumeric {

static Logger log_cudalibs("cupynumeric.cudalibs");

cufftContext::cufftContext(cufftPlan* plan) : plan_(plan) {}

cufftContext::~cufftContext()
{
  auto& hdl = handle();
  for (auto type : callback_types_) {
    CHECK_CUFFT(cufftXtClearCallback(hdl, type));
  }
  CHECK_CUFFT(cufftSetWorkArea(hdl, nullptr));
}

cufftHandle& cufftContext::handle() { return plan_->handle; }

size_t cufftContext::workareaSize() { return plan_->workarea_size; }

void cufftContext::setCallback(cufftXtCallbackType type, void* callback, void* data)
{
  void* callbacks[] = {callback};
  void* datas[]     = {data};
  CHECK_CUFFT(cufftXtSetCallback(handle(), callbacks, type, datas));
}

cufftPlanParams::cufftPlanParams(const DomainPoint& size)
  : rank(size.dim),
    n{0},
    inembed{0},
    onembed{0},
    istride(1),
    idist(1),
    ostride(1),
    odist(1),
    batch(1)
{
  for (int dim = 0; dim < rank; ++dim) {
    n[dim] = size[dim];
  }
}

cufftPlanParams::cufftPlanParams(int rank,
                                 long long int* n_,
                                 long long int* inembed_,
                                 long long int istride,
                                 long long int idist,
                                 long long int* onembed_,
                                 long long int ostride,
                                 long long int odist,
                                 long long int batch)
  : rank(rank), istride(istride), idist(idist), ostride(ostride), odist(odist), batch(batch)
{
  for (int dim = 0; dim < rank; ++dim) {
    n[dim]       = n_[dim];
    inembed[dim] = inembed_[dim];
    onembed[dim] = onembed_[dim];
  }
}

bool cufftPlanParams::operator==(const cufftPlanParams& other) const
{
  bool equal = rank == other.rank && istride == other.istride && idist == other.idist &&
               ostride == other.ostride && odist == other.odist && batch == other.batch;
  if (equal) {
    for (int dim = 0; dim < rank; ++dim) {
      equal = equal && (n[dim] == other.n[dim]);
      equal = equal && (inembed[dim] == other.inembed[dim]);
      equal = equal && (onembed[dim] == other.onembed[dim]);
      if (!equal) {
        break;
      }
    }
  }
  return equal;
}

std::string cufftPlanParams::to_string() const
{
  std::ostringstream ss;
  ss << "cufftPlanParams[rank(" << rank << "), n(" << n[0];
  for (int i = 1; i < rank; ++i) {
    ss << "," << n[i];
  }
  ss << "), inembed(" << inembed[0];
  for (int i = 1; i < rank; ++i) {
    ss << "," << inembed[i];
  }
  ss << "), istride(" << istride << "), idist(" << idist << "), onembed(" << onembed[0];
  for (int i = 1; i < rank; ++i) {
    ss << "," << onembed[i];
  }
  ss << "), ostride(" << ostride << "), odist(" << odist << "), batch(" << batch << ")]";
  return std::move(ss).str();
}

struct cufftPlanCache {
 private:
  // Maximum number of plans to keep per dimension
  static constexpr int32_t MAX_PLANS = 4;

 private:
  struct LRUEntry {
    std::unique_ptr<cufftPlan> plan{nullptr};
    std::unique_ptr<cufftPlanParams> params{nullptr};
    uint32_t lru_index{0};
  };

 public:
  cufftPlanCache(cufftType type);
  ~cufftPlanCache();

 public:
  cufftPlan* get_cufft_plan(const cufftPlanParams& params);

 private:
  using Cache = std::array<LRUEntry, MAX_PLANS>;
  std::array<Cache, LEGATE_MAX_DIM + 1> cache_{};
  cufftType type_;
  int64_t cache_hits_{0};
  int64_t cache_requests_{0};
};

cufftPlanCache::cufftPlanCache(cufftType type) : type_(type)
{
  for (auto& cache : cache_) {
    for (auto& entry : cache) {
      assert(0 == entry.lru_index);
    }
  }
}

cufftPlanCache::~cufftPlanCache()
{
  for (auto& cache : cache_) {
    for (auto& entry : cache) {
      if (entry.plan != nullptr) {
        CHECK_CUFFT(cufftDestroy(entry.plan->handle));
      }
    }
  }
}

cufftPlan* cufftPlanCache::get_cufft_plan(const cufftPlanParams& params)
{
  cache_requests_++;
  int32_t match = -1;
  auto& cache   = cache_[params.rank];
  for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
    auto& entry = cache[idx];
    if (nullptr == entry.plan) {
      break;
    }
    if (*entry.params == params) {
      match = idx;
      cache_hits_++;
      break;
    }
  }

  float hit_rate = static_cast<float>(cache_hits_) / cache_requests_;

  cufftPlan* result{nullptr};
  // If there's no match, we create a new plan
  if (-1 == match) {
    log_cudalibs.debug() << "[cufftPlanCache] no match found for " << params.to_string()
                         << " (type: " << type_ << ", hitrate: " << hit_rate << ")";
    int32_t plan_index = -1;
    for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
      auto& entry = cache[idx];
      if (nullptr == entry.plan) {
        log_cudalibs.debug() << "[cufftPlanCache] found empty entry " << idx << " (type: " << type_
                             << ")";
        entry.plan      = std::make_unique<cufftPlan>();
        entry.lru_index = idx;
        plan_index      = idx;
        break;
      } else if (entry.lru_index == MAX_PLANS - 1) {
        log_cudalibs.debug() << "[cufftPlanCache] evict entry " << idx << " for "
                             << entry.params->to_string() << " (type: " << type_ << ")";
        CHECK_CUFFT(cufftDestroy(entry.plan->handle));
        plan_index = idx;
        // create new plan
        entry.plan = std::make_unique<cufftPlan>();
        break;
      } else {
        entry.lru_index++;
      }
    }
    assert(plan_index != -1);
    auto& entry = cache[plan_index];

    if (entry.lru_index != 0) {
      for (int32_t idx = plan_index + 1; idx < MAX_PLANS; ++idx) {
        auto& other = cache[idx];
        if (nullptr == other.plan) {
          break;
        }
        ++other.lru_index;
      }
      entry.lru_index = 0;
    }

    entry.params = std::make_unique<cufftPlanParams>(params);
    result       = entry.plan.get();

    auto stream = get_cached_stream();
    CHECK_CUFFT(cufftCreate(&result->handle));
    CHECK_CUFFT(cufftSetAutoAllocation(result->handle, 0 /*we'll do the allocation*/));
    // this should always be the correct stream, as we have a cache per GPU-proc
    CHECK_CUFFT(cufftSetStream(result->handle, stream));
    CHECK_CUFFT(cufftMakePlanMany64(result->handle,
                                    entry.params->rank,
                                    entry.params->n,
                                    entry.params->inembed[0] != 0 ? entry.params->inembed : nullptr,
                                    entry.params->istride,
                                    entry.params->idist,
                                    entry.params->onembed[0] != 0 ? entry.params->onembed : nullptr,
                                    entry.params->ostride,
                                    entry.params->odist,
                                    type_,
                                    entry.params->batch,
                                    &result->workarea_size));

  }
  // Otherwise, we return the cached plan and adjust the LRU count
  else {
    log_cudalibs.debug() << "[cufftPlanCache] found match for " << params.to_string()
                         << " (type: " << type_ << ", hitrate: " << hit_rate << ")";
    auto& entry = cache[match];
    result      = entry.plan.get();

    if (entry.lru_index != 0) {
      for (int32_t idx = 0; idx < MAX_PLANS; ++idx) {
        auto& other = cache[idx];
        if (other.lru_index < entry.lru_index) {
          ++other.lru_index;
        }
      }
      entry.lru_index = 0;
    }
    auto stream = get_cached_stream();
    CHECK_CUFFT(cufftSetStream(result->handle, stream));
  }
  return result;
}

CuSolverExtraSymbols::CuSolverExtraSymbols()
{
  cusolver_lib = dlopen("libcusolver.so", RTLD_LAZY | RTLD_DEEPBIND);
  void* fn1    = dlsym(cusolver_lib, "cusolverDnXgeev_bufferSize");
  if (fn1 == nullptr) {
    dlerror();
  } else {
    cusolver_geev_bufferSize = (cusolverDnXgeev_bufferSize_handle)fn1;
    has_geev                 = true;
  }

  void* fn2 = dlsym(cusolver_lib, "cusolverDnXgeev");
  if (fn2 == nullptr) {
    has_geev                 = false;
    cusolver_geev_bufferSize = nullptr;
    dlerror();
  } else {
    cusolver_geev = (cusolverDnXgeev_handle)fn2;
  }
}

void CuSolverExtraSymbols::finalize()
{
  cusolver_geev            = nullptr;
  cusolver_geev_bufferSize = nullptr;
  has_geev                 = false;
  if (cusolver_lib != nullptr) {
    dlclose(cusolver_lib);
  }
}

CuSolverExtraSymbols::~CuSolverExtraSymbols() { finalize(); }

CUDALibraries::CUDALibraries()
  : finalized_(false),
    cublas_(nullptr),
    cusolver_(nullptr),
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
    cusolvermp_(nullptr),
#endif
    plan_caches_()
{
}

CUDALibraries::~CUDALibraries() { finalize(); }

void CUDALibraries::finalize()
{
  if (finalized_) {
    return;
  }
  if (cublas_ != nullptr) {
    finalize_cublas();
  }
  if (cusolver_ != nullptr) {
    finalize_cusolver();
  }

#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
  if (cusolvermp_ != nullptr) {
    finalize_cusolvermp();
  }
#endif
  if (cutensor_ != nullptr) {
    finalize_cutensor();
  }
  for (auto& pair : plan_caches_) {
    delete pair.second;
  }
  finalized_ = true;
}

void CUDALibraries::finalize_cublas()
{
  CHECK_CUBLAS(cublasDestroy(cublas_));
  cublas_ = nullptr;
}

void CUDALibraries::finalize_cusolver()
{
  CHECK_CUSOLVER(cusolverDnDestroy(cusolver_));
  cusolver_ = nullptr;
}

#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
void CUDALibraries::finalize_cusolvermp()
{
  CHECK_CUSOLVER(cusolverMpDestroy(cusolvermp_));
  cusolvermp_ = nullptr;
}
#endif

void CUDALibraries::finalize_cutensor()
{
  if (cutensor_.has_value()) {
    CHECK_CUTENSOR(cutensorDestroy(*cutensor_));
  }
  cutensor_.reset();
}

int CUDALibraries::get_device_ordinal()
{
  if (ordinal_.has_value()) {
    return *ordinal_;
  }
  int ordinal{-1};
  CUPYNUMERIC_CHECK_CUDA(cudaGetDevice(&ordinal));
  ordinal_ = ordinal;
  return ordinal;
}

const cudaDeviceProp& CUDALibraries::get_device_properties()
{
  if (device_prop_) {
    return *device_prop_;
  }
  device_prop_ = std::make_unique<cudaDeviceProp>();
  CUPYNUMERIC_CHECK_CUDA(cudaGetDeviceProperties(device_prop_.get(), get_device_ordinal()));
  return *device_prop_;
}

cublasHandle_t CUDALibraries::get_cublas()
{
  if (nullptr == cublas_) {
    CHECK_CUBLAS(cublasCreate(&cublas_));
    const char* fast_math = getenv("CUPYNUMERIC_FAST_MATH");
    if (fast_math != nullptr && atoi(fast_math) > 0) {
      // Enable acceleration of single precision routines using TF32 tensor cores.
      cublasStatus_t status = cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "WARNING: cuBLAS does not support Tensor cores!");
      }
    }
  }
  return cublas_;
}

cusolverDnHandle_t CUDALibraries::get_cusolver()
{
  if (nullptr == cusolver_) {
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_));
  }
  return cusolver_;
}

#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
cusolverMpHandle_t CUDALibraries::get_cusolvermp()
{
  if (nullptr == cusolvermp_) {
    int device = -1;
    CUPYNUMERIC_CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUSOLVER(cusolverMpCreate(&cusolvermp_, device, get_cached_stream()));
  }
  return cusolvermp_;
}
#endif

const cutensorHandle_t& CUDALibraries::get_cutensor()
{
  if (!cutensor_.has_value()) {
    CHECK_CUTENSOR(cutensorCreate(&cutensor_.emplace()));
  }
  return *cutensor_;
}

cufftContext CUDALibraries::get_cufft_plan(cufftType type, const cufftPlanParams& params)
{
  auto finder = plan_caches_.find(type);
  cufftPlanCache* cache{nullptr};

  if (plan_caches_.end() == finder) {
    cache              = new cufftPlanCache(type);
    plan_caches_[type] = cache;
  } else {
    cache = finder->second;
  }
  return cufftContext(cache->get_cufft_plan(params));
}

static CUDALibraries& get_cuda_libraries(legate::Processor proc)
{
  if (proc.kind() != legate::Processor::TOC_PROC) {
    LEGATE_ABORT("Illegal request for CUDA libraries for non-GPU processor");
  }

  static CUDALibraries cuda_libraries[LEGION_MAX_NUM_PROCS];
  const auto proc_id = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return cuda_libraries[proc_id];
}

legate::cuda::StreamView get_cached_stream()
{
  return legate::cuda::StreamPool::get_stream_pool().get_stream();
}

cublasContext* get_cublas()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cublas();
}

cusolverDnContext* get_cusolver()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusolver();
}

static CuSolverExtraSymbols& static_cusolver_extra_symbols()
{
  static CuSolverExtraSymbols cusolver_extra_symbols;
  return cusolver_extra_symbols;
}

CuSolverExtraSymbols* get_cusolver_extra_symbols()
{
  auto& symbols = static_cusolver_extra_symbols();
  return &symbols;
}

#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
cusolverMpHandle* get_cusolvermp()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusolvermp();
}
#endif

const cutensorHandle_t& get_cutensor()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cutensor();
}

cufftContext get_cufft_plan(cufftType type, const cufftPlanParams& params)
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cufft_plan(type, params);
}

const cudaDeviceProp& get_device_properties()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_device_properties();
}

int get_device_ordinal()
{
  const auto proc = legate::Processor::get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_device_ordinal();
}

class LoadCUDALibsTask : public CuPyNumericTask<LoadCUDALibsTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_LOAD_CUDALIBS}};

 public:
  static void gpu_variant(legate::TaskContext context)
  {
    const auto proc = legate::Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.get_cublas();
    lib.get_cusolver();
    auto* extra = get_cusolver_extra_symbols();
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
    lib.get_cusolvermp();
#endif
    static_cast<void>(lib.get_cutensor());
  }
};

class UnloadCUDALibsTask : public CuPyNumericTask<UnloadCUDALibsTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_UNLOAD_CUDALIBS}};

 public:
  static void gpu_variant(legate::TaskContext context)
  {
    const auto proc = legate::Processor::get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.finalize();
    auto* extra = get_cusolver_extra_symbols();
    extra->finalize();
    destroy_bitgenerator(proc);
  }
};

const auto reg_ = []() -> char {
  LoadCUDALibsTask::register_variants();
  UnloadCUDALibsTask::register_variants();
  return 0;
}();

}  // namespace cupynumeric

extern "C" {

bool cupynumeric_cusolver_has_geev() { return cupynumeric::get_cusolver_extra_symbols()->has_geev; }
}
