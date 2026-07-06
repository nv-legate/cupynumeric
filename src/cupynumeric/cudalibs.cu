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
#include <realm/cuda/cuda_module.h>

#include <dlfcn.h>
#include <link.h>
#include <stdio.h>

#include <cstring>
#include <string>

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
  static constexpr int32_t MAX_PLANS = 16;

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
  cufftPlan* get_cufft_plan(const cufftPlanParams& params, cudaStream_t stream);

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

cufftPlan* cufftPlanCache::get_cufft_plan(const cufftPlanParams& params, cudaStream_t stream)
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
    CHECK_CUFFT(cufftSetStream(result->handle, stream));
  }
  return result;
}

namespace {

// Return a handle to the libcusolver instance ALREADY resident in this process from the
// link against cuSOLVER -- i.e. the same library that provides cusolverDnCreate(), and hence
// the cuSOLVER handle. We must NOT dlopen the unversioned "libcusolver.so": that name can
// resolve to a DIFFERENT (foreign) libcusolver than the linked one (e.g. /usr/local/cuda/
// lib64/libcusolver.so when no unversioned symlink sits next to the linked
// libcusolver.so.<major>, as in conda envs without libcusolver-dev). Calling geev /
// syevBatched from a different library instance than the handle yields
// CUSOLVER_STATUS_INTERNAL_ERROR. Binding the already-resident instance makes this robust to
// a stray toolkit on the loader path.
[[nodiscard]] void* open_resident_cusolver()
{
  // Locate the resident libcusolver by name (any libcusolver.so.<major>, so there is no
  // per-CUDA-version maintenance) and re-open that exact file with RTLD_NOLOAD, which returns
  // the already-loaded instance rather than loading a new (possibly foreign) copy.
  std::string path;
  dl_iterate_phdr(
    [](struct dl_phdr_info* info, size_t, void* data) -> int {
      if (info->dlpi_name == nullptr) {
        return 0;
      }
      const char* base = std::strrchr(info->dlpi_name, '/');
      base             = (base != nullptr) ? base + 1 : info->dlpi_name;
      // Match "libcusolver.so" / "libcusolver.so.<N>" but NOT "libcusolverMp.so*".
      if (std::strncmp(base, "libcusolver.so", sizeof("libcusolver.so") - 1) == 0) {
        *static_cast<std::string*>(data) = info->dlpi_name;
        return 1;  // found -> stop iterating
      }
      return 0;
    },
    &path);
  if (!path.empty()) {
    if (void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD); handle != nullptr) {
      return handle;
    }
  }
  // Fallback: known versioned SONAMEs, resident-only (RTLD_NOLOAD). We never dlopen without
  // RTLD_NOLOAD: loading a fresh (possibly foreign) libcusolver is exactly the failure this
  // function exists to prevent.
  for (const char* soname : {"libcusolver.so.12", "libcusolver.so.11"}) {
    if (void* handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD); handle != nullptr) {
      return handle;
    }
  }
  // No resident libcusolver. The cuSOLVER handle is created from the linked libcusolver before
  // any extra symbol is needed, so this is unreachable in practice; abort rather than load a new
  // (possibly foreign) instance.
  LEGATE_ABORT("No resident libcusolver found; refusing to dlopen a new copy for extra symbols");
}

}  // namespace

// Bind to the libcusolver already resident in this process (see open_resident_cusolver) and
// resolve the optional "extra" cuSOLVER entry points that are absent from our compile-time
// headers: the expert eigensolver cusolverDnXgeev[_bufferSize] and the batched symmetric
// eigensolver cusolverDnXsyevBatched[_bufferSize]. Each pair is looked up with dlsym; if either
// symbol of a pair is missing (older cuSOLVER), that feature is disabled -- has_geev /
// has_syev_batched stays false -- so callers fall back to the supported path.
CuSolverExtraSymbols::CuSolverExtraSymbols()
  : cusolver_lib(nullptr),
    cusolver_geev_bufferSize(nullptr),
    cusolver_geev(nullptr),
    has_geev(false),
    cusolver_syev_batched_bufferSize(nullptr),
    cusolver_syev_batched(nullptr),
    has_syev_batched(false)
{
  cusolver_lib = open_resident_cusolver();
  {
    void* fn1 = dlsym(cusolver_lib, "cusolverDnXgeev_bufferSize");
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

  {
    void* fn1 = dlsym(cusolver_lib, "cusolverDnXsyevBatched_bufferSize");
    if (fn1 == nullptr) {
      dlerror();
    } else {
      cusolver_syev_batched_bufferSize = (cusolverDnXsyevBatched_bufferSize_handle)fn1;
      has_syev_batched                 = true;
    }

    void* fn2 = dlsym(cusolver_lib, "cusolverDnXsyevBatched");
    if (fn2 == nullptr) {
      has_syev_batched                 = false;
      cusolver_syev_batched_bufferSize = nullptr;
      dlerror();
    } else {
      cusolver_syev_batched = (cusolverDnXsyevBatched_handle)fn2;
    }
  }
}

void CuSolverExtraSymbols::finalize()
{
  cusolver_geev            = nullptr;
  cusolver_geev_bufferSize = nullptr;
  has_geev                 = false;

  cusolver_syev_batched_bufferSize = nullptr;
  cusolver_syev_batched            = nullptr;
  has_syev_batched                 = false;

  if (cusolver_lib != nullptr) {
    dlclose(cusolver_lib);
    cusolver_lib = nullptr;
  }
}

CuSolverExtraSymbols::~CuSolverExtraSymbols() { finalize(); }

CUDALibraries::CUDALibraries()
  : finalized_(false), cublas_(nullptr), cusolver_(nullptr), plan_caches_()
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
  finalize_cusolvermp();
#endif
  if (cutensor_.has_value()) {
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
  for (auto& [_, handle] : cusolvermp_handle_map_) {
    CHECK_CUSOLVER(cusolverMpDestroy(handle));
  }
  cusolvermp_handle_map_.clear();
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
    if (is_fast_math()) {
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
cusolverMpHandle_t CUDALibraries::get_cusolvermp(cudaStream_t stream, int nprow, int npcol)
{
  cusolverMpHandle* cusolvermp_h{nullptr};
  auto key = std::make_pair(nprow, npcol);

  if (auto pos = cusolvermp_handle_map_.find(key);
      pos != cusolvermp_handle_map_.end() && pos->second != nullptr) {
    cusolvermp_h = pos->second;
  } else {
    int device = -1;
    CUPYNUMERIC_CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUSOLVER(cusolverMpCreate(&cusolvermp_h, device, stream));

    cusolvermp_handle_map_.insert(std::make_pair(key, cusolvermp_h));
  }

  return cusolvermp_h;
}
#endif

const cutensorHandle_t& CUDALibraries::get_cutensor()
{
  if (!cutensor_.has_value()) {
    CHECK_CUTENSOR(cutensorCreate(&cutensor_.emplace()));
  }
  return *cutensor_;
}

cufftContext CUDALibraries::get_cufft_plan(cufftType type,
                                           const cufftPlanParams& params,
                                           cudaStream_t stream)
{
  auto finder = plan_caches_.find(type);
  cufftPlanCache* cache{nullptr};

  if (plan_caches_.end() == finder) {
    cache              = new cufftPlanCache(type);
    plan_caches_[type] = cache;
  } else {
    cache = finder->second;
  }
  return cufftContext(cache->get_cufft_plan(params, stream));
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

cublasContext* get_cublas()
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cublas();
}

cusolverDnContext* get_cusolver()
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
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
cusolverMpHandle* get_cusolvermp(cudaStream_t stream, int nprow, int npcol)
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cusolvermp(stream, nprow, npcol);
}
#endif

const cutensorHandle_t& get_cutensor()
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cutensor();
}

cufftContext get_cufft_plan(cufftType type, const cufftPlanParams& params, cudaStream_t stream)
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_cufft_plan(type, params, stream);
}

const cudaDeviceProp& get_device_properties()
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
  auto& lib       = get_cuda_libraries(proc);
  return lib.get_device_properties();
}

int get_device_ordinal()
{
  const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
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
    const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.get_cublas();
    lib.get_cusolver();
    auto* extra = get_cusolver_extra_symbols();
#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
    lib.get_cusolvermp(context.get_task_stream(), 0, 0);
#endif
    static_cast<void>(lib.get_cutensor());
  }
};

class UnloadCUDALibsTask : public CuPyNumericTask<UnloadCUDALibsTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_UNLOAD_CUDALIBS}};

#if LEGATE_DEFINED(CUPYNUMERIC_USE_CUSOLVERMP)
  // cusolverMpDestroy internally calls ncclCommFinalize which is a collective operation.
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);
#endif

 public:
  static void gpu_variant(legate::TaskContext context)
  {
    const auto proc = legate::Runtime::get_runtime()->get_executing_processor();
    auto& lib       = get_cuda_libraries(proc);
    lib.finalize();
    auto* extra = get_cusolver_extra_symbols();
    extra->finalize();
    destroy_bitgenerator(proc);
  }
};

const auto cupynumeric_reg_task_ = []() -> char {
  LoadCUDALibsTask::register_variants();
  UnloadCUDALibsTask::register_variants();
  return 0;
}();

}  // namespace cupynumeric

extern "C" {

bool cupynumeric_cusolver_has_geev() { return cupynumeric::get_cusolver_extra_symbols()->has_geev; }
}
