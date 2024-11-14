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

#include "cupynumeric/random/bitgenerator.h"
#include "cupynumeric/random/bitgenerator_template.inl"
#include "cupynumeric/random/bitgenerator_util.h"

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/random/curand_help.h"

#include "cupynumeric/random/bitgenerator_curand.inl"

#include <string_view>
#include <mutex>

namespace cupynumeric {

using namespace legate;

// required by CHECK_CURAND_DEVICE:
//
void randutil_check_curand_device(curandStatus_t error, std::string_view file, int line)
{
  if (error != CURAND_STATUS_SUCCESS) {
    randutil_log().fatal() << "Internal CURAND failure with error " << (int)error << " in file "
                           << file << " at line " << line;
    assert(false);
  }
}

struct GPUGenerator : public CURANDGenerator {
  cudaStream_t stream_;
  GPUGenerator(BitGeneratorType gentype, uint64_t seed, uint64_t generatorId, uint32_t flags)
    : CURANDGenerator(gentype, seed, generatorId)
  {
    CUPYNUMERIC_CHECK_CUDA(::cudaStreamCreate(&stream_));
    CHECK_CURAND(::randutilCreateGenerator(&gen_, type_, seed, generatorId, flags, stream_));
  }

  virtual ~GPUGenerator()
  {
    CUPYNUMERIC_CHECK_CUDA(::cudaStreamSynchronize(stream_));
    CHECK_CURAND(::randutilDestroyGenerator(gen_));
  }
};

template <>
struct CURANDGeneratorBuilder<VariantKind::GPU> {
  static CURANDGenerator* build(BitGeneratorType gentype,
                                uint64_t seed,
                                uint64_t generatorId,
                                uint32_t flags)
  {
    return new GPUGenerator(gentype, seed, generatorId, flags);
  }

  static void destroy(CURANDGenerator* cugenptr) { delete cugenptr; }
};

template <>
std::map<legate::Processor, std::unique_ptr<generator_map<VariantKind::GPU>>>
  BitGeneratorImplBody<VariantKind::GPU>::m_generators = {};

template <>
std::mutex BitGeneratorImplBody<VariantKind::GPU>::lock_generators = {};

/*static*/ void BitGeneratorTask::gpu_variant(legate::TaskContext context)
{
  bitgenerator_template<VariantKind::GPU>(context);
}

void destroy_bitgenerator(const legate::Processor& proc)
{
  BitGeneratorImplBody<VariantKind::GPU>::destroy_bitgenerator(proc);
}

}  // namespace cupynumeric
