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

#include "generator.cuh"
#include "generator_create.inl"

extern "C" curandStatus_t randutilCreateGenerator(randutilGenerator_t* generator,
                                                  curandRngType_t rng_type,
                                                  uint64_t seed,
                                                  uint64_t generatorID,
                                                  uint32_t flags,
                                                  cudaStream_t stream)
{
  return inner_randutilCreateGenerator<randutilimpl::execlocation::DEVICE>(
    generator, rng_type, seed, generatorID, stream);
}

// explicit instantiations of distributions
#include "generator_integers.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, integers<int16_t>, int16_t>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, integers<int32_t>, int32_t>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, integers<int64_t>, int64_t>;

#include "generator_lognormal.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, lognormal_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, lognormal_t<double>, double>;

#include "generator_normal.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, normal_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, normal_t<double>, double>;

#include "generator_poisson.inl"
template struct randutilimpl::dispatcher<randutilimpl::execlocation::DEVICE, poisson, uint32_t>;

#include "generator_raw.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, raw<uint32_t>, uint32_t>;

#include "generator_uniform.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, uniform_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, uniform_t<double>, double>;
