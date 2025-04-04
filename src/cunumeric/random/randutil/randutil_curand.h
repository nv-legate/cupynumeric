/* Copyright 2021-2022 NVIDIA Corporation
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

// This header file imports some head-only part of curand for the HOST-side implementation of
// generators

// also allow usage of generators on host
#ifdef LEGATE_USE_CUDA

#define QUALIFIERS static __forceinline__ __device__ __host__
#define RANDUTIL_QUALIFIERS __forceinline__ __device__ __host__
#include <curand_kernel.h>

#else
// host generators are not compiled with nvcc
#define QUALIFIERS static
#define RANDUTIL_QUALIFIERS
#ifdef USE_STL_RANDOM_ENGINE_
#include <random>
#else
#include <curand_kernel.h>
#endif

#endif
