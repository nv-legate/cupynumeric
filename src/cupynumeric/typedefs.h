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

#include <memory>

#include "legate.h"

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

namespace cupynumeric {

using Array  = legate::PhysicalStore;
using Scalar = legate::Scalar;

}  // namespace cupynumeric
