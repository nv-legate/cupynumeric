/* Copyright 2026 NVIDIA Corporation
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

#include "cupynumeric/cupynumeric_task.h"

namespace cupynumeric {

using namespace legate;

enum class FourierFilterType : int32_t { Gaussian = 1, Uniform = 2, Shift = 3, Ellipsoid = 4 };

struct NdimageFourierFilterParams {
  int64_t n{-1};
  int32_t axis{-1};
  int32_t filter_type{0};
  int64_t extents[LEGATE_MAX_DIM]{};
  double sigmas[LEGATE_MAX_DIM]{};
};

struct NdimageFourierFilterArgs {
  legate::PhysicalStore output{nullptr};
  legate::PhysicalStore input{nullptr};
  NdimageFourierFilterParams params;
};

class NdimageFourierFilterTask : public CuPyNumericTask<NdimageFourierFilterTask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_NDIMAGE_FOURIER_FILTER}};

 public:
  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

template <int DIM>
LEGATE_HOST_DEVICE inline double fft_frequency(const Point<DIM>& p, int dim, int64_t extent)
{
  const int64_t idx = p[dim];

  const int64_t split = (extent + 1) / 2;
  return idx < split ? static_cast<double>(idx) : static_cast<double>(idx - extent);
}

template <typename FactorF, int DIM>
LEGATE_HOST_DEVICE inline double fourier_factor_dispatcher(const Point<DIM>& p,
                                                           const Rect<DIM>& rect,
                                                           const NdimageFourierFilterParams& params,
                                                           FactorF factor_func)
{
  double factor = 1.0;

  for (int dim = 0; dim < DIM; ++dim) {
    const int64_t extent = params.extents[dim];
    if (extent <= 1) {
      continue;
    }

    const bool real_fft_axis = params.n >= 0 && dim == params.axis;
    const double shape =
      real_fft_axis ? static_cast<double>(params.n) : static_cast<double>(extent);

    const double k =
      real_fft_axis ? static_cast<double>(p[dim]) : fft_frequency<DIM>(p, dim, extent);

    auto iter_factor = factor_func(params, dim, k, shape);

    factor *= iter_factor;
  }

  return factor;
}
}  // namespace cupynumeric
