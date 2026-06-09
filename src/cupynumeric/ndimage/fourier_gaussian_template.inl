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

#include "cupynumeric/ndimage/fourier_gaussian.h"

#include <cuda/std/cmath>

#include <cassert>
#include <cmath>

// SciPy compatibility formula implemented (where?):
// factor *= exp(-2 * (sigma[dim] * pi / shape)^2 * k^2)
// with:
// shape = (n >= 0 && dim == axis) ? n : extent;
// k     = real_fft_axis ? index : wrapped_fft_frequency(index, extent);

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct NdimageFourierGaussianImplBody;

template <int DIM>
LEGATE_HOST_DEVICE inline double fft_frequency(const Point<DIM>& p, int dim, int64_t extent)
{
  // Convert tile point to global array index. If rect.lo is 0 this is a no-op;
  // if a partition is offset, this avoids using an absolute Legion coordinate
  // as though it were a zero-based FFT bin.
  const int64_t idx = p[dim];

  const int64_t split = (extent + 1) / 2;
  return idx < split ? static_cast<double>(idx) : static_cast<double>(idx - extent);
}

template <int DIM>
LEGATE_HOST_DEVICE inline double fourier_gaussian_factor(const Point<DIM>& p,
                                                         const Rect<DIM>& rect,
                                                         const NdimageFourierGaussianParams params)
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

    const double q   = params.sigmas[dim] * M_PI / shape;
    const double tmp = -2.0 * q * q * k * k;

    factor *= (-tmp) > 50.0 ? 0.0 : cuda::std::exp(tmp);
  }

  return factor;
}

template <VariantKind KIND>
struct NdimageFourierGaussian {
  TaskContext context;

  explicit NdimageFourierGaussian(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM>
  void operator()(NdimageFourierGaussianArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto out_rect = args.output.shape<DIM>();
    if (out_rect.empty()) {
      return;
    }

    auto output = args.output.write_accessor<VAL, DIM>(out_rect);
    auto input  = args.input.read_accessor<VAL, DIM>(out_rect);

    NdimageFourierGaussianImplBody<KIND, VAL, DIM>{context}(output, input, out_rect, args.params);
  }
};

template <VariantKind KIND>
static void ndimage_fourier_gaussian_template(TaskContext& context)
{
  const int32_t ndim = context.output(0).dim();

  NdimageFourierGaussianArgs args{};
  args.output = context.output(0);
  args.input  = context.input(0);

  args.params.n    = context.scalar(0).value<int64_t>();
  args.params.axis = context.scalar(1).value<int32_t>();

  auto extents = context.scalar(2).values<int64_t>();
  auto sigmas  = context.scalar(3).values<double>();

  assert(extents.size() == ndim);
  assert(sigmas.size() == ndim);

  for (int32_t dim = 0; dim < ndim; ++dim) {
    args.params.extents[dim] = extents[dim];
    args.params.sigmas[dim]  = sigmas[dim];
  }

  double_dispatch(
    args.output.dim(), args.output.code(), NdimageFourierGaussian<KIND>{context}, args);
}

}  // namespace cupynumeric
