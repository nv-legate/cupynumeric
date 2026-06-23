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

#include "cupynumeric/ndimage/fourier_filter.h"

#include <cuda/std/cmath>

#include <cassert>
#include <cmath>
#include <cstdint>

// SciPy compatibility formula implemented (where?):
// factor *= exp(-2 * (sigma[dim] * pi / shape)^2 * k^2)
// with:
// shape = (n >= 0 && dim == axis) ? n : extent;
// k     = real_fft_axis ? index : wrapped_fft_frequency(index, extent);

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, typename VAL, int DIM>
struct NdimageFourierFilterImplBody;

template <int DIM>
LEGATE_HOST_DEVICE inline double fourier_filter_factor(const Point<DIM>& p,
                                                       const Rect<DIM>& rect,
                                                       const NdimageFourierFilterParams params)
{
  FourierFilterType filter_tid = static_cast<FourierFilterType>(params.filter_type);

  switch (filter_tid) {
    case FourierFilterType::Gaussian: {
      auto factor_func =
        [](const NdimageFourierFilterParams& params, int dim, double k, double shape) -> double {
        const double q   = params.sigmas[dim] * M_PI / shape;
        const double tmp = -2.0 * q * q * k * k;
        return (-tmp) > 50.0 ? 0.0 : cuda::std::exp(tmp);
      };

      return fourier_factor_dispatcher(p, rect, params, factor_func);
    }
    case FourierFilterType::Uniform: {
      auto factor_func =
        [](const NdimageFourierFilterParams& params, int dim, double k, double shape) -> double {
        const double q = M_PI * params.sigmas[dim] / shape;  // `sigmas` interpreted as `sizes`

        double axis_factor = 1.0;
        if (k != 0.0) {
          const double tmp = q * k;
          axis_factor      = q > 0.0 ? cuda::std::sin(tmp) / tmp : 0.0;
        }

        return axis_factor;
      };

      return fourier_factor_dispatcher(p, rect, params, factor_func);
    }
    case FourierFilterType::Shift: {
      // cannot throw from device code...
      //
      LEGATE_ABORT("Not yet implemented.");
    }
    case FourierFilterType::Ellipsoid: {
      // cannot throw from device code...
      //
      LEGATE_ABORT("Not yet implemented.");
    }
    default: {
      // cannot throw from device code...
      //
      LEGATE_ABORT("Not yet implemented.");
    }
  }
}

template <Type::Code CODE>
struct _fourier_supported {
  static constexpr bool value = CODE == Type::Code::FLOAT64 || CODE == Type::Code::FLOAT32 ||
                                CODE == Type::Code::COMPLEX64 || CODE == Type::Code::COMPLEX128;
};

template <VariantKind KIND>
struct NdimageFourierFilter {
  TaskContext context;

  explicit NdimageFourierFilter(TaskContext context) : context(context) {}

  template <Type::Code CODE, int DIM, std::enable_if_t<_fourier_supported<CODE>::value>* = nullptr>
  void operator()(NdimageFourierFilterArgs& args) const
  {
    using VAL = type_of<CODE>;

    auto out_rect = args.output.shape<DIM>();
    if (out_rect.empty()) {
      return;
    }

    auto output = args.output.write_accessor<VAL, DIM>(out_rect);
    auto input  = args.input.read_accessor<VAL, DIM>(out_rect);

    NdimageFourierFilterImplBody<KIND, VAL, DIM>{context}(output, input, out_rect, args.params);
  }

  template <Type::Code CODE, int DIM, std::enable_if_t<!_fourier_supported<CODE>::value>* = nullptr>
  void operator()(NdimageFourierFilterArgs& args) const
  {
    throw legate::TaskException("Non floating-point / complex types are not supported.");
  }
};

#ifdef DEBUG_CUPYNUMERIC
namespace {
template <FourierFilterType... filter_ids>
inline bool any_of(FourierFilterType filter_tid,
                   std::integer_sequence<FourierFilterType, filter_ids...>)
{
  return (... || (filter_tid == filter_ids));
}
}  // namespace
#endif

template <VariantKind KIND>
static void ndimage_fourier_filter_template(TaskContext& context)
{
  const int32_t ndim = context.output(0).dim();

  NdimageFourierFilterArgs args{};
  args.output = context.output(0);
  args.input  = context.input(0);

  args.params.n           = context.scalar(0).value<int64_t>();
  args.params.axis        = context.scalar(1).value<int32_t>();
  args.params.filter_type = context.scalar(2).value<int32_t>();

  auto extents = context.scalar(3).values<int64_t>();
  auto sigmas  = context.scalar(4).values<double>();

#ifdef DEBUG_CUPYNUMERIC
  assert(any_of(static_cast<FourierFilterType>(args.params.filter_type),
                std::integer_sequence<FourierFilterType,
                                      FourierFilterType::Gaussian,
                                      FourierFilterType::Uniform,
                                      FourierFilterType::Shift,
                                      FourierFilterType::Ellipsoid>{}));
#endif

  assert(extents.size() == ndim);
  assert(sigmas.size() == ndim);

  for (int32_t dim = 0; dim < ndim; ++dim) {
    args.params.extents[dim] = extents[dim];
    args.params.sigmas[dim]  = sigmas[dim];
  }

  double_dispatch(args.output.dim(), args.output.code(), NdimageFourierFilter<KIND>{context}, args);
}

}  // namespace cupynumeric
