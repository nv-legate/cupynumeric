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

#include "cupynumeric/ndimage/fourier_gaussian.h"
#include "cupynumeric/ndimage/fourier_gaussian_template.inl"

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"

#include <cmath>
#include <cstdint>

namespace cupynumeric {

using namespace legate;

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  fourier_gaussian_kernel(VAL* output,
                          const Point<DIM> output_strides,
                          const VAL* input,
                          const Point<DIM> input_strides,
                          const Rect<DIM> rect,
                          const Pitches<DIM - 1> pitches,
                          const size_t volume,
                          const Point<DIM> lo,
                          const NdimageFourierGaussianParams params)
{
  const size_t idx = global_tid_1d();

  if (idx >= volume) {
    return;
  }

  const Point<DIM> p                            = pitches.unflatten(idx, lo);
  const double factor                           = fourier_gaussian_factor<DIM>(p, rect, params);
  const VAL in                                  = *(input + (p - rect.lo).dot(input_strides));
  *(output + (p - rect.lo).dot(output_strides)) = in * VAL{factor};
}

template <typename VAL, int DIM>
struct NdimageFourierGaussianImplBody<VariantKind::GPU, VAL, DIM> {
  TaskContext context;

  explicit NdimageFourierGaussianImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> output,
                  AccessorRO<VAL, DIM> input,
                  const Rect<DIM>& rect,
                  const NdimageFourierGaussianParams params) const
  {
    size_t input_strides[DIM];
    const auto input_ptr = input.ptr(rect, input_strides);

    size_t output_strides[DIM];
    const auto output_ptr = output.ptr(rect, output_strides);

    Pitches<DIM - 1> pitches;
    const size_t volume = pitches.flatten(rect);

    const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    const auto stream = context.get_task_stream();

    fourier_gaussian_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      output_ptr,
      Point<DIM>(output_strides),
      input_ptr,
      Point<DIM>(input_strides),
      rect,
      pitches,
      volume,
      rect.lo,
      params);
  }
};

/*static*/ void NdimageFourierGaussianTask::gpu_variant(TaskContext context)
{
  ndimage_fourier_gaussian_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
