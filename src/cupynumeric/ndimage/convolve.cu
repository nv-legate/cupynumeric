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

#include "cupynumeric/ndimage/convolve.h"
#include "cupynumeric/ndimage/convolve_template.inl"

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"
#include "cupynumeric/cupynumeric_c.h"

#include <cstdint>

namespace cupynumeric {

using namespace legate;

static __device__ int32_t get_boundary_index(const int32_t idx,
                                             const int32_t size,
                                             const CuPyNumericNdimageConvolveMode mode,
                                             bool* use_constant)
{
  const bool has_input_value = 0 <= idx && idx < size;
  if (has_input_value) {
    return idx;
  }

  switch (mode) {
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_REFLECT: {
      int32_t out = idx;

      out = max(out, -1 - out);
      out %= size * 2;
      out = min(out, 2 * size - 1 - out);
      return out;
    }
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_MIRROR: {
      if (size == 1) {
        return 0;
      }
      int32_t out = idx;

      out = max(out, -out);
      out %= 2 * size - 2;
      out = min(out, 2 * size - 2 - out);
      return out;
    }
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_NEAREST: return min(max(idx, int32_t{0}), size - 1);
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_WRAP: {
      const int32_t out = idx % size;
      return out < 0 ? out + size : out;
    }
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_CONSTANT: {
      *use_constant = true;
      return -1;
    }
    default: {
      LEGATE_ABORT("Invalid convolution mode, %d", mode);
      return idx;
    }
  }
}

template <int DIM_IDX, int DIM, typename VAL>
struct FilterLoop {
  static __device__ void run(const Point<DIM> output_point,
                             const Rect<DIM> output_rect,
                             const VAL* input,
                             const Point<DIM> input_strides,
                             const Rect<DIM> input_rect,
                             const VAL* weights_base,
                             const Point<DIM> weights_strides,
                             const Rect<DIM> weights_rect,
                             const CuPyNumericNdimageConvolveMode mode,
                             const VAL cval,
                             const Point<DIM> origins,
                             const bool use_cval,
                             const bool correlate,
                             VAL& acc)
  {
    const int32_t w_size_dim     = weights_rect.hi[DIM_IDX] - weights_rect.lo[DIM_IDX] + 1;
    const int32_t input_size_dim = input_rect.hi[DIM_IDX] - input_rect.lo[DIM_IDX] + 1;

    const VAL* weights_ptr =
      correlate ? weights_base : weights_base + (w_size_dim - 1) * weights_strides[DIM_IDX];
    int32_t traversal_dir = correlate ? 1 : -1;

    const int32_t center_dim   = w_size_dim / 2 + origins[DIM_IDX];
    const int32_t input_offset = output_point[DIM_IDX] - input_rect.lo[DIM_IDX];
    for (int32_t iw = 0, curr_idx_dim = input_offset + center_dim - w_size_dim + 1; iw < w_size_dim;
         iw++, curr_idx_dim++, weights_ptr += traversal_dir * weights_strides[DIM_IDX]) {
      bool iter_use_constant = false;
      const int32_t actual_idx_dim =
        get_boundary_index(curr_idx_dim, input_size_dim, mode, &iter_use_constant);

      const bool next_use_cval = use_cval || iter_use_constant;
      const VAL* next_input =
        next_use_cval ? input : (input + input_strides[DIM_IDX] * actual_idx_dim);

      FilterLoop<DIM_IDX + 1, DIM, VAL>::run(output_point,
                                             output_rect,
                                             next_input,
                                             input_strides,
                                             input_rect,
                                             weights_ptr,
                                             weights_strides,
                                             weights_rect,
                                             mode,
                                             cval,
                                             origins,
                                             next_use_cval,
                                             correlate,
                                             acc);
    }
  }
};

template <int DIM, typename VAL>
struct FilterLoop<DIM, DIM, VAL> {
  static __device__ void run(const Point<DIM>,
                             const Rect<DIM>,
                             const VAL* input,
                             const Point<DIM>,
                             const Rect<DIM>,
                             const VAL* weights,
                             const Point<DIM>,
                             const Rect<DIM>,
                             const CuPyNumericNdimageConvolveMode,
                             const VAL cval,
                             const Point<DIM>,
                             const bool use_cval,
                             const bool,
                             VAL& acc)
  {
    const VAL input_value = use_cval ? cval : (*input);
    acc += input_value * (*weights);
  }
};

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  batched_convolve_kernel(VAL* out,
                          const Point<DIM> output_strides,
                          const Rect<DIM> output_rect,
                          const Pitches<DIM - 1> out_pitches,
                          const size_t output_volume,
                          const Point<DIM> output_lo,
                          const VAL* input,
                          const Point<DIM> input_strides,
                          const Rect<DIM> input_rect,
                          const VAL* weights,
                          const Point<DIM> weights_strides,
                          const Rect<DIM> weights_rect,
                          const CuPyNumericNdimageConvolveMode mode,
                          const VAL cval,
                          const Point<DIM> origins,
                          const bool use_cval,
                          const bool correlate)
{
  const size_t idx = global_tid_1d();

  if (idx >= output_volume) {
    return;
  }

  // determine the output point that this thread is responsible for
  auto output_point = out_pitches.unflatten(idx, output_lo);

  // Dimension idx 1 of input and dimension idx 0 of weights are broadcasted
  // so we don't need to index them here.
  const VAL* input_base   = input + input_strides[0] * (output_point[0] - output_rect.lo[0]);
  const VAL* weights_base = weights + weights_strides[1] * (output_point[1] - weights_rect.lo[1]);

  VAL acc = 0;

  FilterLoop<2, DIM, VAL>::run(output_point,
                               output_rect,
                               input_base,
                               input_strides,
                               input_rect,
                               weights_base,
                               weights_strides,
                               weights_rect,
                               mode,
                               cval,
                               origins,
                               false,
                               correlate,
                               acc);

  *(out + (output_point - output_rect.lo).dot(output_strides)) = acc;
}

template <typename VAL, int DIM>
struct NdimageConvolveImplBody<VariantKind::GPU, VAL, DIM, false> {
  TaskContext context;
  explicit NdimageConvolveImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> input,
                  AccessorRO<VAL, DIM> weights,
                  const Rect<DIM>& input_rect,
                  const Rect<DIM>& output_rect,
                  const Rect<DIM>& weights_rect,
                  CuPyNumericNdimageConvolveMode mode,
                  VAL cval,
                  Point<DIM> origins) const
  {
    // Note: in order to re-use the batched kernel and reduce binary size,
    // we will add 2 fake dimensions to each of the input, weight, and output arrays.
    auto expand_rect = [](const Rect<DIM>& rect) -> Rect<DIM + 2> {
      Rect<DIM + 2> expanded;

      expanded.lo[0] = 0;
      expanded.hi[0] = 0;
      expanded.lo[1] = 0;
      expanded.hi[1] = 0;
      for (int dim = 0; dim < DIM; ++dim) {
        expanded.lo[dim + 2] = rect.lo[dim];
        expanded.hi[dim + 2] = rect.hi[dim];
      }
      return expanded;
    };

    // Get the input pointer with respect to the output rect because
    // we would like to get the pointer to be aligned to the output rect.
    // Offset indexing in convolve_kernel will handle indexing into the bloated regions.
    size_t input_strides[DIM + 2];
    input_strides[0]     = 0;
    input_strides[1]     = 0;
    const auto input_ptr = input.ptr(input_rect, input_strides + 2);

    size_t weights_strides[DIM + 2];
    weights_strides[0]     = 0;
    weights_strides[1]     = 0;
    const auto weights_ptr = weights.ptr(weights_rect, weights_strides + 2);

    size_t output_strides[DIM + 2];
    output_strides[0]     = 0;
    output_strides[1]     = 0;
    const auto output_ptr = out.ptr(output_rect, output_strides + 2);

    Rect<DIM + 2> expanded_input_rect   = expand_rect(input_rect);
    Rect<DIM + 2> expanded_weights_rect = expand_rect(weights_rect);
    Rect<DIM + 2> expanded_output_rect  = expand_rect(output_rect);

    Pitches<DIM + 1> output_pitches;
    const size_t output_volume = output_pitches.flatten(expanded_output_rect);

    const size_t num_blocks = (output_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const auto stream       = context.get_task_stream();

    // Expand out the origins point to include 2 dummy batch dimensions
    Point<DIM + 2> expanded_origins;

    expanded_origins[0] = 0;
    expanded_origins[1] = 0;
    for (int dim = 2; dim < DIM + 2; ++dim) {
      expanded_origins[dim] = origins[dim - 2];
    }

    batched_convolve_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      output_ptr,
      Point<DIM + 2>(output_strides),
      expanded_output_rect,
      output_pitches,
      output_volume,
      expanded_output_rect.lo,
      input_ptr,
      Point<DIM + 2>(input_strides),
      expanded_input_rect,
      weights_ptr,
      Point<DIM + 2>(weights_strides),
      expanded_weights_rect,
      mode,
      cval,
      expanded_origins,
      false,
      false);
  }
};

template <typename VAL, int DIM>
struct NdimageConvolveImplBody<VariantKind::GPU, VAL, DIM, true> {
  static_assert(DIM > 2, "ndimage.batched_convolve requires DIM > 2");

  TaskContext context;
  explicit NdimageConvolveImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> input,
                  AccessorRO<VAL, DIM> weights,
                  const Rect<DIM>& input_rect,
                  const Rect<DIM>& output_rect,
                  const Rect<DIM>& weights_rect,
                  CuPyNumericNdimageConvolveMode mode,
                  VAL cval,
                  Point<DIM> origins) const
  {
    size_t input_strides[DIM];
    const auto input_ptr = input.ptr(input_rect, input_strides);

    size_t weights_strides[DIM];
    const auto weights_ptr = weights.ptr(weights_rect, weights_strides);

    size_t output_strides[DIM];
    const auto output_ptr = out.ptr(output_rect, output_strides);

    Pitches<DIM - 1> output_pitches;
    const size_t output_volume = output_pitches.flatten(output_rect);

    const size_t num_blocks = (output_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const auto stream       = context.get_task_stream();

    batched_convolve_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      output_ptr,
      Point<DIM>(output_strides),
      output_rect,
      output_pitches,
      output_volume,
      output_rect.lo,
      input_ptr,
      Point<DIM>(input_strides),
      input_rect,
      weights_ptr,
      Point<DIM>(weights_strides),
      weights_rect,
      mode,
      cval,
      origins,
      false,
      false);
  }
};

/*static*/ void NdimageConvolveTask::gpu_variant(TaskContext context)
{
  ndimage_convolve_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
