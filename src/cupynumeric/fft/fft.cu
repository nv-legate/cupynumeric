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
#include <csignal>

#include "cupynumeric/fft/fft.h"
#include "cupynumeric/fft/fft_template.inl"

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;
using dim_t = long long int32_t;

template <int32_t DIM, typename TYPE>
__global__ static void copy_kernel(
  size_t volume, TYPE* target, AccessorRO<TYPE, DIM> acc, Pitches<DIM - 1> pitches, Point<DIM> lo)
{
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= volume) {
    return;
  }
  auto p         = pitches.unflatten(offset, Point<DIM>::ZEROES());
  target[offset] = acc[p + lo];
}

template <int32_t DIM, typename TYPE>
__host__ static inline void copy_into_buffer(TYPE* target,
                                             AccessorRO<TYPE, DIM>& acc,
                                             const Rect<DIM>& rect,
                                             size_t volume,
                                             cudaStream_t stream)
{
  if (acc.accessor.is_dense_row_major(rect)) {
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      target, acc.ptr(rect.lo), volume * sizeof(TYPE), cudaMemcpyDeviceToDevice, stream));
  } else {
    Pitches<DIM - 1> pitches{};
    pitches.flatten(rect);

    const size_t num_blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copy_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      volume, target, acc, pitches, rect.lo);

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
}

// perform FFT with single optimized cufft operation
// only available for axes = range(DIM) and DIM <=3
template <int32_t DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_operation(AccessorWO<OUTPUT_TYPE, DIM> out,
                                            AccessorRO<INPUT_TYPE, DIM> in,
                                            const Rect<DIM>& out_rect,
                                            const Rect<DIM>& in_rect,
                                            std::vector<int64_t>& axes,
                                            CuPyNumericFFTType type,
                                            CuPyNumericFFTDirection direction,
                                            cudaStream_t stream)
{
  size_t num_elements;
  dim_t n[DIM];
  dim_t inembed[DIM];
  dim_t onembed[DIM];

  const Point<DIM> zero   = Point<DIM>::ZEROES();
  const Point<DIM> one    = Point<DIM>::ONES();
  Point<DIM> fft_size_in  = in_rect.hi - in_rect.lo + one;
  Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
  num_elements            = 1;
  for (int32_t i = 0; i < DIM; ++i) {
    n[i]       = (type == CUPYNUMERIC_FFT_R2C || type == CUPYNUMERIC_FFT_D2Z) ? fft_size_in[i]
                                                                              : fft_size_out[i];
    inembed[i] = fft_size_in[i];
    onembed[i] = fft_size_out[i];
    num_elements *= n[i];
  }

  // get plan from cache
  auto cufft_context = get_cufft_plan(
    (cufftType)type, cufftPlanParams(DIM, n, inembed, 1, 1, onembed, 1, 1, 1), stream);

  if (cufft_context.workareaSize() > 0) {
    auto workarea_buffer =
      create_buffer<uint8_t>(cufft_context.workareaSize(), Memory::Kind::GPU_FB_MEM);
    CHECK_CUFFT(cufftSetWorkArea(cufft_context.handle(), workarea_buffer.ptr(0)));
  }

  const void* in_ptr{nullptr};
  if (in.accessor.is_dense_row_major(in_rect)) {
    in_ptr = in.ptr(in_rect.lo);
  } else {
    auto buffer = create_buffer<INPUT_TYPE, DIM>(fft_size_in, Memory::Kind::GPU_FB_MEM);
    in_ptr      = buffer.ptr(zero);
    copy_into_buffer((INPUT_TYPE*)in_ptr, in, in_rect, in_rect.volume(), stream);
  }
  // FFT the input data
  CHECK_CUFFT(cufftXtExec(cufft_context.handle(),
                          const_cast<void*>(in_ptr),
                          static_cast<void*>(out.ptr(out_rect.lo)),
                          static_cast<int32_t>(direction)));
  // synchronize before cufft_context runs out of scope
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
}

// Perform the FFT operation as multiple 1D FFTs along the specified axes (Complex-to-complex case).
template <int32_t DIM, typename INOUT_TYPE>
__host__ static inline void cufft_over_axes_c2c(INOUT_TYPE* out,
                                                const INOUT_TYPE* in,
                                                const Rect<DIM>& inout_rect,
                                                std::vector<int64_t>& axes,
                                                CuPyNumericFFTType type,
                                                CuPyNumericFFTDirection direction,
                                                cudaStream_t stream)
{
  dim_t n[DIM];

  // Full volume dimensions / strides
  const Point<DIM> one = Point<DIM>::ONES();

  Point<DIM> fft_size = inout_rect.hi - inout_rect.lo + one;
  size_t num_elements = 1;
  for (int32_t i = 0; i < DIM; ++i) {
    n[i] = fft_size[i];
    num_elements *= fft_size[i];
  }

  // Copy input to output buffer (if needed)
  // the computation will be done inplace of the target
  if (in != out) {
    CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(
      out, in, num_elements * sizeof(INOUT_TYPE), cudaMemcpyDeviceToDevice, stream));
  }

  Buffer<uint8_t> workarea_buffer;
  size_t last_workarea_size = 0;
  for (auto& axis : axes) {
    // Single axis dimensions / strides
    dim_t size_1d = n[axis];

    // Extract number of slices and batches per slice
    int64_t num_slices = 1;
    if (axis != DIM - 1) {
      for (int32_t i = 0; i < axis; ++i) {
        num_slices *= n[i];
      }
    }
    dim_t batches  = num_elements / (num_slices * size_1d);
    int64_t offset = batches * size_1d;

    dim_t stride = 1;
    for (int32_t i = axis + 1; i < DIM; ++i) {
      stride *= fft_size[i];
    }
    dim_t dist = (axis == DIM - 1) ? size_1d : 1;

    // get plan from cache
    auto cufft_context =
      get_cufft_plan((cufftType)type,
                     cufftPlanParams(1, &size_1d, n, stride, dist, n, stride, dist, batches),
                     stream);

    if (cufft_context.workareaSize() > 0) {
      if (cufft_context.workareaSize() > last_workarea_size) {
        if (last_workarea_size > 0) {
          workarea_buffer.destroy();
        }
        workarea_buffer =
          create_buffer<uint8_t>(cufft_context.workareaSize(), Memory::Kind::GPU_FB_MEM);
        last_workarea_size = cufft_context.workareaSize();
      }
      CHECK_CUFFT(cufftSetWorkArea(cufft_context.handle(), workarea_buffer.ptr(0)));
    }

    for (uint32_t n = 0; n < num_slices; ++n) {
      CHECK_CUFFT(cufftXtExec(cufft_context.handle(),
                              static_cast<void*>(out + (n * offset)),
                              static_cast<void*>(out + (n * offset)),
                              static_cast<int32_t>(direction)));
    }
    // synchronize before cufft_context runs out of scope
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
  }
}

// Perform the single 1D R2C/C2R FFT along the specified axis
template <int32_t DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_r2c_c2r(OUTPUT_TYPE* out,
                                          INPUT_TYPE* in,  // might be modified!
                                          const Rect<DIM>& out_rect,
                                          const Rect<DIM>& in_rect,
                                          const int64_t axis,
                                          CuPyNumericFFTType type,
                                          CuPyNumericFFTDirection direction,
                                          cudaStream_t stream)
{
  dim_t n[DIM];
  dim_t inembed[DIM];
  dim_t onembed[DIM];

  // Full volume dimensions / strides
  const Point<DIM> one    = Point<DIM>::ONES();
  Point<DIM> fft_size_in  = in_rect.hi - in_rect.lo + one;
  Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
  size_t num_elements_in  = 1;
  size_t num_elements_out = 1;
  for (int32_t i = 0; i < DIM; ++i) {
    n[i]       = (direction == CUPYNUMERIC_FFT_FORWARD) ? fft_size_in[i] : fft_size_out[i];
    inembed[i] = fft_size_in[i];
    onembed[i] = fft_size_out[i];
    num_elements_in *= fft_size_in[i];
    num_elements_out *= fft_size_out[i];
  }

  // Batched 1D dimension
  dim_t size_1d = n[axis];

  // Extract number of slices and batches per slice
  int64_t num_slices = 1;
  if (axis != DIM - 1) {
    for (int32_t i = 0; i < axis; ++i) {
      num_slices *= n[i];
    }
  }
  dim_t batches = ((direction == CUPYNUMERIC_FFT_FORWARD) ? num_elements_in : num_elements_out) /
                  (num_slices * size_1d);
  int64_t offset_in  = num_elements_in / num_slices;
  int64_t offset_out = num_elements_out / num_slices;

  dim_t istride = 1;
  dim_t ostride = 1;
  for (int32_t i = axis + 1; i < DIM; ++i) {
    istride *= fft_size_in[i];
    ostride *= fft_size_out[i];
  }
  dim_t idist = (axis == DIM - 1) ? fft_size_in[axis] : 1;
  dim_t odist = (axis == DIM - 1) ? fft_size_out[axis] : 1;

  // get plan from cache
  auto cufft_context = get_cufft_plan(
    (cufftType)type,
    cufftPlanParams(1, &size_1d, inembed, istride, idist, onembed, ostride, odist, batches),
    stream);

  if (cufft_context.workareaSize() > 0) {
    auto workarea_buffer =
      create_buffer<uint8_t>(cufft_context.workareaSize(), Memory::Kind::GPU_FB_MEM);
    CHECK_CUFFT(cufftSetWorkArea(cufft_context.handle(), workarea_buffer.ptr(0)));
  }

  for (uint32_t n = 0; n < num_slices; ++n) {
    CHECK_CUFFT(cufftXtExec(cufft_context.handle(),
                            static_cast<void*>(in + (n * offset_in)),
                            static_cast<void*>(out + (n * offset_out)),
                            static_cast<int32_t>(direction)));
  }
  // synchronize before cufft_context runs out of scope
  CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
}

// Perform the FFT operation as multiple 1D FFTs along the specified axes.
// C2C - batch process all axes one after another
// R2C - pre-process R2C along the LAST axis, follow up by C2C on remaining axes
// C2R - run C2C on all but last axis, post-process with C2R along the LAST axis
template <int32_t DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_over_axes(AccessorWO<OUTPUT_TYPE, DIM> out,
                                            AccessorRO<INPUT_TYPE, DIM> in,
                                            const Rect<DIM>& out_rect,
                                            const Rect<DIM>& in_rect,
                                            std::vector<int64_t>& axes,
                                            CuPyNumericFFTType type,
                                            CuPyNumericFFTDirection direction,
                                            cudaStream_t stream)
{
  bool is_c2c = (type == CUPYNUMERIC_FFT_Z2Z || type == CUPYNUMERIC_FFT_C2C);
  bool is_r2c = !is_c2c && (type == CUPYNUMERIC_FFT_D2Z || type == CUPYNUMERIC_FFT_R2C);
  bool is_c2r = !is_c2c && !is_r2c;

  bool is_double_precision =
    (type == CUPYNUMERIC_FFT_Z2Z || type == CUPYNUMERIC_FFT_D2Z || type == CUPYNUMERIC_FFT_Z2D);
  auto c2c_subtype = is_double_precision ? CUPYNUMERIC_FFT_Z2Z : CUPYNUMERIC_FFT_C2C;

  // C2C, R2C, C2R all modify input buffer --> create a copy
  OUTPUT_TYPE* out_ptr = out.ptr(out_rect.lo);
  INPUT_TYPE* in_ptr   = nullptr;
  {
    Point<DIM> fft_size_in = in_rect.hi - in_rect.lo + Point<DIM>::ONES();
    size_t num_elements_in = 1;
    for (int32_t i = 0; i < DIM; ++i) {
      num_elements_in *= fft_size_in[i];
    }
    if (is_c2c) {
      // utilize out as temporary store for c2c
      in_ptr = (INPUT_TYPE*)out.ptr(out_rect.lo);
    } else {
      auto input_buffer = create_buffer<INPUT_TYPE, DIM>(fft_size_in, Memory::Kind::GPU_FB_MEM);
      in_ptr            = input_buffer.ptr(Point<DIM>::ZEROES());
    }
    copy_into_buffer<DIM, INPUT_TYPE>(in_ptr, in, in_rect, num_elements_in, stream);
  }

  std::vector<int64_t> c2c_axes(axes.begin(), axes.end() - (is_c2c ? 0 : 1));

  if (is_r2c) {
    // pre-process r2c on last axis
    cufft_r2c_c2r<DIM, OUTPUT_TYPE, INPUT_TYPE>(
      out.ptr(out_rect.lo), in_ptr, out_rect, in_rect, axes.back(), type, direction, stream);
    // run c2c on remaining axes (inplace)
    if (!c2c_axes.empty()) {
      cufft_over_axes_c2c<DIM, OUTPUT_TYPE>(
        out_ptr, out_ptr, out_rect, c2c_axes, c2c_subtype, direction, stream);
    }
  } else if (is_c2c) {
    assert(!c2c_axes.empty());
    // run c2c on all axes (INPUT_TYPE == OUTPUT_TYPE)
    cufft_over_axes_c2c<DIM, INPUT_TYPE>(
      (INPUT_TYPE*)out_ptr, in_ptr, in_rect, c2c_axes, type, direction, stream);
  } else if (is_c2r) {
    // run c2c on all but last axis (inplace)
    if (!c2c_axes.empty()) {
      cufft_over_axes_c2c<DIM, INPUT_TYPE>(
        in_ptr, in_ptr, in_rect, c2c_axes, c2c_subtype, direction, stream);
    }
    // run c2r on last axis
    cufft_r2c_c2r<DIM, OUTPUT_TYPE, INPUT_TYPE>(
      out_ptr, in_ptr, out_rect, in_rect, axes.back(), type, direction, stream);
  }
}

template <CuPyNumericFFTType FFT_TYPE, Type::Code CODE_OUT, Type::Code CODE_IN, int32_t DIM>
struct FFTImplBody<VariantKind::GPU, FFT_TYPE, CODE_OUT, CODE_IN, DIM> {
  TaskContext context;
  explicit FFTImplBody(TaskContext context) : context(context) {}

  using INPUT_TYPE  = type_of<CODE_IN>;
  using OUTPUT_TYPE = type_of<CODE_OUT>;

  __host__ void operator()(AccessorWO<OUTPUT_TYPE, DIM> out,
                           AccessorRO<INPUT_TYPE, DIM> in,
                           const Rect<DIM>& out_rect,
                           const Rect<DIM>& in_rect,
                           std::vector<int64_t>& axes,
                           CuPyNumericFFTDirection direction,
                           bool operate_over_axes) const
  {
    auto stream = context.get_task_stream();
    assert(out.accessor.is_dense_row_major(out_rect));
    // FFTs are computed as 1D over different axes. Slower than performing the full FFT in a single
    // step.
    if (operate_over_axes || DIM > 3) {
      cufft_over_axes<DIM, OUTPUT_TYPE, INPUT_TYPE>(
        out, in, out_rect, in_rect, axes, FFT_TYPE, direction, stream);
    }
    // If we have one axis per dimension, then it can be done as a single operation (more
    // performant). Only available for DIM <= 3
    else {
      // FFTs are computed as a single step of DIM
      cufft_operation<DIM, OUTPUT_TYPE, INPUT_TYPE>(
        out, in, out_rect, in_rect, axes, FFT_TYPE, direction, stream);
    }
  }
};

/*static*/ void FFTTask::gpu_variant(TaskContext context)
{
  fft_template<VariantKind::GPU>(context);
};

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  FFTTask::register_variants();
  return 0;
}();
}  // namespace

}  // namespace cupynumeric
