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
#include <algorithm>
#include <csignal>
#include <optional>
#include <vector>

#include "cupynumeric/fft/fft.h"
#include "cupynumeric/fft/fft_template.inl"

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;
using dim_t = long long;

// Target work area memory footprint is bounded by input size
constexpr double FFT_MAX_WORKAREA_FACTOR = 1.0;

// Compute batch size
__host__ static inline dim_t fft_batch_size(size_t full_workarea_bytes,
                                            dim_t batch,
                                            size_t budget_bytes)
{
  if (batch <= 1 || budget_bytes == 0 || full_workarea_bytes <= budget_bytes) {
    return batch;
  }
  const double ratio = static_cast<double>(budget_bytes) / static_cast<double>(full_workarea_bytes);
  dim_t sub_batch    = static_cast<dim_t>(static_cast<double>(batch) * ratio);
  if (sub_batch < 1) {
    sub_batch = 1;
  }
  if (sub_batch > batch) {
    sub_batch = batch;
  }
  return sub_batch;
}

// Bind a single work area, sized for the largest of the provided plans, to each of them.
__host__ static inline void fft_set_workarea(std::initializer_list<cufftContext*> ctxs,
                                             Buffer<uint8_t>& buffer,
                                             size_t& buffer_size)
{
  size_t need = 0;
  for (auto* ctx : ctxs) {
    if (ctx != nullptr) {
      need = std::max(need, ctx->workareaSize());
    }
  }
  if (need == 0) {
    return;
  }
  if (need > buffer_size) {
    if (buffer_size > 0) {
      buffer.destroy();
    }
    buffer      = create_buffer<uint8_t>(need, Memory::Kind::GPU_FB_MEM);
    buffer_size = need;
  }
  for (auto* ctx : ctxs) {
    if (ctx != nullptr) {
      CHECK_CUFFT(cufftSetWorkArea(ctx->handle(), buffer.ptr(0)));
    }
  }
}

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

// Decide whether a multi-axis transform can be expressed as a single batched N-D cufft plan.
__host__ static inline bool nd_subset_applicable(const std::vector<int64_t>& axes, int32_t dim)
{
  std::vector<int64_t> u(axes.begin(), axes.end());
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());

  // Repeated axes mean the transform is applied multiple times -> cannot be a single plan.
  if (u.size() != axes.size()) {
    return false;
  }
  const int32_t m = static_cast<int32_t>(u.size());
  // cuFFT supports at most rank-3 transforms.
  if (m < 1 || m > 3) {
    return false;
  }
  // Axes must form a contiguous block.
  if (u.back() - u.front() + 1 != m) {
    return false;
  }
  // The block must touch an array end so all batch axes collapse to one uniform-stride dimension.
  return (u.front() == 0 || u.back() == dim - 1);
}

// Perform a multi-axis C2C/Z2Z FFT as a single batched N-D cufft plan (one cufftXtExec), instead
// of looping over 1D plans. Only valid when nd_subset_applicable() holds.
template <int32_t DIM, typename INOUT_TYPE>
__host__ static inline void cufft_batched_subset(AccessorWO<INOUT_TYPE, DIM> out,
                                                 AccessorRO<INOUT_TYPE, DIM> in,
                                                 const Rect<DIM>& out_rect,
                                                 const Rect<DIM>& in_rect,
                                                 std::vector<int64_t>& axes,
                                                 CuPyNumericFFTType type,
                                                 CuPyNumericFFTDirection direction,
                                                 cudaStream_t stream)
{
  const Point<DIM> one = Point<DIM>::ONES();
  Point<DIM> fft_size  = in_rect.hi - in_rect.lo + one;

  // Deduplicated, sorted transform axes. The C2C/Z2Z transform commutes across axes, so the
  // user-provided order does not affect the result.
  std::vector<int64_t> u(axes.begin(), axes.end());
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());

  const int32_t lo = static_cast<int32_t>(u.front());
  const int32_t hi = static_cast<int32_t>(u.back());
  const int32_t m  = hi - lo + 1;

  // Transform-block dimensions and the volume of a single m-D transform.
  dim_t n[DIM];
  dim_t blockvol = 1;
  for (int32_t i = 0; i < m; ++i) {
    n[i] = fft_size[lo + i];
    blockvol *= n[i];
  }
  // Batch axes split into the dims before (pre) and after (post) the block.
  dim_t pre = 1;
  for (int32_t i = 0; i < lo; ++i) {
    pre *= fft_size[i];
  }
  dim_t post = 1;
  for (int32_t i = hi + 1; i < DIM; ++i) {
    post *= fft_size[i];
  }

  dim_t istride;
  dim_t idist;
  dim_t batch;
  if (post == 1) {
    // Block touches the last axis: leading dims are the (contiguous) batch.
    istride = 1;
    idist   = blockvol;
    batch   = pre;
  } else {
    // Block touches the first axis (pre == 1): trailing dims are the batch and provide the
    // per-element stride inside each transform.
    istride = post;
    idist   = 1;
    batch   = post;
  }

  // Materialize a dense input pointer (copy non-dense inputs into a buffer).
  const void* in_ptr{nullptr};
  if (in.accessor.is_dense_row_major(in_rect)) {
    in_ptr = in.ptr(in_rect.lo);
  } else {
    auto buffer = create_buffer<INOUT_TYPE, DIM>(fft_size, Memory::Kind::GPU_FB_MEM);
    in_ptr      = buffer.ptr(Point<DIM>::ZEROES());
    copy_into_buffer((INOUT_TYPE*)in_ptr, in, in_rect, in_rect.volume(), stream);
  }
  const INOUT_TYPE* in_base = static_cast<const INOUT_TYPE*>(in_ptr);
  INOUT_TYPE* out_base      = out.ptr(out_rect.lo);

  Buffer<uint8_t> workarea_buffer;
  size_t workarea_size = 0;
  auto cufft_context   = get_cufft_plan(
    (cufftType)type, cufftPlanParams(m, n, n, istride, idist, n, istride, idist, batch), stream);
  fft_set_workarea({&cufft_context}, workarea_buffer, workarea_size);

  // C2C/Z2Z share the in/out layout, so both advance by b0 * idist elements.
  CHECK_CUFFT(cufftXtExec(cufft_context.handle(),
                          const_cast<INOUT_TYPE*>(in_base),
                          static_cast<void*>(out_base),
                          static_cast<int32_t>(direction)));

  // synchronize before the plans / work area run out of scope
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
                                                int32_t bluestein_mask,
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
  size_t workarea_size = 0;
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

    // Only this axis' Bluestein bit matters: a non-Bluestein axis keeps the full batch.
    const bool bluestein = ((bluestein_mask >> axis) & 1) != 0;

    // Bound the Bluestein work area by sub_batching the per-slice batch dimension.
    dim_t sub_batch = batches;
    if (bluestein) {
      auto probe =
        get_cufft_plan((cufftType)type,
                       cufftPlanParams(1, &size_1d, n, stride, dist, n, stride, dist, batches),
                       stream);
      const size_t budget =
        static_cast<size_t>(FFT_MAX_WORKAREA_FACTOR * num_elements * sizeof(INOUT_TYPE));
      sub_batch = fft_batch_size(probe.workareaSize(), batches, budget);
    }

    // At most two distinct chunk sizes occur across all slices: the full chunk (sub_batch)
    // and, when batches is not a multiple of sub_batch, the tail. Fetch the plan(s) once here
    // rather than per slice/chunk, and bind a single shared work area to them.
    const dim_t tail = batches % sub_batch;
    auto ctx_full =
      get_cufft_plan((cufftType)type,
                     cufftPlanParams(1, &size_1d, n, stride, dist, n, stride, dist, sub_batch),
                     stream);
    std::optional<cufftContext> ctx_tail;
    if (tail != 0) {
      ctx_tail.emplace(
        get_cufft_plan((cufftType)type,
                       cufftPlanParams(1, &size_1d, n, stride, dist, n, stride, dist, tail),
                       stream));
    }
    fft_set_workarea({&ctx_full, ctx_tail ? &*ctx_tail : nullptr}, workarea_buffer, workarea_size);

    for (int64_t slice = 0; slice < num_slices; ++slice) {
      INOUT_TYPE* slice_ptr = out + slice * offset;
      for (dim_t b0 = 0; b0 < batches; b0 += sub_batch) {
        const dim_t cb      = std::min(sub_batch, batches - b0);
        cufftHandle& handle = (cb == sub_batch) ? ctx_full.handle() : ctx_tail->handle();
        CHECK_CUFFT(cufftXtExec(handle,
                                static_cast<void*>(slice_ptr + b0 * dist),
                                static_cast<void*>(slice_ptr + b0 * dist),
                                static_cast<int32_t>(direction)));
      }
    }
    // synchronize before the plans / work area run out of scope
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
                                          int32_t bluestein_mask,
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

  dim_t inembed_1d[1] = {inembed[axis]};
  dim_t onembed_1d[1] = {onembed[axis]};

  // cuFFT requires *both* pointers handed to cufftXtExec for R2C/C2R to be
  // aligned to the complex element type (cufftComplex / cufftDoubleComplex),
  // including the real side. The real base of an exec is
  //   real_buffer + slice * real_offset + b0 * real_dist
  // so both the per-slice and the per-chunk term must keep it complex-aligned
  // (i.e. land on an even real-element offset).
  const bool real_is_input  = (type == CUPYNUMERIC_FFT_D2Z || type == CUPYNUMERIC_FFT_R2C);
  const dim_t real_dist     = real_is_input ? idist : odist;
  const int64_t real_offset = real_is_input ? offset_in : offset_out;

  // Only this axis' Bluestein bit matters: a non-Bluestein axis keeps the full batch.
  const bool bluestein = ((bluestein_mask >> axis) & 1) != 0;

  dim_t sub_batch = batches;
  if (bluestein) {
    auto probe = get_cufft_plan(
      (cufftType)type,
      cufftPlanParams(1, &size_1d, inembed_1d, istride, idist, onembed_1d, ostride, odist, batches),
      stream);
    const size_t data_bytes =
      std::max(num_elements_in * sizeof(INPUT_TYPE), num_elements_out * sizeof(OUTPUT_TYPE));
    const size_t budget = static_cast<size_t>(FFT_MAX_WORKAREA_FACTOR * data_bytes);
    sub_batch           = fft_batch_size(probe.workareaSize(), batches, budget);

    // Chunking advances the real base by b0 * real_dist; force sub_batch even so
    // every chunk base stays complex-aligned when the real distance is odd.
    if ((real_dist % 2 != 0) && (sub_batch % 2 != 0) && (sub_batch < batches)) {
      sub_batch = (sub_batch >= 3) ? (sub_batch - 1) : (batches >= 2 ? dim_t{2} : batches);
    }
  }

  // The per-slice term (slice * real_offset) is fixed by the data layout and is
  // unaffected by chunking. When the transform axis is interior (num_slices > 1)
  // and the real per-slice volume is odd, odd slices would land the real base on
  // an odd element offset. Stage the real side of each slice through an aligned
  // scratch buffer (one slice's worth of real data) so cuFFT always sees a
  // complex-aligned real base.
  const bool stage_real = (num_slices > 1) && (real_offset % 2 != 0);
  Buffer<uint8_t> stage_buffer;
  INPUT_TYPE* in_real_stage   = nullptr;
  OUTPUT_TYPE* out_real_stage = nullptr;
  if (stage_real) {
    const size_t real_elem = real_is_input ? sizeof(INPUT_TYPE) : sizeof(OUTPUT_TYPE);
    stage_buffer           = create_buffer<uint8_t>(static_cast<size_t>(real_offset) * real_elem,
                                          Memory::Kind::GPU_FB_MEM);
    if (real_is_input) {
      in_real_stage = reinterpret_cast<INPUT_TYPE*>(stage_buffer.ptr(0));
    } else {
      out_real_stage = reinterpret_cast<OUTPUT_TYPE*>(stage_buffer.ptr(0));
    }
  }

  Buffer<uint8_t> workarea_buffer;
  size_t workarea_size = 0;

  // At most two distinct chunk sizes occur across all slices: the full chunk (sub_batch)
  // and, when batches is not a multiple of sub_batch, the tail. Fetch the plan(s) once here
  // rather than per slice/chunk, and bind a single shared work area to them.
  const dim_t tail = batches % sub_batch;
  auto ctx_full    = get_cufft_plan(
    (cufftType)type,
    cufftPlanParams(1, &size_1d, inembed_1d, istride, idist, onembed_1d, ostride, odist, sub_batch),
    stream);
  std::optional<cufftContext> ctx_tail;
  if (tail != 0) {
    ctx_tail.emplace(get_cufft_plan(
      (cufftType)type,
      cufftPlanParams(1, &size_1d, inembed_1d, istride, idist, onembed_1d, ostride, odist, tail),
      stream));
  }
  fft_set_workarea({&ctx_full, ctx_tail ? &*ctx_tail : nullptr}, workarea_buffer, workarea_size);

  for (int64_t slice = 0; slice < num_slices; ++slice) {
    INPUT_TYPE* in_slice   = in + slice * offset_in;
    OUTPUT_TYPE* out_slice = out + slice * offset_out;

    // R2C staging: copy this slice's real input into the aligned buffer up-front.
    if (in_real_stage != nullptr) {
      CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(in_real_stage,
                                             in_slice,
                                             static_cast<size_t>(real_offset) * sizeof(INPUT_TYPE),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
    }

    INPUT_TYPE* exec_in   = (in_real_stage != nullptr) ? in_real_stage : in_slice;
    OUTPUT_TYPE* exec_out = (out_real_stage != nullptr) ? out_real_stage : out_slice;

    for (dim_t b0 = 0; b0 < batches; b0 += sub_batch) {
      const dim_t cb      = std::min(sub_batch, batches - b0);
      cufftHandle& handle = (cb == sub_batch) ? ctx_full.handle() : ctx_tail->handle();
      CHECK_CUFFT(cufftXtExec(handle,
                              static_cast<void*>(exec_in + b0 * idist),
                              static_cast<void*>(exec_out + b0 * odist),
                              static_cast<int32_t>(direction)));
    }

    // C2R staging: copy the aligned real output for this slice back into place.
    if (out_real_stage != nullptr) {
      CUPYNUMERIC_CHECK_CUDA(cudaMemcpyAsync(out_slice,
                                             out_real_stage,
                                             static_cast<size_t>(real_offset) * sizeof(OUTPUT_TYPE),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
    }
  }
  // synchronize before the plans / work area run out of scope
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
                                            int32_t bluestein_mask,
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
    cufft_r2c_c2r<DIM, OUTPUT_TYPE, INPUT_TYPE>(out.ptr(out_rect.lo),
                                                in_ptr,
                                                out_rect,
                                                in_rect,
                                                axes.back(),
                                                type,
                                                direction,
                                                bluestein_mask,
                                                stream);
    // run c2c on remaining axes (inplace)
    if (!c2c_axes.empty()) {
      cufft_over_axes_c2c<DIM, OUTPUT_TYPE>(
        out_ptr, out_ptr, out_rect, c2c_axes, c2c_subtype, direction, bluestein_mask, stream);
    }
  } else if (is_c2c) {
    assert(!c2c_axes.empty());
    // run c2c on all axes (INPUT_TYPE == OUTPUT_TYPE)
    cufft_over_axes_c2c<DIM, INPUT_TYPE>(
      (INPUT_TYPE*)out_ptr, in_ptr, in_rect, c2c_axes, type, direction, bluestein_mask, stream);
  } else if (is_c2r) {
    // run c2c on all but last axis (inplace)
    if (!c2c_axes.empty()) {
      cufft_over_axes_c2c<DIM, INPUT_TYPE>(
        in_ptr, in_ptr, in_rect, c2c_axes, c2c_subtype, direction, bluestein_mask, stream);
    }
    // run c2r on last axis
    cufft_r2c_c2r<DIM, OUTPUT_TYPE, INPUT_TYPE>(
      out_ptr, in_ptr, out_rect, in_rect, axes.back(), type, direction, bluestein_mask, stream);
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
                           bool operate_over_axes,
                           int32_t bluestein_mask) const
  {
    auto stream = context.get_task_stream();
    assert(out.accessor.is_dense_row_major(out_rect));

    constexpr bool is_c2c = (FFT_TYPE == CUPYNUMERIC_FFT_C2C || FFT_TYPE == CUPYNUMERIC_FFT_Z2Z);

    // If we have one axis per dimension, then it can be done as a single operation (more
    // performant). Only available for DIM <= 3.
    if (!operate_over_axes && DIM <= 3 && bluestein_mask == 0) {
      // FFTs are computed as a single step of DIM
      cufft_operation<DIM, OUTPUT_TYPE, INPUT_TYPE>(
        out, in, out_rect, in_rect, axes, FFT_TYPE, direction, stream);
      return;
    }

    // A multi-axis C2C/Z2Z transform over a contiguous block of axes that touches an array end can
    // be expressed as a single batched N-D cufft plan (one cufftXtExec).
    if constexpr (is_c2c) {
      if (bluestein_mask == 0 && nd_subset_applicable(axes, DIM)) {
        cufft_batched_subset<DIM, INPUT_TYPE>(
          out, in, out_rect, in_rect, axes, FFT_TYPE, direction, stream);
        return;
      }
    }

    // Fallback: FFTs are computed as multiple 1D FFTs over the requested axes. Slower than
    // performing the FFT in a single step.
    cufft_over_axes<DIM, OUTPUT_TYPE, INPUT_TYPE>(
      out, in, out_rect, in_rect, axes, FFT_TYPE, direction, bluestein_mask, stream);
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
