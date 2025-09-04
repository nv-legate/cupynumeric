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

#include "cupynumeric/index/select.h"
#include "cupynumeric/index/select_template.inl"
#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  select_kernel_dense(VAL* outptr,
                      uint32_t narrays,
                      legate::Buffer<const bool*, 1> condlist,
                      legate::Buffer<const VAL*, 1> choicelist,
                      VAL default_val,
                      int volume)
{
  const size_t idx = global_tid_1d();
  if (idx >= volume) {
    return;
  }
  outptr[idx] = default_val;
  for (int32_t c = (narrays - 1); c >= 0; c--) {
    if (condlist[c][idx]) {
      outptr[idx] = choicelist[c][idx];
    }
  }
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  select_kernel(const AccessorWO<VAL, DIM> out,
                uint32_t narrays,
                const legate::Buffer<AccessorRO<bool, DIM>> condlist,
                const legate::Buffer<AccessorRO<VAL, DIM>> choicelist,
                VAL default_val,
                const Rect<DIM> rect,
                const Pitches<DIM - 1> pitches,
                int volume)
{
  const size_t tid = global_tid_1d();
  if (tid >= volume) {
    return;
  }
  auto p = pitches.unflatten(tid, rect.lo);
  out[p] = default_val;
  for (int32_t c = (narrays - 1); c >= 0; c--) {
    if (condlist[c][p]) {
      out[p] = choicelist[c][p];
    }
  }
}

using namespace legate;

template <Type::Code CODE, int DIM>
struct SelectImplBody<VariantKind::GPU, CODE, DIM> {
  TaskContext context;
  explicit SelectImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const std::vector<AccessorRO<bool, DIM>>& condlist,
                  const std::vector<AccessorRO<VAL, DIM>>& choicelist,
                  VAL default_val,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  bool dense) const
  {
    uint32_t narrays = condlist.size();
#ifdef DEBUG_CUPYNUMERIC
    assert(narrays == choicelist.size());
#endif
    const size_t blocks = (rect.volume() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto stream = context.get_task_stream();

    if (dense) {
      auto cond_arr = create_buffer<const bool*>(condlist.size(), legate::Memory::Kind::Z_COPY_MEM);
      for (uint32_t idx = 0; idx < condlist.size(); ++idx) {
        cond_arr[idx] = condlist[idx].ptr(rect);
      }
      auto choice_arr =
        create_buffer<const VAL*>(choicelist.size(), legate::Memory::Kind::Z_COPY_MEM);
      for (uint32_t idx = 0; idx < choicelist.size(); ++idx) {
        choice_arr[idx] = choicelist[idx].ptr(rect);
      }

      VAL* outptr = out.ptr(rect);
      select_kernel_dense<VAL><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        outptr, narrays, cond_arr, choice_arr, default_val, rect.volume());

    } else {  // not dense
      auto cond_arr =
        create_buffer<AccessorRO<bool, DIM>>(condlist.size(), legate::Memory::Kind::Z_COPY_MEM);
      for (uint32_t idx = 0; idx < condlist.size(); ++idx) {
        cond_arr[idx] = condlist[idx];
      }
      auto choice_arr =
        create_buffer<AccessorRO<VAL, DIM>>(choicelist.size(), legate::Memory::Kind::Z_COPY_MEM);
      for (uint32_t idx = 0; idx < choicelist.size(); ++idx) {
        choice_arr[idx] = choicelist[idx];
      }

      select_kernel<VAL, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        out, narrays, cond_arr, choice_arr, default_val, rect, pitches, rect.volume());
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void SelectTask::gpu_variant(TaskContext context)
{
  select_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
