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

#include "cupynumeric/search/nonzero.h"
#include "cupynumeric/search/nonzero_template.inl"

#include "cupynumeric/cuda_help.h"

#include "cupynumeric/utilities/thrust_util.h"

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <cuda/std/utility>

namespace cupynumeric {

template <typename Pitches, typename VAL, int32_t DIM>
static void nonzeros(size_t volume,
                     const AccessorRO<VAL, DIM>& in,
                     std::vector<Array>& outputs,
                     Pitches pitches,
                     Point<DIM> origin,
                     cudaStream_t stream)
{
  // step 1:
  // count nonzeros:
  //
  auto point_lambda = [pitches, origin] __host__ __device__(int64_t indx) -> Point<DIM> {
    return pitches.unflatten(indx, origin);
  };

  auto point_iter_begin =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0), point_lambda);

  auto count_nz = thrust::count_if(DEFAULT_POLICY.on(stream),
                                   point_iter_begin,
                                   point_iter_begin + volume,
                                   [in] __device__(auto&& point) { return (in[point] != VAL{0}); });

  // step 2:
  // allocate output accordingly:
  //
  std::vector<Buffer<int64_t>> results;
  for (auto& output : outputs) {
    results.push_back(output.create_output_buffer<int64_t, 1>(Point<1>(count_nz), true));
  }

  auto p_results = create_buffer<int64_t*>(DIM, legate::Memory::Kind::Z_COPY_MEM);
  for (int32_t dim = 0; dim < DIM; ++dim) {
    p_results[dim] = results[dim].ptr(0);
  }

  // step 3:
  // copy non-zero points into output's 1st dimension buffer,
  // repurposed as scratch-space:
  //
  thrust::copy_if(DEFAULT_POLICY.on(stream),
                  thrust::make_counting_iterator<int64_t>(0),
                  thrust::make_counting_iterator<int64_t>(volume),
                  p_results[0],  // use 1st dimension range as scratch space to hold linear indices
                  [in, pitches, origin] __device__(int64_t indx) {
                    auto point = pitches.unflatten(indx, origin);
                    return (in[point] != VAL{0});
                  });

  // step 4:
  // transform output by "scattering" its 1st dimension's linear index (from previous step)
  // to multi-dimensional indices acrross each of the output's dimensions:
  //
  thrust::for_each(DEFAULT_POLICY.on(stream),
                   thrust::make_counting_iterator<int64_t>(0),
                   thrust::make_counting_iterator<int64_t>(count_nz),
                   [pitches, origin, p_results] __device__(int64_t indx) mutable {
                     auto nz_indx = p_results[0][indx];
                     auto point   = pitches.unflatten(nz_indx, origin);

                     for (auto dim = 0; dim < DIM; ++dim) {
                       p_results[dim][indx] = point[dim];
                     }
                   });
}

template <Type::Code CODE, int32_t DIM>
struct NonzeroImplBody<VariantKind::GPU, CODE, DIM> {
  TaskContext context;
  explicit NonzeroImplBody(TaskContext context) : context(context) {}

  using VAL = type_of<CODE>;

  void operator()(std::vector<Array>& outputs,
                  const AccessorRO<VAL, DIM>& in,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume)
  {
    auto stream = context.get_task_stream();
    nonzeros(volume, in, outputs, pitches, rect.lo, stream);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void NonzeroTask::gpu_variant(TaskContext context)
{
  nonzero_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
