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

// Useful for IDEs
#include "cupynumeric/set/unique_reduce.h"
#include "cupynumeric/pitches.h"

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

namespace cupynumeric {

using namespace legate;

template <typename exe_pol_t>
struct UniqueReduceImpl {
  TaskContext context;
  explicit UniqueReduceImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE>
  void operator()(legate::PhysicalStore output,
                  const std::vector<legate::PhysicalArray>& input_arrs,
                  exe_pol_t exe_pol)
  {
    using VAL = type_of<CODE>;

    size_t res_size = 0;
    for (auto& input_arr : input_arrs) {
      auto shape = input_arr.shape<1>();
      res_size += shape.hi[0] - shape.lo[0] + 1;
    }
    auto result  = output.create_output_buffer<VAL, 1>(Point<1>(res_size));
    VAL* res_ptr = result.ptr(0);

    size_t offset = 0;
    for (auto& input_arr : input_arrs) {
      size_t strides[1];
      Rect<1> shape     = input_arr.shape<1>();
      size_t volume     = shape.volume();
      const VAL* in_ptr = input_arr.data().read_accessor<VAL, 1>(shape).ptr(shape, strides);
      assert(shape.volume() <= 1 || strides[0] == 1);
      thrust::copy(exe_pol, in_ptr, in_ptr + volume, res_ptr + offset);
      offset += volume;
    }
    assert(offset == res_size);

    thrust::sort(exe_pol, res_ptr, res_ptr + res_size);
    VAL* actual_end = thrust::unique(exe_pol, res_ptr, res_ptr + res_size);
    output.bind_data(result, Point<1>(actual_end - res_ptr));
  }
};

template <typename exe_pol_t>
static void unique_reduce_template(TaskContext& context, const exe_pol_t& exe_pol)
{
  auto inputs = context.inputs();
  auto output = context.output(0);
  type_dispatch(
    output.type().code(), UniqueReduceImpl<exe_pol_t>{context}, output, inputs, exe_pol);
}

}  // namespace cupynumeric
