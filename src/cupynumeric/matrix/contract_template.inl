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
#include "cupynumeric/matrix/contract.h"

#if 0  // debugging output
#include "legate/utilities/debug.h"
#include <unistd.h>
#endif

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct ContractImplBody;

template <Type::Code CODE>
struct support_contract : std::false_type {};
template <>
struct support_contract<Type::Code::FLOAT16> : std::true_type {};
template <>
struct support_contract<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_contract<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_contract<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_contract<Type::Code::COMPLEX128> : std::true_type {};

#if 0  // debugging output

template <typename T>
void print_vec(const char* title, const std::vector<T>& vals)
{
  std::cout << title << " =";
  for (const T& v : vals) std::cout << " " << v;
  std::cout << std::endl;
  std::cout.flush();
}

template <typename T>
void print_span(const char* title, legate::Span<T>& vals)
{
  std::cout << title << " =";
  for (size_t i = 0; i < vals.size(); ++i) std::cout << " " << vals[i];
  std::cout << std::endl;
  std::cout.flush();
}

template <typename T>
void print_ptr(const char* title, const T* vals, size_t len)
{
  std::cout << title << " =";
  for (size_t i = 0; i < len; ++i) std::cout << " " << vals[i];
  std::cout << std::endl;
  std::cout.flush();
}

#endif

template <VariantKind KIND>
struct ContractImpl {
  template <Type::Code CODE, int DIM, std::enable_if_t<support_contract<CODE>::value>* = nullptr>
  void operator()(ContractArgs& args) const
  {
    using T = type_of<CODE>;

    std::vector<int64_t> lhs_shape;
    std::vector<int64_t> lhs_strides;
    std::vector<int32_t> lhs_modes;
    Rect<DIM> lhs_bloated_shape = args.lhs.shape<DIM>();
    size_t lhs_bloated_strides[DIM];
    AccessorRD<SumReduction<T>, true, DIM> lhs_acc =
      args.lhs.reduce_accessor<SumReduction<T>, true, DIM>(lhs_bloated_shape);
    T* lhs_data = lhs_acc.ptr(lhs_bloated_shape, lhs_bloated_strides);
    for (int i = 0; i < DIM; ++i) {
      if (!args.lhs_dim_mask[i]) {
        continue;
      }
      lhs_shape.push_back(lhs_bloated_shape.hi[i] - lhs_bloated_shape.lo[i] + 1);
      lhs_strides.push_back(lhs_bloated_strides[i]);
      lhs_modes.push_back(i + 'a');
    }

    std::vector<int64_t> rhs1_shape;
    std::vector<int64_t> rhs1_strides;
    std::vector<int32_t> rhs1_modes;
    Rect<DIM> rhs1_bloated_shape = args.rhs1.shape<DIM>();
    size_t rhs1_bloated_strides[DIM];
    AccessorRO<T, DIM> rhs1_acc = args.rhs1.read_accessor<T, DIM>(rhs1_bloated_shape);
    const T* rhs1_data          = rhs1_acc.ptr(rhs1_bloated_shape, rhs1_bloated_strides);
    for (int i = 0; i < DIM; ++i) {
      if (!args.rhs1_dim_mask[i]) {
        continue;
      }
      rhs1_shape.push_back(rhs1_bloated_shape.hi[i] - rhs1_bloated_shape.lo[i] + 1);
      rhs1_strides.push_back(rhs1_bloated_strides[i]);
      rhs1_modes.push_back(i + 'a');
    }

    std::vector<int64_t> rhs2_shape;
    std::vector<int64_t> rhs2_strides;
    std::vector<int32_t> rhs2_modes;
    Rect<DIM> rhs2_bloated_shape = args.rhs2.shape<DIM>();
    size_t rhs2_bloated_strides[DIM];
    AccessorRO<T, DIM> rhs2_acc = args.rhs2.read_accessor<T, DIM>(rhs2_bloated_shape);
    const T* rhs2_data          = rhs2_acc.ptr(rhs2_bloated_shape, rhs2_bloated_strides);
    for (int i = 0; i < DIM; ++i) {
      if (!args.rhs2_dim_mask[i]) {
        continue;
      }
      rhs2_shape.push_back(rhs2_bloated_shape.hi[i] - rhs2_bloated_shape.lo[i] + 1);
      rhs2_strides.push_back(rhs2_bloated_strides[i]);
      rhs2_modes.push_back(i + 'a');
    }

    // Intersect the bloated shapes of all arrays, to get the accurate shape of the (bloated) tile
    // we are operating on. Each array on its own will not have accurate bounds information for all
    // dimensions. Specifically, it will not know the tile bounds on any dimensions it has been
    // promoted on. However, each dimension should be actually present on at least one array.
    Rect<DIM> bloated_shape =
      lhs_bloated_shape.intersection(rhs1_bloated_shape).intersection(rhs2_bloated_shape);
    // cuTensor will not work correctly with empty domains, so check this here
    if (bloated_shape.empty()) {
      return;
    }

#if 0  // debugging output
    // Stagger the debugging output from different processors
    sleep(Processor::get_executing_processor().id % 4);
    std::cout << "start contract kernel:" << std::endl;
    std::cout << "lhs:" << std::endl;
    std::cout << "lhs_bloated_shape = " << lhs_bloated_shape << std::endl;
    print_ptr("lhs_bloated_strides", lhs_bloated_strides, DIM);
    print_span("lhs_dim_mask", args.lhs_dim_mask);
    print_vec("lhs_shape", lhs_shape);
    print_vec("lhs_strides", lhs_strides);
    print_vec("lhs_modes", lhs_modes);
    std::cout << "lhs_data = " << print_dense_array(lhs_acc, lhs_bloated_shape) << std::endl;
    std::cout << "rhs1:" << std::endl;
    std::cout << "rhs1_bloated_shape = " << rhs1_bloated_shape << std::endl;
    print_ptr("rhs1_bloated_strides", rhs1_bloated_strides, DIM);
    print_span("rhs1_dim_mask", args.rhs1_dim_mask);
    print_vec("rhs1_shape", rhs1_shape);
    print_vec("rhs1_strides", rhs1_strides);
    print_vec("rhs1_modes", rhs1_modes);
    std::cout << "rhs1_data = " << print_dense_array(rhs1_acc, rhs1_bloated_shape) << std::endl;
    std::cout << "rhs2:" << std::endl;
    std::cout << "rhs2_bloated_shape = " << rhs2_bloated_shape << std::endl;
    print_ptr("rhs2_bloated_strides", rhs2_bloated_strides, DIM);
    print_span("rhs2_dim_mask", args.rhs2_dim_mask);
    print_vec("rhs2_shape", rhs2_shape);
    print_vec("rhs2_strides", rhs2_strides);
    print_vec("rhs2_modes", rhs2_modes);
    std::cout << "rhs2_data = " << print_dense_array(rhs2_acc, rhs2_bloated_shape) << std::endl;
    std::cout << std::endl;
    std::cout.flush();
#endif

    ContractImplBody<KIND, CODE>()(lhs_data,
                                   lhs_shape.size(),
                                   lhs_shape.data(),
                                   lhs_strides.data(),
                                   lhs_modes.data(),
                                   rhs1_data,
                                   rhs1_shape.size(),
                                   rhs1_shape.data(),
                                   rhs1_strides.data(),
                                   rhs1_modes.data(),
                                   rhs2_data,
                                   rhs2_shape.size(),
                                   rhs2_shape.data(),
                                   rhs2_strides.data(),
                                   rhs2_modes.data(),
                                   args.lhs.is_readable());

#if 0  // debugging output
    std::cout << "end contract kernel:" << std::endl;
    std::cout << "lhs_data = " << print_dense_array(lhs_acc, lhs_bloated_shape) << std::endl;
    std::cout << "rhs1_data = " << print_dense_array(rhs1_acc, rhs1_bloated_shape) << std::endl;
    std::cout << "rhs2_data = " << print_dense_array(rhs2_acc, rhs2_bloated_shape) << std::endl;
    std::cout << std::endl;
    std::cout.flush();
#endif
  }

  template <Type::Code CODE, int DIM, std::enable_if_t<!support_contract<CODE>::value>* = nullptr>
  void operator()(ContractArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void contract_template(legate::TaskContext& context)
{
  ContractArgs args{context.reduction(0),
                    context.input(0),
                    context.input(1),
                    context.scalar(0).values<const bool>(),
                    context.scalar(1).values<const bool>(),
                    context.scalar(2).values<const bool>()};

  auto dim  = args.lhs.dim();
  auto code = args.lhs.code();

#ifdef DEBUG_CUPYNUMERIC
  assert(dim == args.rhs1.dim());
  assert(dim == args.rhs2.dim());
  assert(dim == static_cast<int32_t>(args.lhs_dim_mask.size()));
  assert(dim == static_cast<int32_t>(args.rhs1_dim_mask.size()));
  assert(dim == static_cast<int32_t>(args.rhs2_dim_mask.size()));
  assert(code == args.rhs1.code());
  assert(code == args.rhs2.code());
#endif

  double_dispatch(dim, code, ContractImpl<KIND>{}, args);
}

}  // namespace cupynumeric
