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

#include "cupynumeric/matrix/contract.h"
#include "cupynumeric/matrix/contract_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

namespace {  // anonymous

template <typename T>
struct contract_helper {};

template <>
struct contract_helper<__half> {
  static constexpr auto data_type_code = CUTENSOR_R_16F;
  static cutensorComputeDescriptor_t compute_type_code() { return CUTENSOR_COMPUTE_DESC_32F; }
  using scalar_t = float;
};

template <>
struct contract_helper<float> {
  static constexpr auto data_type_code = CUTENSOR_R_32F;
  static cutensorComputeDescriptor_t compute_type_code() { return CUTENSOR_COMPUTE_DESC_32F; }
  using scalar_t = float;
};

template <>
struct contract_helper<double> {
  static constexpr auto data_type_code = CUTENSOR_R_64F;
  static cutensorComputeDescriptor_t compute_type_code() { return CUTENSOR_COMPUTE_DESC_64F; }
  using scalar_t = double;
};

template <>
struct contract_helper<complex<float>> {
  static constexpr auto data_type_code = CUTENSOR_C_32F;
  static cutensorComputeDescriptor_t compute_type_code() { return CUTENSOR_COMPUTE_DESC_32F; }
  using scalar_t = complex<float>;
};

template <>
struct contract_helper<complex<double>> {
  static constexpr auto data_type_code = CUTENSOR_C_64F;
  static cutensorComputeDescriptor_t compute_type_code() { return CUTENSOR_COMPUTE_DESC_64F; }
  using scalar_t = complex<double>;
};

}  // anonymous namespace

template <typename T>
__host__ void contract(T* lhs_data,
                       size_t lhs_ndim,
                       int64_t* lhs_shape,
                       int64_t* lhs_strides,
                       int32_t* lhs_modes,
                       const T* rhs1_data,
                       size_t rhs1_ndim,
                       int64_t* rhs1_shape,
                       int64_t* rhs1_strides,
                       int32_t* rhs1_modes,
                       const T* rhs2_data,
                       size_t rhs2_ndim,
                       int64_t* rhs2_shape,
                       int64_t* rhs2_strides,
                       int32_t* rhs2_modes,
                       bool lhs_overwritable,
                       cudaStream_t task_stream)
{
  // Initialization
  auto handle = get_cutensor();

  // Create tensor descriptors
  constexpr auto data_type_code = contract_helper<T>::data_type_code;
  cutensorTensorDescriptor_t lhs_desc;
  cutensorTensorDescriptor_t rhs1_desc;
  cutensorTensorDescriptor_t rhs2_desc;
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
    handle, &lhs_desc, lhs_ndim, lhs_shape, lhs_strides, data_type_code, sizeof(T)));
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
    handle, &rhs1_desc, rhs1_ndim, rhs1_shape, rhs1_strides, data_type_code, sizeof(T)));
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
    handle, &rhs2_desc, rhs2_ndim, rhs2_shape, rhs2_strides, data_type_code, sizeof(T)));

  // Prepare algorithm description
  cutensorOperationDescriptor_t desc;
  CHECK_CUTENSOR(cutensorCreateContraction(handle,
                                           &desc,
                                           rhs1_desc,
                                           rhs1_modes,
                                           CUTENSOR_OP_IDENTITY,
                                           rhs2_desc,
                                           rhs2_modes,
                                           CUTENSOR_OP_IDENTITY,
                                           lhs_desc,
                                           lhs_modes,
                                           CUTENSOR_OP_IDENTITY,
                                           lhs_desc,
                                           lhs_modes,
                                           contract_helper<T>::compute_type_code()));
  cutensorPlanPreference_t plan_pref;
  CHECK_CUTENSOR(cutensorCreatePlanPreference(
    handle, &plan_pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

  // Allocate intermediate storage
  uint64_t work_size = 0;
  CHECK_CUTENSOR(
    cutensorEstimateWorkspaceSize(handle, desc, plan_pref, CUTENSOR_WORKSPACE_DEFAULT, &work_size));
  // Workspace must be 256-byte aligned per the contract with cuTensor
  auto work_buf = create_buffer<int8_t>(work_size, legate::Memory::GPU_FB_MEM, 256);
  void* work    = work_buf.ptr(Point<1>(0));

  // Execute contraction
  cutensorPlan_t plan;
  CHECK_CUTENSOR(cutensorCreatePlan(handle, &plan, desc, plan_pref, work_size));
  const typename contract_helper<T>::scalar_t alpha = 1.0;
  // lhs_overwritable being true means that the contraciton tasks can overwrite the lhs
  const typename contract_helper<T>::scalar_t beta = lhs_overwritable ? 0.0 : 1.0;
  CHECK_CUTENSOR(cutensorContract(handle,
                                  plan,
                                  &alpha,
                                  rhs1_data,
                                  rhs2_data,
                                  &beta,
                                  lhs_data,
                                  lhs_data,
                                  work,
                                  work_size,
                                  task_stream));

  CUPYNUMERIC_CHECK_CUDA_STREAM(task_stream);

  CHECK_CUTENSOR(cutensorDestroyPlan(plan));
  CHECK_CUTENSOR(cutensorDestroyPlanPreference(plan_pref));
  CHECK_CUTENSOR(cutensorDestroyOperationDescriptor(desc));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(rhs2_desc));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(rhs1_desc));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(lhs_desc));
}

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::FLOAT16> {
  TaskContext context;
  explicit ContractImplBody(TaskContext context) : context(context) {}

  void operator()(__half* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const __half* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const __half* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    auto task_stream = context.get_task_stream();
    contract(lhs_data,
             lhs_ndim,
             lhs_shape,
             lhs_strides,
             lhs_modes,
             rhs1_data,
             rhs1_ndim,
             rhs1_shape,
             rhs1_strides,
             rhs1_modes,
             rhs2_data,
             rhs2_ndim,
             rhs2_shape,
             rhs2_strides,
             rhs2_modes,
             lhs_overwritable,
             task_stream);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::FLOAT32> {
  TaskContext context;
  explicit ContractImplBody(TaskContext context) : context(context) {}

  void operator()(float* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const float* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const float* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    auto task_stream = context.get_task_stream();
    contract(lhs_data,
             lhs_ndim,
             lhs_shape,
             lhs_strides,
             lhs_modes,
             rhs1_data,
             rhs1_ndim,
             rhs1_shape,
             rhs1_strides,
             rhs1_modes,
             rhs2_data,
             rhs2_ndim,
             rhs2_shape,
             rhs2_strides,
             rhs2_modes,
             lhs_overwritable,
             task_stream);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::FLOAT64> {
  TaskContext context;
  explicit ContractImplBody(TaskContext context) : context(context) {}

  void operator()(double* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const double* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const double* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    auto task_stream = context.get_task_stream();
    contract(lhs_data,
             lhs_ndim,
             lhs_shape,
             lhs_strides,
             lhs_modes,
             rhs1_data,
             rhs1_ndim,
             rhs1_shape,
             rhs1_strides,
             rhs1_modes,
             rhs2_data,
             rhs2_ndim,
             rhs2_shape,
             rhs2_strides,
             rhs2_modes,
             lhs_overwritable,
             task_stream);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::COMPLEX64> {
  TaskContext context;
  explicit ContractImplBody(TaskContext context) : context(context) {}

  void operator()(complex<float>* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const complex<float>* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const complex<float>* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    auto task_stream = context.get_task_stream();
    contract(lhs_data,
             lhs_ndim,
             lhs_shape,
             lhs_strides,
             lhs_modes,
             rhs1_data,
             rhs1_ndim,
             rhs1_shape,
             rhs1_strides,
             rhs1_modes,
             rhs2_data,
             rhs2_ndim,
             rhs2_shape,
             rhs2_strides,
             rhs2_modes,
             lhs_overwritable,
             task_stream);
  }
};

template <>
struct ContractImplBody<VariantKind::GPU, Type::Code::COMPLEX128> {
  TaskContext context;
  explicit ContractImplBody(TaskContext context) : context(context) {}

  void operator()(complex<double>* lhs_data,
                  size_t lhs_ndim,
                  int64_t* lhs_shape,
                  int64_t* lhs_strides,
                  int32_t* lhs_modes,
                  const complex<double>* rhs1_data,
                  size_t rhs1_ndim,
                  int64_t* rhs1_shape,
                  int64_t* rhs1_strides,
                  int32_t* rhs1_modes,
                  const complex<double>* rhs2_data,
                  size_t rhs2_ndim,
                  int64_t* rhs2_shape,
                  int64_t* rhs2_strides,
                  int32_t* rhs2_modes,
                  bool lhs_overwritable)
  {
    auto task_stream = context.get_task_stream();
    contract(lhs_data,
             lhs_ndim,
             lhs_shape,
             lhs_strides,
             lhs_modes,
             rhs1_data,
             rhs1_ndim,
             rhs1_shape,
             rhs1_strides,
             rhs1_modes,
             rhs2_data,
             rhs2_ndim,
             rhs2_shape,
             rhs2_strides,
             rhs2_modes,
             lhs_overwritable,
             task_stream);
  }
};

/*static*/ void ContractTask::gpu_variant(TaskContext context)
{
  contract_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
