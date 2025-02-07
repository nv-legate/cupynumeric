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
#include "cupynumeric/matrix/util.h"

#include <tblis/tblis.h>
#include <omp.h>

namespace cupynumeric {

using namespace tblis;

template <>
struct ContractImplBody<VariantKind::OMP, Type::Code::FLOAT32> {
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
    tblis_tensor lhs;
    tblis_init_tensor_s(&lhs, lhs_ndim, lhs_shape, lhs_data, lhs_strides);

    tblis_tensor rhs1;
    tblis_init_tensor_s(&rhs1, rhs1_ndim, rhs1_shape, const_cast<float*>(rhs1_data), rhs1_strides);

    tblis_tensor rhs2;
    tblis_init_tensor_s(&rhs2, rhs2_ndim, rhs2_shape, const_cast<float*>(rhs2_data), rhs2_strides);

    tblis_tensor_mult(nullptr, nullptr, &rhs1, rhs1_modes, &rhs2, rhs2_modes, &lhs, lhs_modes);
  }
};

template <>
struct ContractImplBody<VariantKind::OMP, Type::Code::FLOAT64> {
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
    tblis_tensor lhs;
    tblis_init_tensor_d(&lhs, lhs_ndim, lhs_shape, lhs_data, lhs_strides);

    tblis_tensor rhs1;
    tblis_init_tensor_d(&rhs1, rhs1_ndim, rhs1_shape, const_cast<double*>(rhs1_data), rhs1_strides);

    tblis_tensor rhs2;
    tblis_init_tensor_d(&rhs2, rhs2_ndim, rhs2_shape, const_cast<double*>(rhs2_data), rhs2_strides);

    tblis_tensor_mult(nullptr, nullptr, &rhs1, rhs1_modes, &rhs2, rhs2_modes, &lhs, lhs_modes);
  }
};

template <>
struct ContractImplBody<VariantKind::OMP, Type::Code::FLOAT16> {
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
    // TBLIS doesn't handle half-precision floating point directly, so we have to go through a
    // conversion to single-precision.

    std::vector<int64_t> lhs_copy_strides(lhs_ndim);
    int64_t lhs_size     = calculate_volume(lhs_ndim, lhs_shape, lhs_copy_strides.data());
    float* lhs_copy_data = allocate_buffer(lhs_size);
    half_tensor_to_float(lhs_copy_data, lhs_data, lhs_ndim, lhs_shape, lhs_strides);

    std::vector<int64_t> rhs1_copy_strides(rhs1_ndim);
    int64_t rhs1_size     = calculate_volume(rhs1_ndim, rhs1_shape, rhs1_copy_strides.data());
    float* rhs1_copy_data = allocate_buffer(rhs1_size);
    half_tensor_to_float(rhs1_copy_data, rhs1_data, rhs1_ndim, rhs1_shape, rhs1_strides);

    std::vector<int64_t> rhs2_copy_strides(rhs2_ndim);
    int64_t rhs2_size     = calculate_volume(rhs2_ndim, rhs2_shape, rhs2_copy_strides.data());
    float* rhs2_copy_data = allocate_buffer(rhs2_size);
    half_tensor_to_float(rhs2_copy_data, rhs2_data, rhs2_ndim, rhs2_shape, rhs2_strides);

    ContractImplBody<VariantKind::OMP, Type::Code::FLOAT32>{}(lhs_copy_data,
                                                              lhs_ndim,
                                                              lhs_shape,
                                                              lhs_copy_strides.data(),
                                                              lhs_modes,
                                                              rhs1_copy_data,
                                                              rhs1_ndim,
                                                              rhs1_shape,
                                                              rhs1_copy_strides.data(),
                                                              rhs1_modes,
                                                              rhs2_copy_data,
                                                              rhs2_ndim,
                                                              rhs2_shape,
                                                              rhs2_copy_strides.data(),
                                                              rhs2_modes,
                                                              lhs_overwritable);

    float_tensor_to_half(lhs_data, lhs_copy_data, lhs_ndim, lhs_shape, lhs_strides);
  }
};

template <>
struct ContractImplBody<VariantKind::OMP, Type::Code::COMPLEX64> {
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
    tblis_tensor lhs;
    tblis_init_tensor_c(
      &lhs, lhs_ndim, lhs_shape, reinterpret_cast<std::complex<float>*>(lhs_data), lhs_strides);

    tblis_tensor rhs1;
    tblis_init_tensor_c(
      &rhs1,
      rhs1_ndim,
      rhs1_shape,
      reinterpret_cast<std::complex<float>*>(const_cast<complex<float>*>(rhs1_data)),
      rhs1_strides);

    tblis_tensor rhs2;
    tblis_init_tensor_c(
      &rhs2,
      rhs2_ndim,
      rhs2_shape,
      reinterpret_cast<std::complex<float>*>(const_cast<complex<float>*>(rhs2_data)),
      rhs2_strides);

    tblis_tensor_mult(nullptr, nullptr, &rhs1, rhs1_modes, &rhs2, rhs2_modes, &lhs, lhs_modes);
  }
};

template <>
struct ContractImplBody<VariantKind::OMP, Type::Code::COMPLEX128> {
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
    tblis_tensor lhs;
    tblis_init_tensor_z(
      &lhs, lhs_ndim, lhs_shape, reinterpret_cast<std::complex<double>*>(lhs_data), lhs_strides);

    tblis_tensor rhs1;
    tblis_init_tensor_z(
      &rhs1,
      rhs1_ndim,
      rhs1_shape,
      reinterpret_cast<std::complex<double>*>(const_cast<complex<double>*>(rhs1_data)),
      rhs1_strides);

    tblis_tensor rhs2;
    tblis_init_tensor_z(
      &rhs2,
      rhs2_ndim,
      rhs2_shape,
      reinterpret_cast<std::complex<double>*>(const_cast<complex<double>*>(rhs2_data)),
      rhs2_strides);

    tblis_tensor_mult(nullptr, nullptr, &rhs1, rhs1_modes, &rhs2, rhs2_modes, &lhs, lhs_modes);
  }
};

/*static*/ void ContractTask::omp_variant(legate::TaskContext context)
{
  std::stringstream ss;
  ss << omp_get_max_threads();
  std::string str = ss.str();
  setenv("TBLIS_NUM_THREADS", str.data(), false /*overwrite*/);
  contract_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
