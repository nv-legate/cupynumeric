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

#include "cupynumeric/cuda_help.h"
#include "legate/data/buffer.h"

namespace cupynumeric {

template <typename REDOP>
class DeviceScalarReductionBuffer {
 private:
  using VAL = typename REDOP::RHS;

 public:
  DeviceScalarReductionBuffer(cudaStream_t stream)
    : buffer_(legate::create_buffer<VAL>(1, legate::Memory::Kind::GPU_FB_MEM))
  {
    VAL identity{REDOP::identity};
    ptr_ = buffer_.ptr(0);
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemcpyAsync(ptr_, &identity, sizeof(VAL), cudaMemcpyHostToDevice, stream));
  }

  template <bool EXCLUSIVE>
  __device__ void reduce(const VAL& value) const
  {
    REDOP::template fold<EXCLUSIVE /*exclusive*/>(*ptr_, value);
  }

  __host__ VAL read(cudaStream_t stream) const
  {
    VAL result{REDOP::identity};
    CUPYNUMERIC_CHECK_CUDA(
      cudaMemcpyAsync(&result, ptr_, sizeof(VAL), cudaMemcpyDeviceToHost, stream));
    CUPYNUMERIC_CHECK_CUDA(cudaStreamSynchronize(stream));
    return result;
  }

  __device__ VAL read() const { return *ptr_; }

 private:
  legate::Buffer<VAL> buffer_;
  VAL* ptr_;
};

}  // namespace cupynumeric
