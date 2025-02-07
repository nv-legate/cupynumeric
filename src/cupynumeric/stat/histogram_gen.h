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

namespace cupynumeric {
namespace detail {
// primary templates, to be specialized (SFINAEd)
//
template <typename exe_policy_t, typename weight_t, typename offset_t, typename = void>
struct segmented_sum_t;

template <typename exe_policy_t>
inline constexpr bool is_host_policy_v =
  std::is_same_v<exe_policy_t, std::remove_cv_t<decltype(thrust::host)>> ||
  std::is_same_v<exe_policy_t, std::remove_cv_t<decltype(thrust::omp::par)>>;

template <typename exe_policy_t, typename = void>
struct sync_policy_t;

namespace accessors {

template <typename element_t>
decltype(auto) get_raw_ptr(Buffer<element_t>& v)
{
  return v.ptr(0);
}

// RO accessor (size, pointer) extractor:
//
template <typename VAL>
std::tuple<size_t, const VAL*> get_accessor_ptr(const AccessorRO<VAL, 1>& src_acc,
                                                const Rect<1>& src_rect)
{
  size_t src_strides[1];
  const VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
  assert(src_rect.volume() == 1 || src_strides[0] == 1);
  //
  // const VAL* src_ptr: need to create a copy with create_buffer(...);
  // since src will get sorted (in-place);
  //
  size_t src_size = src_rect.hi - src_rect.lo + 1;
  return std::make_tuple(src_size, src_ptr);
}
// RD accessor (size, pointer) extractor:
//
template <typename VAL>
std::tuple<size_t, VAL*> get_accessor_ptr(const AccessorRD<SumReduction<VAL>, true, 1>& src_acc,
                                          const Rect<1>& src_rect)
{
  size_t src_strides[1];
  VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
  assert(src_rect.volume() == 1 || src_strides[0] == 1);
  //
  // const VAL* src_ptr: need to create a copy with create_buffer(...);
  // since src will get sorted (in-place);
  //
  size_t src_size = src_rect.hi - src_rect.lo + 1;
  return std::make_tuple(src_size, src_ptr);
}
// accessor copy utility:
//
template <typename VAL, typename exe_policy_t>
std::tuple<size_t, Buffer<VAL>, const VAL*> make_accessor_copy(exe_policy_t exe_pol,
                                                               const AccessorRO<VAL, 1>& src_acc,
                                                               const Rect<1>& src_rect)
{
  size_t src_strides[1];
  const VAL* src_ptr = src_acc.ptr(src_rect, src_strides);
  assert(src_rect.volume() == 1 || src_strides[0] == 1);
  //
  // const VAL* src_ptr: need to create a copy with create_buffer(...);
  // since src will get sorted (in-place);
  //
  size_t src_size      = src_rect.hi - src_rect.lo + 1;
  Buffer<VAL> src_copy = create_buffer<VAL>(src_size);

  thrust::copy_n(exe_pol, src_ptr, src_size, src_copy.ptr(0));

  return std::make_tuple(src_size, src_copy, src_ptr);
}

}  // namespace accessors

// device / host allocator:
//
template <typename elem_t, typename exe_policy_t>
struct allocator_t {
  allocator_t(void) {}

  allocator_t(elem_t, exe_policy_t) {}  // tag-dispatch for CTAD

  elem_t* operator()(exe_policy_t exe_pol, size_t size)
  {
    d_buffer_     = create_buffer<elem_t>(size);
    elem_t* d_ptr = accessors::get_raw_ptr(d_buffer_);

    return d_ptr;
  }

  elem_t* operator()(exe_policy_t exe_pol, size_t size, elem_t init)
  {
    d_buffer_     = create_buffer<elem_t>(size);
    elem_t* d_ptr = accessors::get_raw_ptr(d_buffer_);

    thrust::fill_n(exe_pol, d_ptr, size, init);

    return d_ptr;
  }

 private:
  Buffer<elem_t> d_buffer_;
};

}  // namespace detail
}  // namespace cupynumeric
