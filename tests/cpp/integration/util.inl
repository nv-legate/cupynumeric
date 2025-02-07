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

#include <iomanip>

namespace {

template <typename T, typename U = void>
struct has_operator_left_shift : std::false_type {};

template <typename T>
struct has_operator_left_shift<T, std::void_t<decltype(std::cout << std::declval<T>())>>
  : std::true_type {};

template <typename T>
constexpr bool has_operator_left_shift_v = has_operator_left_shift<T>::value;

template <typename T>
std::stringstream& print_value(std::stringstream& ss, T value)
{
  if constexpr (has_operator_left_shift_v<T>) {
    ss << value;
  }
  return ss;
}

template <typename T, int32_t DIM>
std::string to_string(legate::AccessorRO<T, DIM> acc,
                      const std::vector<uint64_t>& shape,
                      legate::Rect<DIM> rect)
{
  std::stringstream ss;
  auto size = static_cast<int32_t>(shape.size());

  auto count = 0;
  auto pro   = 1;
  std::vector<size_t> item_count;
  for (int32_t i = size - 1; i >= 0; --i) {
    pro *= shape[i];
    item_count.push_back(pro);
  }

  auto print_brackets_in_start_end = [&](bool start) {
    if (start) {
      for (int32_t i = 0; i < size; ++i) {
        ss << "[";
      }
    } else {
      for (int32_t i = 0; i < size; ++i) {
        ss << "]";
      }
    }
  };

  auto print_brackets_in_middle = [&]() -> bool {
    for (int32_t i = size - 1; i >= 0; --i) {
      if ((count % item_count[i]) == 0) {
        for (int32_t j = i; j >= 0; --j) {
          ss << "]";
        }
        ss << ",\n";
        for (int32_t j = i; j >= 0; --j) {
          ss << "[";
        }
        return true;
      }
    }
    return false;
  };

  print_brackets_in_start_end(true);
  for (legate::PointInRectIterator<DIM> itr(rect, false); itr.valid(); ++itr) {
    if (count > 0) {
      if (!print_brackets_in_middle()) {
        ss << ",";
      }
    }
    ss << std::setw(9) << std::setprecision(3);
    print_value(ss, acc[*itr]);
    count += 1;
  }
  print_brackets_in_start_end(false);

  return ss.str();
}

template <typename T, int32_t DIM>
void check_array_eq(legate::AccessorRO<T, DIM> acc,
                    T* values_ptr,
                    const std::vector<uint64_t>& shape,
                    legate::Rect<DIM> rect)
{
  auto index = 0;
  for (legate::PointInRectIterator<DIM> itr(rect, false); itr.valid(); ++itr) {
    auto q      = *itr;
    auto value  = acc[q];
    auto expect = values_ptr[index++];
    if (value != expect) {
      std::stringstream ss;
      auto size = shape.size();
      ss << "size: " << size << "\n";
      ss << std::left << std::setprecision(3);
      ss << std::setw(13) << "Array value: " << std::setw(10);
      print_value(ss, value) << ", ";
      ss << std::setw(16) << "Expected value: " << std::setw(10);
      print_value(ss, expect) << ", ";
      if (size > 0) {
        ss << std::setw(8) << "index: [";
        for (uint32_t i = 0; i < size - 1; ++i) {
          ss << q[i] << ",";
        }
        ss << q[size - 1] << "]\n";
      }
      FAIL() << ss.str();
    }
  }
}

template <typename T, int32_t DIM>
struct print_fn {
  void operator()(legate::AccessorRO<T, DIM> acc,
                  const std::vector<uint64_t>& shape,
                  legate::Rect<DIM> rect)
  {
    std::cerr << to_string<T, DIM>(acc, shape, rect) << std::endl;
  }
};

template <typename T, int32_t DIM>
struct check_array_eq_fn {
  void operator()(legate::AccessorRO<T, DIM> acc,
                  T* values_ptr,
                  const std::vector<uint64_t>& shape,
                  legate::Rect<DIM> rect)
  {
    check_array_eq<T, DIM>(acc, values_ptr, shape, rect);
  }
};

template <typename T, int32_t DIM>
struct assign_array_fn {
  void operator()(legate::AccessorWO<T, DIM> acc, T* values_ptr, legate::Rect<DIM> rect)
  {
    auto index = 0;
    for (legate::PointInRectIterator<DIM> itr(rect, false); itr.valid(); ++itr) {
      acc[*itr] = values_ptr[index++];
    }
  }
};

template <typename T, int32_t DIM>
struct copy_array_fn {
  void operator()(legate::AccessorRO<T, DIM> acc, T* values_ptr, legate::Rect<DIM> rect)
  {
    auto index = 0;
    for (legate::PointInRectIterator<DIM> itr(rect, false); itr.valid(); ++itr) {
      values_ptr[index++] = acc[*itr];
    }
  }
};

template <typename T, int32_t DIM>
void print_array(cupynumeric::NDArray array)
{
  auto acc            = array.get_read_accessor<T, DIM>();
  auto& shape         = array.shape();
  auto logical_store  = array.get_store();
  auto physical_store = logical_store.get_physical_store();
  auto rect           = physical_store.shape<DIM>();
  print_fn<T, DIM>()(acc, shape, rect);
}

template <typename T, int32_t DIM>
void check_array_eq(cupynumeric::NDArray array, T* values_ptr, size_t length)
{
  assert(array.size() == length);
  if (length == 0) {
    return;
  }
  assert(values_ptr != nullptr);
  auto acc            = array.get_read_accessor<T, DIM>();
  auto& shape         = array.shape();
  auto logical_store  = array.get_store();
  auto physical_store = logical_store.get_physical_store();
  auto rect           = physical_store.shape<DIM>();
  check_array_eq_fn<T, DIM>()(acc, values_ptr, shape, rect);
}

template <typename T, int32_t DIM>
void assign_values_to_array(cupynumeric::NDArray array, T* values_ptr, size_t length)
{
  assert(array.size() == length);
  if (length == 0) {
    return;
  }
  assert(values_ptr != nullptr);
  auto acc            = array.get_write_accessor<T, DIM>();
  auto logical_store  = array.get_store();
  auto physical_store = logical_store.get_physical_store();
  auto rect           = physical_store.shape<DIM>();
  assign_array_fn<T, DIM>()(acc, values_ptr, rect);
}

template <typename T, int32_t DIM>
std::vector<T> assign_array_to_values(cupynumeric::NDArray array)
{
  std::vector<T> result(array.size());
  if (array.size() > 0) {
    T* values_ptr = result.data();
    assert(values_ptr != nullptr);
    auto acc            = array.get_read_accessor<T, DIM>();
    auto logical_store  = array.get_store();
    auto physical_store = logical_store.get_physical_store();
    auto rect           = physical_store.shape<DIM>();
    copy_array_fn<T, DIM>()(acc, values_ptr, rect);
  }
  return std::move(result);
}

template <typename T, int32_t DIM>
void check_array_eq(cupynumeric::NDArray array1, cupynumeric::NDArray array2)
{
  assert(array1.size() == array2.size());
  if (array1.size() == 0) {
    return;
  }

  std::vector<T> data2 = assign_array_to_values<T, DIM>(array2);
  check_array_eq<T, DIM>(array1, data2.data(), data2.size());
}

}  // namespace
