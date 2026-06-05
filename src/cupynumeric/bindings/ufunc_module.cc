/*
 * Copyright 2026 NVIDIA Corporation
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
 */

#include "cupynumeric/ufunc/native_array_handle.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(ufunc, module)
{
  module.doc() = "Private cuPyNumeric native ufunc extension.";
  nb::class_<cupynumeric::ufunc::NativeArrayHandle>(module, "_NativeArrayHandle")
    .def("dim", &cupynumeric::ufunc::NativeArrayHandle::dim)
    .def("shape", &cupynumeric::ufunc::NativeArrayHandle::shape)
    .def("writeable", &cupynumeric::ufunc::NativeArrayHandle::writeable)
    .def("copy_store_to_raw_handle",
         &cupynumeric::ufunc::NativeArrayHandle::copy_store_to_raw_handle);
  module.def("_is_available", []() { return true; });
  module.def("_native_array_handle_kind", []() { return "NativeArrayHandle"; });
  module.def("_native_array_handle_capabilities", []() {
    return nb::make_tuple("shape",
                          "dtype",
                          "writeability",
                          "store",
                          "result_allocation",
                          "conversion",
                          "task_launch",
                          "ndarray_extraction",
                          "ndarray_wrapping");
  });
  module.def("_from_store_raw_handle",
             &cupynumeric::ufunc::NativeArrayHandle::from_store_raw_handle,
             nb::arg("raw_handle"),
             nb::arg("writeable"));
}
