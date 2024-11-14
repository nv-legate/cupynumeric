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

#include "legate.h"
#include "cupynumeric/typedefs.h"
#include "cupynumeric/cupynumeric_c.h"

namespace cupynumeric {

enum class VariantKind : int {
  CPU = 0,
  OMP = 1,
  GPU = 2,
};

struct CuPyNumericRegistrar {
  static legate::TaskRegistrar& get_registrar();
};

template <typename T>
struct CuPyNumericTask : public legate::LegateTask<T> {
  using Registrar = CuPyNumericRegistrar;
};

}  // namespace cupynumeric
