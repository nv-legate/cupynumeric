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

#include "cupynumeric/cupynumeric_task.h"
#include "cupynumeric/unary/unary_op_util.h"

namespace cupynumeric {

struct UnaryOpArgs {
  legate::PhysicalStore in;
  legate::PhysicalStore out;
  UnaryOpCode op_code;
  std::vector<legate::Scalar> args;
};

struct MultiOutUnaryOpArgs {
  legate::PhysicalStore in;
  legate::PhysicalStore out1;
  legate::PhysicalStore out2;
  UnaryOpCode op_code;
};

class UnaryOpTask : public CuPyNumericTask<UnaryOpTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{CUPYNUMERIC_UNARY_OP};

 public:
  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

template <int DIM>
struct inner_type_dispatch_fn {
  template <typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(int point_dim, Functor f, Fnargs&&... args)
  {
    switch (point_dim) {
#if LEGATE_MAX_DIM >= 1
      case 1: {
        return f.template operator()<1, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 2
      case 2: {
        return f.template operator()<2, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 3
      case 3: {
        return f.template operator()<3, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 4
      case 4: {
        return f.template operator()<4, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 5
      case 5: {
        return f.template operator()<5, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 6
      case 6: {
        return f.template operator()<6, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 7
      case 7: {
        return f.template operator()<7, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 8
      case 8: {
        return f.template operator()<8, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
#if LEGATE_MAX_DIM >= 9
      case 9: {
        return f.template operator()<9, DIM>(std::forward<Fnargs>(args)...);
      }
#endif
      default: assert(false);
    }
    return f.template operator()<1, DIM>(std::forward<Fnargs>(args)...);
  }
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) double_dispatch(int dim, int point_dim, Functor f, Fnargs&&... args)
{
  switch (dim) {
#if LEGATE_MAX_DIM >= 1
    case 1: {
      return cupynumeric::inner_type_dispatch_fn<1>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 2
    case 2: {
      return cupynumeric::inner_type_dispatch_fn<2>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 3
    case 3: {
      return cupynumeric::inner_type_dispatch_fn<3>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 4
    case 4: {
      return cupynumeric::inner_type_dispatch_fn<4>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 5
    case 5: {
      return cupynumeric::inner_type_dispatch_fn<5>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 6
    case 6: {
      return cupynumeric::inner_type_dispatch_fn<6>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 7
    case 7: {
      return cupynumeric::inner_type_dispatch_fn<7>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 8
    case 8: {
      return cupynumeric::inner_type_dispatch_fn<8>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
#if LEGATE_MAX_DIM >= 9
    case 9: {
      return cupynumeric::inner_type_dispatch_fn<9>{}(point_dim, f, std::forward<Fnargs>(args)...);
    }
#endif
  }
  assert(false);
  return cupynumeric::inner_type_dispatch_fn<1>{}(point_dim, f, std::forward<Fnargs>(args)...);
}

}  // namespace cupynumeric
