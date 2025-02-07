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

#include "common_utils.h"

using namespace cupynumeric;
using Code = legate::Type::Code;

namespace {

const size_t DIM = 4;
std::vector<std::vector<uint64_t>> shape_list{{0},
                                              {1},
                                              {DIM},
                                              {0, 1},
                                              {1, 0},
                                              {1, 1},
                                              {1, DIM},
                                              {DIM, 1},
                                              {DIM, DIM},
                                              {1, 0, 0},
                                              {1, 1, 0},
                                              {1, 0, 1},
                                              {1, 1, 1},
                                              {DIM, 1, 1},
                                              {1, DIM, 1},
                                              {1, 1, DIM},
                                              {DIM, DIM, DIM}};

std::vector<Code> code_list{Code::BOOL,
                            Code::INT8,
                            Code::INT16,
                            Code::INT32,
                            Code::INT64,
                            Code::UINT8,
                            Code::UINT16,
                            Code::UINT32,
                            Code::UINT64,
                            Code::FLOAT32,
                            Code::FLOAT64,
                            Code::COMPLEX64,
                            Code::COMPLEX128};

template <Code CODE>
void _test(std::vector<uint64_t> shape)
{
  auto x    = zeros(shape, legate::primitive_type(CODE));
  using VAL = legate::type_of<CODE>;
  std::vector<VAL> x_gt(x.size());
  check_array(x, x_gt, shape);
  // debug_array(x, false);
}

TEST(Zeros, test_basic_dtype)
{
  for (auto code : code_list) {
    for (auto shape : shape_list) {
      switch (code) {
        case Code::BOOL: _test<Code::BOOL>(shape); break;
        case Code::INT8: _test<Code::INT8>(shape); break;
        case Code::INT16: _test<Code::INT16>(shape); break;
        case Code::INT32: _test<Code::INT32>(shape); break;
        case Code::INT64: _test<Code::INT64>(shape); break;
        case Code::UINT8: _test<Code::UINT8>(shape); break;
        case Code::UINT16: _test<Code::UINT16>(shape); break;
        case Code::UINT32: _test<Code::UINT32>(shape); break;
        case Code::UINT64: _test<Code::UINT64>(shape); break;
        case Code::FLOAT32: _test<Code::FLOAT32>(shape); break;
        case Code::FLOAT64: _test<Code::FLOAT64>(shape); break;
        case Code::COMPLEX64: _test<Code::COMPLEX64>(shape); break;
        case Code::COMPLEX128: _test<Code::COMPLEX128>(shape); break;
        default: FAIL() << "Unsupported data types."; break;
      }
    }
  }
}

TEST(Zeros, test_ndim)
{
  std::vector<uint64_t> shape;
  for (int32_t ndim = 1; ndim <= LEGATE_MAX_DIM; ++ndim) {
    shape.push_back(ndim);
    _test<Code::BOOL>(shape);
    _test<Code::INT32>(shape);
    _test<Code::FLOAT32>(shape);
    _test<Code::COMPLEX64>(shape);
  }
}

TEST(Zeros, test_invalid_type)
{
  EXPECT_THROW(zeros({2, 2}, legate::primitive_type(Code::FIXED_ARRAY)), std::invalid_argument);
}

}  // namespace
