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

#include "legate.h"
#include "cupynumeric.h"
#include "realm/cmdline.h"

#include <iomanip>

namespace stencil {

using cupynumeric::open;
using cupynumeric::slice;

struct Config {
  bool timing{false};
  int32_t iter{100};
  int32_t warmup{5};
  uint64_t N{100};
};

void print_array(cupynumeric::NDArray array)
{
  auto acc         = array.get_read_accessor<double, 2>();
  const auto shape = array.shape();
  std::stringstream ss;
  for (uint32_t i = 0; i < shape[0]; ++i) {
    for (uint32_t j = 0; j < shape[0]; ++j) {
      if (j > 0) {
        ss << " ";
      }
      ss << std::setw(8) << std::setprecision(5) << acc[i][j];
    }
    ss << std::endl;
  }
  std::cerr << std::move(ss).str();
}

cupynumeric::NDArray initialize(uint64_t N)
{
  auto grid = cupynumeric::zeros({N + 2, N + 2});
  grid[{slice(), slice(0, 1)}].assign(legate::Scalar{-273.15});
  grid[{slice(), slice(-1, open)}].assign(legate::Scalar{-273.15});
  grid[{slice(-1, open), slice()}].assign(legate::Scalar{-273.15});
  grid[{slice(0, 1), slice()}].assign(legate::Scalar{40.0});
  return grid;
}

void stencil(const Config& config)
{
  auto grid = initialize(config.N);

  auto center = grid[{slice(1, -1), slice(1, -1)}];
  auto north  = grid[{slice(0, -2), slice(1, -1)}];
  auto east   = grid[{slice(1, -1), slice(2, open)}];
  auto west   = grid[{slice(1, -1), slice(0, -2)}];
  auto south  = grid[{slice(2, open), slice{1, -1}}];

  auto max_iter = config.iter + config.warmup;
  for (int32_t iter = 0; iter < max_iter; ++iter) {
    auto average = center + north + east + west + south;
    auto work    = average * legate::Scalar(double(0.2));
    center.assign(work);
  };
}

}  // namespace stencil

int main(int argc, char** argv)
{
  auto result = legate::start(argc, argv);
  assert(result == 0);

  cupynumeric::initialize(argc, argv);

  stencil::Config config{};

  Realm::CommandLineParser cp;
  cp.add_option_int("--iter", config.iter)
    .add_option_int("--warmup", config.warmup)
    .add_option_int("--num", config.N)
    .add_option_bool("--time", config.timing)
    .parse_command_line(argc, argv);

  stencil::stencil(config);

  return legate::finish();
}
