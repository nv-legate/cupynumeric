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

#include <legate.h>
#include <legate/timing/timing.h>
#include <cupynumeric.h>
#include <realm/cmdline.h>

#include <cstdint>
#include <cstdio>
#include <tuple>

namespace gemm {

struct Config {
  bool timing{false};
  std::int32_t iter{100};
  std::int32_t warmup{5};
  std::uint64_t N{100};
};

[[nodiscard]] std::tuple<cupynumeric::NDArray, cupynumeric::NDArray, cupynumeric::NDArray>
initialize(std::uint64_t N, const legate::Type& ft)
{
  auto A = cupynumeric::random({N, N}).as_type(ft);
  auto B = cupynumeric::random({N, N}).as_type(ft);
  auto C = cupynumeric::zeros({N, N}, ft);
  return {A, B, C};
}

[[nodiscard]] std::size_t total_flops(std::uint64_t M, std::uint64_t N, std::uint64_t K)
{
  return M * N * (2 * K - 1);
}

[[nodiscard]] std::size_t total_space(std::uint64_t M,
                                      std::uint64_t N,
                                      std::uint64_t K,
                                      const legate::Type& ft)
{
  return (M * N + M * K + K * N) * ft.size();
}

void run_gemm(const Config& config)
{
  const auto ft = legate::float32();
  const auto N  = config.N;
  std::printf("Problem Size:     M=%lu N=%lu K=%lu\n", N, N, N);
  std::printf("Total Iterations: %d\n", config.iter);
  const auto flops = total_flops(N, N, N);
  std::printf("Total Flops:      %lf GFLOPS/iter\n", flops / 1e9);
  const auto space = total_space(N, N, N, ft);
  std::printf("Total Size:       %lf MB\n", space / 1e6);
  auto [A, B, C] = initialize(config.N, legate::float32());

  auto start    = legate::timing::measure_microseconds();
  auto max_iter = config.iter + config.warmup;
  for (int32_t iter = 0; iter < max_iter; ++iter) {
    if (iter == config.warmup) {
      start = legate::timing::measure_microseconds();
    }
    C.dot(A, B);
    // We need to rotate the matrices to keep Legate honest
    // about moving data so it can't just duplicate A and B
    // on the first iteration and reuse them, this means
    // that A, B, C all need to be square
    A, B, C = B, C, A;
  }
  auto stop = legate::timing::measure_microseconds();

  const auto total = (stop.value() - start.value()) / 1e3;
  std::printf("Elapsed Time:     %lf ms\n", total);
  const auto average = total / config.iter;
  std::printf("Average GEMM:     %lf ms\n", average);
  std::printf("FLOPS/s:          %lf GFLOPS/s\n", flops / (average * 1e6));
}

}  // namespace gemm

int main(int argc, char** argv)
{
  legate::start();

  cupynumeric::initialize(argc, argv);

  gemm::Config config{};

  Realm::CommandLineParser cp;
  cp.add_option_int("--iter", config.iter)
    .add_option_int("--warmup", config.warmup)
    .add_option_int("--num", config.N)
    .add_option_bool("--time", config.timing)
    .parse_command_line(argc, argv);

  gemm::run_gemm(config);

  return legate::finish();
}
