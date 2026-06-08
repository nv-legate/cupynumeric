/* Copyright 2026 NVIDIA Corporation
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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <tuple>
#include <iostream>

namespace black_scholes {

struct Config {
  std::uint64_t benchmark{10};
  std::uint64_t N{100};
  std::uint64_t precision{32};
};

class CuPyNumericTimer {
 public:
  void start()
  {
    if (use_chrono_) {
      chrono_start_ = std::chrono::steady_clock::now();
    } else {
      start_time_ = legate::timing::measure_microseconds();
    }
  }

  [[nodiscard]] double stop()
  {
    if (use_chrono_) {
      const auto end_time = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(end_time - chrono_start_).count();
    }
    auto end_time = legate::timing::measure_microseconds();
    return static_cast<double>(end_time.value() - start_time_->value()) / 1000.0;
  }

 private:
  // In eager mode (--inline-task-launch), tasks run synchronously without Legion, so the
  // future-based legate timer cannot be used (it aborts trying to set a future on inline
  // storage). Fall back to a wall-clock timer, which is accurate because execution is synchronous.
  [[nodiscard]] static bool inline_task_launch_enabled()
  {
    const char* config = std::getenv("LEGATE_CONFIG");
    return config != nullptr &&
           std::string_view{config}.find("--inline-task-launch") != std::string_view::npos;
  }

  bool use_chrono_{inline_task_launch_enabled()};
  std::optional<legate::timing::Time> start_time_{};
  std::chrono::steady_clock::time_point chrono_start_{};
};

// Build a Scalar whose C++ value type matches the legate Type. legate::Scalar(value, type)
// reinterprets the raw bytes of `value`, so sizeof(value) must equal type.size().
[[nodiscard]] legate::Scalar sc(double value, const legate::Type& ft)
{
  if (ft == legate::float32()) {
    return legate::Scalar(static_cast<float>(value), ft);
  } else if (ft == legate::float64()) {
    return legate::Scalar(static_cast<double>(value), ft);
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

[[nodiscard]] cupynumeric::NDArray zero(const legate::Type& ft)
{
  return cupynumeric::full({1}, sc(0.0, ft));
}

[[nodiscard]] cupynumeric::NDArray one(const legate::Type& ft)
{
  return cupynumeric::full({1}, sc(1.0, ft));
}

[[nodiscard]] legate::Scalar negate(const legate::Scalar& x, const legate::Type& ft)
{
  if (ft == legate::float32()) {
    return sc(-x.value<float>(), ft);
  } else if (ft == legate::float64()) {
    return sc(-x.value<double>(), ft);
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

[[nodiscard]] legate::Scalar square_half_add(const legate::Scalar& V,
                                             const legate::Scalar& R,
                                             const legate::Type& ft)
{
  if (ft == legate::float32()) {
    float v = V.value<float>();
    float r = R.value<float>();
    return sc(v * v * 0.5 + r, ft);
  } else if (ft == legate::float64()) {
    double v = V.value<double>();
    double r = R.value<double>();
    return sc(v * v * 0.5 + r, ft);
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

[[nodiscard]] cupynumeric::NDArray generate_random(std::uint64_t N,
                                                   float min,
                                                   float max,
                                                   const legate::Type& ft)
{
  auto diff  = max - min;
  auto rands = cupynumeric::random({N}).as_type(ft);
  rands      = rands * sc(diff, ft);
  rands      = rands + sc(min, ft);
  return rands;
}

[[nodiscard]] std::tuple<cupynumeric::NDArray,
                         cupynumeric::NDArray,
                         cupynumeric::NDArray,
                         legate::Scalar,
                         legate::Scalar>
initialize(std::uint64_t N, const legate::Type& ft)
{
  auto S = generate_random(N, 5, 30, ft);
  auto X = generate_random(N, 1, 100, ft);
  auto T = generate_random(N, (float)0.25, (float)10, ft);
  auto R = sc(0.02, ft);
  auto V = sc(0.3, ft);
  return {S, X, T, R, V};
}

[[nodiscard]] cupynumeric::NDArray cnd(const cupynumeric::NDArray& d, const legate::Type& ft)
{
  auto A1       = sc(0.31938153, ft);
  auto A2       = sc(-0.356563782, ft);
  auto A3       = sc(1.781477937, ft);
  auto A4       = sc(-1.821255978, ft);
  auto A5       = sc(1.330274429, ft);
  auto RSQRT2PI = sc(0.39894228040143267793994605993438, ft);

  auto K = one(ft) / (cupynumeric::abs(d) * sc(0.2316419, ft) + one(ft));

  // Note: had to perform different ordering of operations to satisfy operator method arguments
  auto cnd = cupynumeric::exp(d * d * sc(-0.5, ft)) *
             (K * (K * (K * (K * (K * A5 + A4) + A3) + A2) + A1)) * RSQRT2PI;

  return cupynumeric::where(d > zero(ft), one(ft) - cnd, cnd);
}

[[nodiscard]] std::tuple<cupynumeric::NDArray, cupynumeric::NDArray> black_scholes(
  const cupynumeric::NDArray& S,
  const cupynumeric::NDArray& X,
  const cupynumeric::NDArray& T,
  const legate::Scalar& R,
  const legate::Scalar& V,
  const legate::Type& ft)
{
  auto sqrt_t      = cupynumeric::sqrt(T);
  auto d1          = cupynumeric::log(S / X) + (T * square_half_add(V, R, ft)) / (sqrt_t * V);
  auto d2          = d1 - sqrt_t * V;
  auto cnd_d1      = cnd(d1, ft);
  auto cnd_d2      = cnd(d2, ft);
  auto exp_rt      = cupynumeric::exp(T * (negate(R, ft)));
  auto call_result = S * cnd_d1 - X * exp_rt * cnd_d2;
  auto put_result  = X * exp_rt * (one(ft) - cnd_d2) - S * (one(ft) - cnd_d1);
  return {call_result, put_result};
}

[[nodiscard]] legate::Type get_type(const Config& config)
{
  if (config.precision == 32) {
    return legate::float32();
  } else if (config.precision == 64) {
    return legate::float64();
  } else {
    throw std::runtime_error("Unsupported precision");
  }
}

// Runs a single Black-Scholes iteration and returns the elapsed time in milliseconds.
[[nodiscard]] double run_black_scholes(const Config& config)
{
  const auto ft = get_type(config);

  CuPyNumericTimer timer{};
  auto [S, X, T, R, V] = initialize(config.N * 1000, ft);

  // Note: we are going to be only timing the computation part of Black-Scholes
  // because the data generation takes a significant amount of time even though
  // the python example times the data generation.
  timer.start();
  auto _ = black_scholes(S, X, T, R, V, ft);
  return timer.stop();
}

}  // namespace black_scholes

int main(int argc, char** argv)
{
  legate::start();

  cupynumeric::initialize(argc, argv);

  black_scholes::Config config{};

  Realm::CommandLineParser cp;
  cp.add_option_int("--benchmark", config.benchmark)
    .add_option_int("--num", config.N)
    .add_option_int("--precision", config.precision)
    .parse_command_line(argc, argv);

  std::cout << "========================================================================="
            << std::endl;
  std::cout << "Running Black-Scholes with " << config.N << "K options for " << config.benchmark
            << " runs and precision " << config.precision << std::endl;
  std::cout << "========================================================================="
            << std::endl;

  double total_time = 0.0;
  int num_runs      = config.benchmark;

  auto warmup = black_scholes::run_black_scholes(config);
  std::cout << "Warmup time: " << warmup << " ms" << std::endl;

  for (int i = 0; i < num_runs; ++i) {
    auto duration = black_scholes::run_black_scholes(config);
    total_time += duration;

    std::cout << "Run " << i + 1 << " time: " << duration << " ms" << std::endl;
  }

  if (num_runs > 1) {
    std::cout << "Average time: " << total_time / num_runs << " ms" << std::endl;
  }

  return legate::finish();
}
