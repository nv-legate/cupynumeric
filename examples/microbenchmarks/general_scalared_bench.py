# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
General Scalar Reduction Benchmark

Operations tested:
1. sum
2. prod
3. min
4. max
5. argmin
6. argmax

on float32, float64
"""

from microbenchmark_utilities import create_benchmark_function


# =============================================================================
# GENERAL SCALAR REDUCTION BENCHMARKS: sum, prod, min, max, argmin, argmax
# =============================================================================


def bench_scalar_red(np, timer, size, runs, warmup, dtype, func):
    """[np.sum, np.prod, np.min, np.max, np.argmin, np.argmax]"""

    def operation():
        in_arr = np.random.rand(size).astype(dtype)

        return func(in_arr)

    return create_benchmark_function(np, timer, operation, runs, warmup)()


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size):
    """Run general scalar benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    run_types = [np.float32, np.float64]
    red_types = [np.sum, np.prod, np.min, np.max, np.argmin, np.argmax]

    for dt, func in [(d, f) for d in run_types for f in red_types]:
        # scalar reductions
        suite.run_single_benchmark(
            name="scalar_red",
            bench_func=lambda: bench_scalar_red(
                np, timer, size, runs, warmup, dt, func
            ),
            size_params={"size": size},
        )
