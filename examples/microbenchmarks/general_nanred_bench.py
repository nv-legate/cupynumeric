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
General NaN Reduction Benchmark

Operations tested:
1. nansum
2. nanmean

on float32, float64
"""

from microbenchmark_utilities import create_benchmark_function


# =============================================================================
# GENERAL NaN REDUCTION BENCHMARKS: nansum(), nanmean()
# =============================================================================


def bench_nan_red(np, timer, size, runs, warmup, dtype, func):
    """[np.nansum, np.nanmean](input_with_half_nans)."""

    def operation():
        in_arr = np.empty(shape=(size,), dtype=dtype)
        half_sz = size // 2

        in_arr[0:half_sz] = np.random.rand(half_sz)
        in_arr[half_sz:size] = np.nan

        return func(in_arr)

    return create_benchmark_function(np, timer, operation, runs, warmup)()


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size):
    """Run general nansum(), nanmean() benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    run_types = [np.float32, np.float64]
    red_types = [np.nansum, np.nanmean]

    for dt, func in [(d, f) for d in run_types for f in red_types]:
        # nansum, nanmean
        suite.run_single_benchmark(
            name="nansum",
            bench_func=lambda: bench_nan_red(
                np, timer, size, runs, warmup, dt, func
            ),
            size_params={"size": size},
        )
