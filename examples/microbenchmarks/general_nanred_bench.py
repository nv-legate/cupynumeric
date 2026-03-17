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

from _benchmark import (
    MicrobenchmarkSuite,
    benchmark_info,
    timed_loop,
    format_dtype,
)
from _benchmark.sizing import SizeRequest, resolve_linear_suite_size


# =============================================================================
# GENERAL NaN REDUCTION BENCHMARKS: nansum(), nanmean()
# =============================================================================


# One float64 input plus a half-size float64 random fill buffer.
_NANRED_BYTES_PER_ELEMENT = 12


@benchmark_info(formats={"dtype": format_dtype, "func": lambda f: f.__name__})
def nan_red(np, func, dtype, size, runs, warmup, *, timer):
    """[np.nansum, np.nanmean](input_with_half_nans)."""

    def operation():
        in_arr = np.empty(shape=(size,), dtype=dtype)
        half_sz = size // 2

        in_arr[0:half_sz] = np.random.rand(half_sz)
        in_arr[half_sz:size] = np.nan

        return func(in_arr)

    return timed_loop(operation, timer, runs, warmup)


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run general nansum(), nanmean() benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    size, resolution = resolve_linear_suite_size(
        size_request, bytes_per_element=_NANRED_BYTES_PER_ELEMENT
    )
    if resolution is not None:
        suite.print_size_resolution(resolution)

    dtypes = [np.float32, np.float64]
    red_types = [np.nansum, np.nanmean]

    suite.run_timed(
        nan_red, np, red_types, dtypes, size, runs, warmup, timer=timer
    )


class NanRedSuite(MicrobenchmarkSuite):
    name = "nanred"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
