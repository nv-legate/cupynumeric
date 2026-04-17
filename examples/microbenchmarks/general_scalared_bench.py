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

from _benchmark import (
    MicrobenchmarkSuite,
    benchmark_info,
    timed_loop,
    format_dtype,
)
from _benchmark.sizing import SizeRequest, resolve_linear_suite_size


# =============================================================================
# GENERAL SCALAR REDUCTION BENCHMARKS: sum, prod, min, max, argmin, argmax
# =============================================================================


# Model the random float64 source plus a same-size reduction input buffer.
_SCALARRED_BYTES_PER_ELEMENT = 16


@benchmark_info(formats={"dtype": format_dtype, "func": lambda f: f.__name__})
def scalar_red(np, func, dtype, size, runs, warmup, *, timer):
    """[np.sum, np.prod, np.min, np.max, np.argmin, np.argmax]"""

    in_arr = np.random.rand(size).astype(dtype)

    def operation():
        return func(in_arr)

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run general scalar benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_linear_suite_size(
        size_request, bytes_per_element=_SCALARRED_BYTES_PER_ELEMENT
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    dtypes = [np.float32, np.float64]
    red_types = [np.sum, np.prod, np.min, np.max, np.argmin, np.argmax]

    suite.run_timed(
        scalar_red, np, red_types, dtypes, sizes, runs, warmup, timer=timer
    )


class ScalarRedSuite(MicrobenchmarkSuite):
    name = "scalared"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
