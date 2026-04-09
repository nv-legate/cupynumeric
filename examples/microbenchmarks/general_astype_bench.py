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
General astype Benchmark

Operations tested:
astype()

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
# GENERAL ASTYPE BENCHMARKS: astype
# =============================================================================


# Worst-case setup keeps the int64 source array and float64 cast result live.
_ASTYPE_BYTES_PER_ELEMENT = 16


@benchmark_info(formats={"dtype": format_dtype})
def astype(np, dtype, size, runs, warmup, *, timer):
    """np.astype"""

    def operation():
        in_arr = np.random.randint(1, 1000, size=size)
        out_arr = in_arr.astype(dtype)

        return out_arr

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run general astype benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_linear_suite_size(
        size_request, bytes_per_element=_ASTYPE_BYTES_PER_ELEMENT
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    dtypes = [np.float32, np.float64]

    suite.run_timed(astype, np, dtypes, sizes, runs, warmup, timer=timer)


class AsTypeSuite(MicrobenchmarkSuite):
    name = "astype"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
