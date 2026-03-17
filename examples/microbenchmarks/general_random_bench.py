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
General Random Generation Benchmark

Operations tested:
1. discrete uniform random.randint()
2. uniform with four bigenerator types, for float32, float64
"""

from _benchmark import (
    MicrobenchmarkSuite,
    timed_loop,
    benchmark_info,
    format_dtype,
)
from _benchmark.sizing import SizeRequest, resolve_linear_suite_size

# =============================================================================
# GENERAL RANDOM GENERATION BENCHMARKS: Uniform distribution only
# =============================================================================


# One int64/float64 output array dominates the worst-case working set.
_RANDOM_BYTES_PER_ELEMENT = 8


def randint(np, size, runs, warmup, *, timer):
    """random.randint(low, high, size)."""

    def operation():
        return np.random.randint(1, 1000, size=size)

    return timed_loop(operation, timer, runs, warmup)


@benchmark_info(
    formats={"dtype": format_dtype, "bitgenerator_type": lambda x: x.__name__}
)
def bitgenerator(
    np,
    size,
    runs,
    warmup,
    bitgenerator_type,
    dtype,
    *,
    timer,
    uniform_takes_dtype,
):
    """bitgenerator operation."""

    bitgen = bitgenerator_type(seed=1729)
    gen = np.random.Generator(bitgen)

    def operation():
        low = 1.414
        high = 3.14
        if uniform_takes_dtype:
            return gen.uniform(low, high, size=size, dtype=dtype)
        else:
            return gen.uniform(low, high, size=size)

    return timed_loop(operation, timer, runs, warmup)


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run general random generators benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    size, resolution = resolve_linear_suite_size(
        size_request, bytes_per_element=_RANDOM_BYTES_PER_ELEMENT
    )
    if resolution is not None:
        suite.print_size_resolution(resolution)

    dtypes = [np.float32, np.float64]
    uniform_takes_dtype = True
    bitgen_types = []

    match np.__name__:
        case "cupynumeric":
            bitgen_types = [
                np.random.XORWOW,
                np.random.MRG32k3a,
                np.random.PHILOX4_32_10,
            ]
        case "cupy":
            bitgen_types = [
                np.random.XORWOW,
                np.random.MRG32k3a,
                np.random.Philox4x3210,
            ]
        case "numpy":
            bitgen_types = [
                np.random.MT19937,
                np.random.PCG64,
                np.random.PCG64DXSM,
                np.random.Philox,
                np.random.SFC64,
            ]
            dtypes = [np.float64]
            uniform_takes_dtype = False
        case _:
            assert False, f"Unexpected package: {np.__name__}"

    suite.run_timed(randint, np, size, runs, warmup, timer=timer)
    suite.run_timed(
        bitgenerator,
        np,
        size,
        runs,
        warmup,
        bitgen_types,
        dtypes,
        timer=timer,
        uniform_takes_dtype=uniform_takes_dtype,
    )


class RandomSuite(MicrobenchmarkSuite):
    name = "random"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
