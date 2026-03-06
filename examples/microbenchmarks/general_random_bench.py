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

from microbenchmark_utilities import create_benchmark_function


# =============================================================================
# GENERAL RANDOM GENERATION BENCHMARKS: Uniform distribution only
# =============================================================================


def bench_randint(np, timer, size, runs, warmup):
    """random.randint(low, high, size)."""

    def operation():
        return np.random.randint(1, 1000, size=size)

    return create_benchmark_function(np, timer, operation, runs, warmup)()


def bench_bitgenerator(
    np,
    timer,
    size,
    runs,
    warmup,
    bitgenerator_type,
    dtype,
    *,
    uniform_takes_dtype=True,
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

    return create_benchmark_function(np, timer, operation, runs, warmup)()


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size):
    """Run general random generators benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    # randint
    suite.run_single_benchmark(
        name="randint",
        bench_func=lambda: bench_randint(np, timer, size, runs, warmup),
        size_params={"size": size},
    )

    run_types = [np.float32, np.float64]
    uniform_takes_dtype = True
    if np.__name__ == "cupynumeric":
        Bitgenerator_Types = [
            np.random.XORWOW,
            np.random.MRG32k3a,
            np.random.PHILOX4_32_10,
        ]
    elif np.__name__ == "cupy":
        Bitgenerator_Types = [
            np.random.XORWOW,
            np.random.MRG32k3a,
            np.random.Philox4x3210,
        ]
    elif np.__name__ == "numpy":
        Bitgenerator_Types = [
            np.random.MT19937,
            np.random.PCG64,
            np.random.PCG64DXSM,
            np.random.Philox,
            np.random.SFC64,
        ]
        run_types = [np.float64]
        uniform_takes_dtype = False
    else:
        assert False, f"Unexpected package: {np.__name__}"

    for bitg, dt in [(b, t) for b in Bitgenerator_Types for t in run_types]:
        suite.run_single_benchmark(
            name="bitgen",
            bench_func=lambda: bench_bitgenerator(
                np,
                timer,
                size,
                runs,
                warmup,
                bitg,
                dt,
                uniform_takes_dtype=uniform_takes_dtype,
            ),
            size_params={"size": size, "generator": bitg, "dtype": dt},
        )
