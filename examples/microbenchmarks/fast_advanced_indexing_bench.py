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
Advanced Indexing OPTIMIZED Paths Benchmark

Tests ONLY optimized paths that AVOID ADVANCED_INDEXING task:

1. putmask path: Scalar assignment to boolean mask (a[mask] = scalar)
   - Uses putmask() operation directly

2. einsum path: Integer array indexing for 2D+ arrays (a[indices, :])
   - Converts to einsum tensor contraction (cuTENSOR)

3. take_task path: Take operation (np.take(a, indices, axis=0))
   - Uses specialized TAKE task (works for any dimension)

These optimizations bypass the general ADVANCED_INDEXING task entirely.

For benchmarks that DO use ADVANCED_INDEXING task + Copy operations,
see general_indexing_bench.py

Usage:
    Run via main.py:
    python main.py --suite advanced_indexing [--size SIZE] [--runs RUNS]

    # Compare with numpy backend:
    python main.py --suite advanced_indexing --package numpy
"""

from _benchmark import timed_loop, MicrobenchmarkSuite


# =============================================================================
# OPTIMIZED PATH BENCHMARKS
# =============================================================================


def putmask_scalar(np, size, runs, warmup, *, timer):
    """
    Test putmask optimization for scalar assignment to boolean mask.
    Path: a[bool_mask] = scalar
    Optimization: Uses putmask() operation instead of ADVANCED_INDEXING task
    """
    a = np.random.random(size)
    mask = a > 0.5

    def operation():
        a[mask] = 999.0

    return timed_loop(operation, timer, runs, warmup)


def einsum_2d(np, n, num_indices, runs, warmup, *, timer):
    """
    Test einsum optimization for integer array indexing (2D case).
    Path: a[integer_indices, :]
    Optimization: Converts to einsum tensor contraction (uses cuTENSOR)
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[indices, :]

    return timed_loop(operation, timer, runs, warmup)


def take_1d(np, size, num_indices, runs, warmup, *, timer):
    """
    Test TAKE task optimization (1D case).
    Path: np.take(a, indices, axis=0)
    Optimization: Uses TAKE task (optimized, no ADVANCED_INDEXING)
    """
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)

    def operation():
        return np.take(a, indices, axis=0)

    return timed_loop(operation, timer, runs, warmup)


def take_2d(np, n, num_indices, runs, warmup, *, timer):
    """
    Test TAKE task optimization (2D case).
    Path: np.take(a, indices, axis=0)
    Optimization: Uses TAKE task along specific axis
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return np.take(a, indices, axis=0)

    return timed_loop(operation, timer, runs, warmup)


def take_along_axis(np, n, num_indices, runs, warmup, *, timer):
    """
    Test take_along_axis optimization.
    Path: np.take_along_axis(a, indices, axis=0)
    Optimization: Specialized TAKE task for aligned take operation
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, (num_indices, n))

    def operation():
        return np.take_along_axis(a, indices, axis=0)

    return timed_loop(operation, timer, runs, warmup)


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size):
    """Run optimized advanced indexing benchmarks (NO ADVANCED_INDEXING task)."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    # Derived sizes
    n = int(size**0.5)  # For 2D arrays
    num_indices = min(1000, n // 10)  # Number of indices to select

    suite.run_timed(putmask_scalar, np, size, runs, warmup, timer=timer)

    args = (np, size, num_indices, runs, warmup)
    kwargs = {"timer": timer}

    funcs = [einsum_2d, take_1d, take_2d, take_along_axis]

    for f in funcs:
        suite.run_timed(f, *args, **kwargs)


class FastAdvancedIndexingSuite(MicrobenchmarkSuite):
    name = "advanced_indexing"

    def run_suite(self, size):
        run_benchmarks(self, size)
