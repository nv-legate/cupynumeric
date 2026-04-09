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
    python main.py --suite advanced_indexing [--size SIZE | --memory-size 64MiB]

    # Compare with numpy backend:
    python main.py --suite advanced_indexing --package numpy
"""

from __future__ import annotations

import math

from _benchmark import MicrobenchmarkSuite, timed_loop
from _benchmark.sizing import SizeRequest, clamp, resolve_linear_suite_size


# =============================================================================
# OPTIMIZED PATH BENCHMARKS
# =============================================================================


# Model one float64 data array plus boolean/int index structures.
_ADVANCED_INDEXING_BYTES_PER_ELEMENT = 10


def _describe_size(size: int) -> list[str]:
    n = max(1, math.isqrt(size))
    return [f"resolved_1d_elements: {size:,}", f"resolved_2d_shape: {n} x {n}"]


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

    return timed_loop(operation, timer, runs, warmup) / runs


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

    return timed_loop(operation, timer, runs, warmup) / runs


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

    return timed_loop(operation, timer, runs, warmup) / runs


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

    return timed_loop(operation, timer, runs, warmup) / runs


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

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run optimized advanced indexing benchmarks (NO ADVANCED_INDEXING task)."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_ADVANCED_INDEXING_BYTES_PER_ELEMENT,
        describe_size=_describe_size,
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    def arg_gen_take_1d():
        for size in sizes:
            # Derived sizes
            n = max(1, math.isqrt(size))
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, size, num_indices, runs, warmup)

    def arg_gen_generic():
        for size in sizes:
            # Derived sizes
            n = max(1, math.isqrt(size))
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, n, num_indices, runs, warmup)

    suite.run_timed(putmask_scalar, np, sizes, runs, warmup, timer=timer)
    suite.run_timed_with_generator(
        None, take_1d, arg_gen_take_1d(), timer=timer
    )
    for func in [einsum_2d, take_2d, take_along_axis]:
        suite.run_timed_with_generator(
            None, func, arg_gen_generic(), timer=timer
        )


class FastAdvancedIndexingSuite(MicrobenchmarkSuite):
    name = "advanced_indexing"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
