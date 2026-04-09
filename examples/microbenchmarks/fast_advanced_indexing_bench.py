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

Tests ONLY indexing paths that are already optimized (avoid gather/scatter):

1. putmask path: Scalar assignment to boolean mask (a[mask] = scalar)
   - Uses putmask() operation directly

2. einsum path: Row-axis integer array GET for 2D+ arrays (a[indices, :])
   - Converts to einsum tensor contraction (cuTENSOR)

3. boolean GET (1D): a[bool_mask] — ADVANCED_INDEXING task, no gather
   - Single boolean array at position 0; result returned directly

4. boolean GET (2D): a[mask_2d] — ADVANCED_INDEXING task, no gather
   - Full-array boolean mask; same path as 1D case

5. row selection (2D): a[row_indices] — einsum path (mask_axis=0, ndim=2)
   - Single integer array with implicit trailing slice(None)

6. column GET (2D): a[:, indices] — einsum path (mask_axis=1, ndim=2)
   - Single integer array on non-leading axis

These optimizations bypass gather/scatter entirely.

For proposed (not-yet-implemented) optimizations and their current slow-path
baselines, see indexing_opt_targets_bench.py (opts. 1-3, 6, 8).

For benchmarks that DO use gather/scatter (Copy operations),
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


def boolean_get_1d(np, size, runs, warmup, *, timer):
    """
    Single boolean array GET (1D).
    Path: a[bool_mask] — ADVANCED_INDEXING task, copy_needed=False, no gather called.
    """
    a = np.random.random(size)
    mask = a > 0.5

    def operation():
        return a[mask]

    return timed_loop(operation, timer, runs, warmup) / runs


def boolean_get_2d(np, n, runs, warmup, *, timer):
    """
    Full-array boolean mask GET (2D).
    Path: a[mask_2d] — single boolean array → ADVANCED_INDEXING task, no gather.
    """
    a = np.random.random((n, n))
    mask = a > 0.5

    def operation():
        return a[mask]

    return timed_loop(operation, timer, runs, warmup) / runs


def row_select_2d(np, n, num_rows, runs, warmup, *, timer):
    """
    2D row selection: a[row_indices].
    Path: computed_key=(row_indices,) → einsum check passes (1 array, ndim=2<5)
          → einsum path, no gather.
    """
    a = np.random.random((n, n))
    row_indices = np.random.randint(0, n, num_rows)

    def operation():
        return a[row_indices]

    return timed_loop(operation, timer, runs, warmup) / runs


def array_get_col_2d(np, n, num_indices, runs, warmup, *, timer):
    """
    Column-wise integer array GET: a[:, indices].
    Path: computed_key=(slice(None), indices) → einsum check passes (mask_axis=1, ndim=2<5)
          → einsum path, no gather.
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[:, indices]

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run optimized advanced indexing benchmarks (NO gather/scatter)."""
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

    ns = [max(1, math.isqrt(size)) for size in sizes]

    def arg_gen_2d():
        for size in sizes:
            n = max(1, math.isqrt(size))
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, n, num_indices, runs, warmup)

    # putmask: a[bool_mask] = scalar
    suite.run_timed(putmask_scalar, np, sizes, runs, warmup, timer=timer)

    # einsum path: a[indices, :] (2D row GET)
    suite.run_timed_with_generator(None, einsum_2d, arg_gen_2d(), timer=timer)

    # single boolean array GET (no gather)
    suite.run_timed(boolean_get_1d, np, sizes, runs, warmup, timer=timer)
    suite.run_timed(boolean_get_2d, np, ns, runs, warmup, timer=timer)

    # einsum-routed integer GET
    suite.run_timed_with_generator(
        None, row_select_2d, arg_gen_2d(), timer=timer
    )
    suite.run_timed_with_generator(
        None, array_get_col_2d, arg_gen_2d(), timer=timer
    )


class FastAdvancedIndexingSuite(MicrobenchmarkSuite):
    name = "advanced_indexing"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
