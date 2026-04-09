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
Advanced Indexing Optimization Targets Benchmark

All cases currently go through the slow gather/scatter path; each has a
concrete suggestion for how it could potentially be optimized.

Run this suite to measure the current (pre-optimization) baseline for each
case, and re-run after each implementation to confirm improvement.

Bool non-scalar SET — a[bool_mask] = array:
    today: ADVANCED_INDEXING task + scatter
    could potentially replace with: single-pass fused scatter task (single-GPU)

Integer GET — single array on non-leading axis or high-dimensional arrays:
    a[:, indices]  2D     →  today: ZIP + gather
    a[indices, ...]  5D   →  today: ZIP + gather (einsum skipped at ndim >= 5)
    could potentially replace with: TAKE task for both cases

Integer SET — a[indices] = v and a[:, indices] = v:
    today: ZIP + scatter
    could potentially replace with: PUT task (symmetric to TAKE, partitions
    along k, broadcasts values and indices)

np.put and np.put_along_axis benchmarks are in put_bench.py (suite name: put).

Flat indexing GET (SET via a.flat[array]=v unsupported in cupynumeric):
    a.flat[indices]  (GET)  →  today: WRAP + gather
    could potentially replace with: direct copy / reshape (no gather needed)

Boolean in mixed-key position:
    a[bool_mask, :]   GET  →  today: nonzero + ZIP + gather
    a[:, bool_mask]   GET  →  today: nonzero + ZIP + gather
    a[:, bool_mask] = v    →  today: nonzero + ZIP + scatter
    could potentially replace with: relax _prepare_boolean_array_indexing to
    allow slice(None) co-keys, use transpose + ADVANCED_INDEXING task instead

Usage:
    python main.py --suite indexing_opt_targets [--size SIZE | --memory-size 64MiB]

    # Compare with numpy backend:
    python main.py --suite indexing_opt_targets --package numpy
"""

from __future__ import annotations

import math

from _benchmark import MicrobenchmarkSuite, timed_loop
from _benchmark.sizing import SizeRequest, clamp, resolve_linear_suite_size


# =============================================================================
# OPTIMIZATION TARGET BENCHMARKS
# =============================================================================

# float64 data + int64/bool index structures across the suite.
_OPT_TARGETS_BYTES_PER_ELEMENT = 17


def _describe_size(size: int) -> list[str]:
    n = math.isqrt(size)
    n_5d = max(2, int(size ** (1 / 5)))
    return [f"resolved_2d_shape: {n} x {n}", f"resolved_5d_shape: {n_5d}^5"]


# ---------------------------------------------------------------------------
# Bool non-scalar SET: a[bool_mask] = array
# ---------------------------------------------------------------------------


def opt1_bool_nonscalar_set(np, size, runs, warmup, *, timer):
    """
    a[bool_mask] = array
    Current path: ADVANCED_INDEXING task → Point<N> index array → issue_scatter
    Could potentially be replaced with: single-pass fused scatter task (single-GPU)
    """
    a = np.random.random(size)
    mask = a > 0.5
    num_selected = int(mask.sum())
    values = np.random.random(num_selected)

    def operation():
        a[mask] = values

    return timed_loop(operation, timer, runs, warmup) / runs


# ---------------------------------------------------------------------------
# Integer GET on non-leading axis / high-dimensional arrays
# ---------------------------------------------------------------------------


def opt2_int_col_get_2d(np, n, num_indices, runs, warmup, *, timer):
    """
    a[:, indices]  (2D, non-leading axis)
    Current path: ZIP task → issue_gather
    Could potentially be replaced with: einsum (ndim < 5) or TAKE task
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[:, indices]

    return timed_loop(operation, timer, runs, warmup) / runs


def opt2_int_get_5d(np, n_5d, num_indices, runs, warmup, *, timer):
    """
    a[indices, :, :, :, :]  (5D — einsum skipped at ndim >= 5)
    Current path: ZIP task → issue_gather
    Could potentially be replaced with: TAKE task (_take_decide_algorithm)
    """
    a = np.random.random((n_5d,) * 5)
    indices = np.random.randint(0, n_5d, num_indices)

    def operation():
        return a[indices, :, :, :, :]

    return timed_loop(operation, timer, runs, warmup) / runs


# ---------------------------------------------------------------------------
# Integer SET: a[indices] = v
# ---------------------------------------------------------------------------


def opt3_int_set_1d(np, size, num_indices, runs, warmup, *, timer):
    """
    a[indices] = v  (1D)
    Current path: ZIP task → issue_scatter
    Could potentially be replaced with: PUT task (symmetric to TAKE,
    partitions along k)
    """
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)
    values = np.random.random(num_indices)

    def operation():
        a[indices] = values

    return timed_loop(operation, timer, runs, warmup) / runs


def opt3_int_col_set_2d(np, n, num_indices, runs, warmup, *, timer):
    """
    a[:, indices] = v  (2D column)
    Current path: ZIP task → issue_scatter
    Could potentially be replaced with: PUT task (broadcasts values + indices,
    partitions along k)
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)
    values = np.random.random((n, num_indices))

    def operation():
        a[:, indices] = values

    return timed_loop(operation, timer, runs, warmup) / runs


# ---------------------------------------------------------------------------
# Flat indexing GET (SET via a.flat[array]=v is not supported in cupynumeric)
# ---------------------------------------------------------------------------


def opt6_flat_get_contiguous(np, n, num_indices, runs, warmup, *, timer):
    """
    a.flat[indices]  (C-contiguous array, GET)
    Current path: WRAP task → issue_gather
    Could potentially be replaced with: direct copy / reshape (no gather needed)
    """
    a = np.random.random((n, n))  # C-contiguous
    flat_size = n * n
    indices = np.random.randint(0, flat_size, num_indices)

    def operation():
        return a.flat[indices]

    return timed_loop(operation, timer, runs, warmup) / runs


# ---------------------------------------------------------------------------
# Boolean in mixed-key position
# ---------------------------------------------------------------------------


def opt8_bool_row_get(np, n, runs, warmup, *, timer):
    """
    a[bool_mask, :]  (row GET)
    Current path: nonzero → multiple int64 arrays → ZIP + gather
    Could potentially be replaced with: relax _prepare_boolean_array_indexing
    to allow slice(None) co-keys → ADVANCED_INDEXING task + transpose back,
    no ZIP/gather
    """
    a = np.random.random((n, n))
    mask = a[:, 0] > 0.5

    def operation():
        return a[mask, :]

    return timed_loop(operation, timer, runs, warmup) / runs


def opt8_bool_col_get(np, n, runs, warmup, *, timer):
    """
    a[:, bool_mask]  (column GET)
    Current path: nonzero → multiple int64 arrays → ZIP + gather
    Could potentially be replaced with: transpose bool axis to front →
    ADVANCED_INDEXING task → transpose back
    """
    a = np.random.random((n, n))
    mask = a[0, :] > 0.5

    def operation():
        return a[:, mask]

    return timed_loop(operation, timer, runs, warmup) / runs


def opt8_bool_col_set_scalar(np, n, runs, warmup, *, timer):
    """
    a[:, bool_mask] = scalar  (column SET, scalar RHS)
    Current path: nonzero → multiple int64 arrays → ZIP + scatter
    Could potentially be replaced with: transpose → PUTMASK task → transpose
    back, no scatter
    """
    a = np.random.random((n, n))
    mask = a[0, :] > 0.5

    def operation():
        a[:, mask] = 0.0

    return timed_loop(operation, timer, runs, warmup) / runs


def opt8_bool_col_set_array(np, n, runs, warmup, *, timer):
    """
    a[:, bool_mask] = array  (column SET, array RHS)
    Current path: nonzero → multiple int64 arrays → ZIP + scatter
    Could potentially be replaced with: transpose → ADVANCED_INDEXING task +
    scatter → transpose back (ZIP eliminated; scatter still needed for
    non-scalar multi-GPU case)
    """
    a = np.random.random((n, n))
    mask = a[0, :] > 0.5
    num_selected = int(mask.sum())
    values = np.random.random((n, num_selected))

    def operation():
        a[:, mask] = values

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run optimization-target benchmarks (all currently use gather/scatter)."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_OPT_TARGETS_BYTES_PER_ELEMENT,
        describe_size=_describe_size,
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    ns = [math.isqrt(size) for size in sizes]

    def arg_gen_2d():
        for size in sizes:
            n = math.isqrt(size)
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, n, num_indices, runs, warmup)

    def arg_gen_5d():
        for size in sizes:
            n_5d = max(2, int(size ** (1 / 5)))
            num_indices = clamp(n_5d // 2, 1, 100)
            yield (np, n_5d, num_indices, runs, warmup)

    def arg_gen_1d():
        for size in sizes:
            n = math.isqrt(size)
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, size, num_indices, runs, warmup)

    # Bool non-scalar SET: a[bool_mask] = array
    suite.run_timed(
        opt1_bool_nonscalar_set, np, sizes, runs, warmup, timer=timer
    )

    # Integer GET on non-leading axis / high-dimensional arrays
    suite.run_timed_with_generator(
        None, opt2_int_col_get_2d, arg_gen_2d(), timer=timer
    )
    suite.run_timed_with_generator(
        None, opt2_int_get_5d, arg_gen_5d(), timer=timer
    )

    # Integer SET: a[indices] = v
    suite.run_timed_with_generator(
        None, opt3_int_set_1d, arg_gen_1d(), timer=timer
    )
    suite.run_timed_with_generator(
        None, opt3_int_col_set_2d, arg_gen_2d(), timer=timer
    )

    # Flat indexing GET (SET via a.flat[array] = v is not supported in cupynumeric)
    suite.run_timed_with_generator(
        None, opt6_flat_get_contiguous, arg_gen_2d(), timer=timer
    )

    # Boolean in mixed-key position
    suite.run_timed(opt8_bool_row_get, np, ns, runs, warmup, timer=timer)
    suite.run_timed(opt8_bool_col_get, np, ns, runs, warmup, timer=timer)
    suite.run_timed(
        opt8_bool_col_set_scalar, np, ns, runs, warmup, timer=timer
    )
    suite.run_timed(opt8_bool_col_set_array, np, ns, runs, warmup, timer=timer)


class IndexingOptTargetsSuite(MicrobenchmarkSuite):
    name = "indexing_opt_targets"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
