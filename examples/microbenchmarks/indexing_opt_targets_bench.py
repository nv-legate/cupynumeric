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

    Each test group uses its own bytes-per-element value (8, 12, or 13) so that
    --memory-size accurately bounds the peak allocation for every test.

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

# Per-test peak bytes per resolved element (float64 = 8 bytes):
#
#   _BPE_8  — one float64 array only; index/values arrays are ≤1000 elements
#             and contribute negligible memory relative to n² or n^5:
#             opt2_int_col_get_2d, opt3_int_set_1d, opt3_int_col_set_2d,
#             opt6_flat_get_contiguous, opt8_bool_col_set_scalar
#
#   _BPE_12 — one float64 array + ~50% proportional float64 output or values:
#             opt2_int_get_5d (a + indexed output),
#             opt8_bool_row_get / opt8_bool_col_get (a + ~n/2 rows/cols),
#             opt8_bool_col_set_array (a + values of shape n×(n/2))
#
#   _BPE_13 — float64 array + bool mask (×1 byte) + ~50% float64 values:
#             opt1_bool_nonscalar_set (a + mask + ~size/2 values)
_BPE_8 = 8
_BPE_12 = 12
_BPE_13 = 13


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
        return a.ravel()[indices]

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

    # Resolve problem sizes independently per test group so that --memory-size
    # accurately bounds peak allocation for every test.  Each group uses its
    # own bytes-per-element constant that matches the actual peak footprint.
    sizes_8, _ = resolve_linear_suite_size(
        size_request, bytes_per_element=_BPE_8
    )
    sizes_12, resolutions_12 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_12,
        describe_size=lambda s: [
            f"resolved_2d_shape (12 B/el): {math.isqrt(s)} x {math.isqrt(s)}"
        ],
    )
    sizes_13, resolutions_13 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_13,
        describe_size=lambda s: [f"resolved_1d_size (13 B/el): {s}"],
    )
    if resolutions_12 is not None:
        suite.print_size_resolution(resolutions_12)
    if resolutions_13 is not None:
        suite.print_size_resolution(resolutions_13)

    # n values for 2D tests derived from each category's sizes.
    ns_8 = [math.isqrt(size) for size in sizes_8]
    ns_12 = [math.isqrt(size) for size in sizes_12]

    def arg_gen_2d_8():
        # Tests that hold only `a` (n²×8); indices/values are ≤1000 elements.
        for size in sizes_8:
            n = math.isqrt(size)
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, n, num_indices, runs, warmup)

    def arg_gen_5d_12():
        # opt2_int_get_5d: `a` (n_5d^5×8) + indexed output (~n_5d^5×4).
        for size in sizes_12:
            n_5d = max(2, int(size ** (1 / 5)))
            num_indices = clamp(n_5d // 2, 1, 100)
            yield (np, n_5d, num_indices, runs, warmup)

    def arg_gen_1d_8():
        # opt3_int_set_1d: `a` (size×8); values array is ≤1000 elements.
        for size in sizes_8:
            n = math.isqrt(size)
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, size, num_indices, runs, warmup)

    # Bool non-scalar SET: a[bool_mask] = array  — 13 B/el (a + mask + values)
    suite.run_timed(
        opt1_bool_nonscalar_set, np, sizes_13, runs, warmup, timer=timer
    )

    # Integer GET on non-leading axis / high-dimensional arrays
    suite.run_timed_with_generator(
        None,
        opt2_int_col_get_2d,
        arg_gen_2d_8(),
        timer=timer,  # 8 B/el: a only
    )
    suite.run_timed_with_generator(
        None,
        opt2_int_get_5d,
        arg_gen_5d_12(),
        timer=timer,  # 12 B/el: a + output
    )

    # Integer SET: a[indices] = v
    suite.run_timed_with_generator(
        None,
        opt3_int_set_1d,
        arg_gen_1d_8(),
        timer=timer,  # 8 B/el: a only
    )
    suite.run_timed_with_generator(
        None,
        opt3_int_col_set_2d,
        arg_gen_2d_8(),
        timer=timer,  # 8 B/el: a only
    )

    # Flat indexing GET (SET via a.flat[array] = v is not supported in cupynumeric)
    suite.run_timed_with_generator(
        None,
        opt6_flat_get_contiguous,
        arg_gen_2d_8(),
        timer=timer,  # 8 B/el: a only
    )

    # Boolean in mixed-key position
    suite.run_timed(
        opt8_bool_row_get,
        np,
        ns_12,
        runs,
        warmup,
        timer=timer,  # 12 B/el: a + output
    )
    suite.run_timed(
        opt8_bool_col_get,
        np,
        ns_12,
        runs,
        warmup,
        timer=timer,  # 12 B/el: a + output
    )
    suite.run_timed(
        opt8_bool_col_set_scalar,
        np,
        ns_8,
        runs,
        warmup,
        timer=timer,  # 8 B/el: a only (scalar RHS)
    )
    suite.run_timed(
        opt8_bool_col_set_array,
        np,
        ns_12,
        runs,
        warmup,
        timer=timer,  # 12 B/el: a + values
    )


class IndexingOptTargetsSuite(MicrobenchmarkSuite):
    name = "indexing_opt_targets"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
