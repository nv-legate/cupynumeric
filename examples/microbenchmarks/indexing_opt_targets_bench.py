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
    a[indices, ...]  5D   →  today: ZIP + gather (TAKE task not available at ndim >= 5)
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
from _benchmark.sizing import SizeRequest, resolve_linear_suite_size


# =============================================================================
# OPTIMIZATION TARGET BENCHMARKS
# =============================================================================

# Per-test peak bytes per resolved element (float64 = 8 bytes):
#
# Fractional BPE from ZIP Point<N> index arrays:
#   Legion's ZIP gather/scatter operates on arrays of Point<N> structs, where
#   each Point<N> stores N int64 coordinates — one per array dimension.
#   One Point<N> therefore costs N×8 bytes.  When the ZIP index array holds
#   p points over an n²-element problem, its contribution to BPE is:
#
#       p × (N × 8) / n²
#
#   Example — _BPE_11 uses p = n × (n // 10) points with N = 2:
#       n × (n/10) × 16 / n²  =  16/10  =  1.6 B/el
#
#   _BPE_8  — float64 array only; intermediate index arrays scale as sqrt(size)
#             or smaller and are negligible for large n:
#             opt3_int_set_1d (indices ∝ sqrt(size)),
#             opt6_flat_get_contiguous (indices ∝ n, output ∝ n vs array ∝ n²)
#
#   _BPE_11 — float64 array (8) + ZIP Point<2> index array (1.6) + output or
#             values (0.8), all ∝ n × num_indices with num_indices = n // 10:
#             opt2_int_col_get_2d, opt3_int_col_set_2d
#
#   _BPE_14 — float64 array (8) + ZIP Point<2> index array (5.3):
#             p = n × (n // 3)  →  n×(n/3) × 16 / n²  =  16/3 ≈ 5.3 B/el
#             opt8_bool_col_set_scalar (a + ZIP indices; scalar RHS adds nothing)
#
#   _BPE_17 — float64 array (8) + bool mask (1) + values ~size/2 (4) +
#             ADVANCED_INDEXING Point<1> index array ~size/2 (4):
#             opt1_bool_nonscalar_set
#
#   _BPE_20 — float64 array (8) + ZIP Point<2> index array (8) + output or
#             values (4), all ∝ n × (n // 2):
#             p = n × (n/2)  →  n×(n/2) × 16 / n²  =  8 B/el
#             opt8_bool_row_get / opt8_bool_col_get (a + ZIP indices + output),
#             opt8_bool_col_set_array (a + ZIP indices + values of shape n×(n/2))
#
#   _BPE_32 — float64 array (8) + output n_5d^5/2 × 8 (4) + ZIP Point<5> array
#             (20), with p = n_5d^5 / 2 and N = 5:
#             n_5d^5/2 × 40 / n_5d^5  =  40/2  =  20 B/el
#             opt2_int_get_5d
_BPE_8 = 8
_BPE_11 = 11
_BPE_14 = 14
_BPE_17 = 17
_BPE_20 = 20
_BPE_32 = 32


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
    Could potentially be replaced with: TAKE task
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[:, indices]

    return timed_loop(operation, timer, runs, warmup) / runs


def opt2_int_get_5d(np, n_5d, num_indices, runs, warmup, *, timer):
    """
    a[indices, :, :, :, :]  (5D — TAKE task not available at ndim >= 5)
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
    # Select every 3rd column (~33% density) so the ZIP Point<2> array always
    # has n × (n//3) < n²/2 elements, staying below the int32 scatter boundary.
    # Avoids np.random.choice which is not implemented in cupynumeric.
    mask = np.arange(n) % 3 == 0

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
    sizes_11, resolutions_11 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_11,
        describe_size=lambda s: [
            f"resolved_2d_shape (11 B/el): {math.isqrt(s)} x {math.isqrt(s)}"
        ],
    )
    sizes_16, resolutions_16 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_14,
        describe_size=lambda s: [
            f"resolved_2d_shape (16 B/el): {math.isqrt(s)} x {math.isqrt(s)}"
        ],
    )
    sizes_17, resolutions_17 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_17,
        describe_size=lambda s: [f"resolved_1d_size (17 B/el): {s}"],
    )
    sizes_20, resolutions_20 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_20,
        describe_size=lambda s: [
            f"resolved_2d_shape (20 B/el): {math.isqrt(s)} x {math.isqrt(s)}"
        ],
    )
    sizes_32, resolutions_32 = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_BPE_32,
        describe_size=lambda s: [
            f"resolved_5d_shape (32 B/el): {max(2, int(s ** (1 / 5)))}^5"
        ],
    )
    if resolutions_11 is not None:
        suite.print_size_resolution(resolutions_11)
    if resolutions_16 is not None:
        suite.print_size_resolution(resolutions_16)
    if resolutions_17 is not None:
        suite.print_size_resolution(resolutions_17)
    if resolutions_20 is not None:
        suite.print_size_resolution(resolutions_20)
    if resolutions_32 is not None:
        suite.print_size_resolution(resolutions_32)

    # n values for the bool-in-mixed-key tests (passed directly to run_timed).
    ns_16 = [math.isqrt(size) for size in sizes_16]
    ns_20 = [math.isqrt(size) for size in sizes_20]

    def arg_gen_2d_11():
        # opt2_int_col_get_2d, opt3_int_col_set_2d: num_indices = n // 10
        # scales with n so ZIP Point<2> output is ~20% of `a`.
        for size in sizes_11:
            n = math.isqrt(size)
            num_indices = max(1, n // 10)
            yield (np, n, num_indices, runs, warmup)

    def arg_gen_5d_32():
        # opt2_int_get_5d: num_indices = n_5d // 2; ZIP Point<5> output is
        # ~2.5× the size of `a` (Point<5> = 40 bytes vs float64 = 8 bytes).
        for size in sizes_32:
            n_5d = max(2, int(size ** (1 / 5)))
            num_indices = max(1, n_5d // 2)
            yield (np, n_5d, num_indices, runs, warmup)

    def arg_gen_2d_8():
        # opt6_flat_get_contiguous: num_indices = n // 10, but output and WRAP
        # indices are O(n) vs O(n²) for `a`, so they stay negligible.
        for size in sizes_8:
            n = math.isqrt(size)
            num_indices = max(1, n // 10)
            yield (np, n, num_indices, runs, warmup)

    def arg_gen_1d_8():
        # opt3_int_set_1d: num_indices = sqrt(size) // 10; intermediates scale
        # as sqrt(size) and are negligible relative to the size-element array.
        for size in sizes_8:
            n = math.isqrt(size)
            num_indices = max(1, n // 10)
            yield (np, size, num_indices, runs, warmup)

    # Bool non-scalar SET: a[bool_mask] = array  — 17 B/el
    suite.run_timed(
        opt1_bool_nonscalar_set, np, sizes_17, runs, warmup, timer=timer
    )

    # Integer GET on non-leading axis / high-dimensional arrays
    suite.run_timed_with_generator(
        None,
        opt2_int_col_get_2d,
        arg_gen_2d_11(),
        timer=timer,  # 11 B/el: a + ZIP Point<2> indices + output
    )
    suite.run_timed_with_generator(
        None,
        opt2_int_get_5d,
        arg_gen_5d_32(),
        timer=timer,  # 32 B/el: a + ZIP Point<5> indices + output
    )

    # Integer SET: a[indices] = v
    suite.run_timed_with_generator(
        None,
        opt3_int_set_1d,
        arg_gen_1d_8(),
        timer=timer,  # 8 B/el: a only (intermediates ∝ sqrt(size), negligible)
    )
    suite.run_timed_with_generator(
        None,
        opt3_int_col_set_2d,
        arg_gen_2d_11(),
        timer=timer,  # 11 B/el: a + ZIP Point<2> indices + values
    )

    # Flat indexing GET (SET via a.flat[array] = v is not supported in cupynumeric)
    suite.run_timed_with_generator(
        None,
        opt6_flat_get_contiguous,
        arg_gen_2d_8(),
        timer=timer,  # 8 B/el: a only (output ∝ n, negligible vs n²)
    )

    # Boolean in mixed-key position
    suite.run_timed(
        opt8_bool_row_get,
        np,
        ns_20,
        runs,
        warmup,
        timer=timer,  # 20 B/el: a + ZIP indices + output
    )
    suite.run_timed(
        opt8_bool_col_get,
        np,
        ns_20,
        runs,
        warmup,
        timer=timer,  # 20 B/el: a + ZIP indices + output
    )
    suite.run_timed(
        opt8_bool_col_set_scalar,
        np,
        ns_16,
        runs,
        warmup,
        timer=timer,  # 16 B/el: a + ZIP indices (scalar RHS adds nothing)
    )
    suite.run_timed(
        opt8_bool_col_set_array,
        np,
        ns_20,
        runs,
        warmup,
        timer=timer,  # 20 B/el: a + values + ZIP indices
    )


class IndexingOptTargetsSuite(MicrobenchmarkSuite):
    name = "indexing_opt_targets"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
