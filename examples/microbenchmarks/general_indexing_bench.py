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
General Advanced Indexing Benchmark - Copy Operation Path

Tests general advanced indexing operations that GO THROUGH Legion/Legate Copy operations:
- They trigger indirect copy operations via Legion runtime
- Performance depends on Copy operation efficiency and task launch overhead

Operations tested (all use gather/scatter):
1.  Boolean mask SET (1D): a[bool_mask] = array_values
2.  Mixed indexing (3D): a[indices, :, slice] - integer array + slices
3.  Non-contiguous (3D): a[idx_row, :, idx_col] - indices on non-adjacent dims
4.  Boolean with slice row (2D): a[mask, :] - boolean on first dim (nonzero + ZIP + gather)
5.  Take one per row: a[arange(M), index] - Legate-Boost pattern (fancy 2D indexing)
6.  1D integer array GET: a[indices] - Hamiltonian pattern
7.  1D integer array SET: a[indices] = values - Hamiltonian pattern
8.  2D scalar + list SET: a[idx, list(cols)] = value - Hamiltonian pattern
9.  Column-wise integer SET (2D): a[:, indices] = v - ZIP + scatter
10. Boolean mask on non-first dim: a[:, bool_mask] - nonzero + ZIP + gather
11. newaxis GET: a[indices, np.newaxis] - newaxis key type forces ZIP path
12. Ellipsis GET (2D): a[..., indices] - Ellipsis key type, tests normalization vs einsum routing
13. Non-contiguous GET: a.T[indices] - F-contiguous source requires copy before gather
14. Scalar RHS + integer array SET: a[indices] = scalar - scatter path, no putmask

For np.take, np.take_along_axis, np.put and np.put_along_axis, see take_put_bench.py.

For boolean GET without gather/scatter (ADVANCED_INDEXING task only), and
other optimized paths (putmask, einsum), see fast_advanced_indexing_bench.py

Usage:
    Run via main.py:
    python main.py --suite general_indexing [--size SIZE | --memory-size 64MiB]

    # Compare with numpy backend:
    python main.py --suite general_indexing --package numpy
"""

from __future__ import annotations

import math
import random

from _benchmark import MicrobenchmarkSuite, timed_loop
from _benchmark.sizing import SizeRequest, clamp, resolve_linear_suite_size


# =============================================================================
# GENERAL ADVANCED INDEXING BENCHMARKS (Uses Copy Operations)
# =============================================================================


# Model float64 data plus int64/boolean index structures across the suite.
_GENERAL_INDEXING_BYTES_PER_ELEMENT = 17


def _describe_size(size: int) -> list[str]:
    n = math.isqrt(size)
    n_3d = int(size ** (1 / 3))
    return [
        f"resolved_2d_shape: {n} x {n}",
        f"resolved_3d_shape: {n_3d} x {n_3d} x {n_3d}",
    ]


def boolean_set_array(np, size, runs, warmup, *, timer):
    """Array assignment to boolean mask."""
    a = np.random.random(size)
    mask = a > 0.5
    num_selected = int(mask.sum())
    values = np.random.random(num_selected)

    def operation():
        a[mask] = values

    return timed_loop(operation, timer, runs, warmup) / runs


def mixed_indexing(np, n, num_indices, runs, warmup, *, timer):
    """Mixed indexing: integer array + slices."""
    a = np.random.random((n, n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[indices, :, : n // 2]

    return timed_loop(operation, timer, runs, warmup) / runs


def non_contiguous_indexing(np, n, num_indices, runs, warmup, *, timer):
    """Non-contiguous indexing: indices on multiple non-adjacent dimensions."""
    a = np.random.random((n, n, n))
    idx_row = np.random.randint(0, n, num_indices)
    idx_col = np.random.randint(0, n, num_indices)

    def operation():
        return a[idx_row, :, idx_col]

    return timed_loop(operation, timer, runs, warmup) / runs


def boolean_with_slice(np, n, runs, warmup, *, timer):
    """Boolean mask on first dim with slice: a[mask, :] — nonzero + ZIP + gather."""
    a = np.random.random((n, n))
    mask = a[:, 0] > 0.5

    def operation():
        return a[mask, :]

    return timed_loop(operation, timer, runs, warmup) / runs


def array_get_1d(np, size, num_indices, runs, warmup, *, timer):
    """1D array GET with integer array (like config_ints[safe_indices])."""
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)

    def operation():
        return a[indices]

    return timed_loop(operation, timer, runs, warmup) / runs


def array_set_1d(np, size, num_indices, runs, warmup, *, timer):
    """1D array SET with integer array (like Hv[batch_indices] = values)."""
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)
    values = np.random.random(num_indices)

    def operation():
        a[indices] = values

    return timed_loop(operation, timer, runs, warmup) / runs


def scalar_list_set_2d(np, n, num_cols, runs, warmup, *, timer):
    """2D assignment with scalar row + list of columns (like result[idx, list(positions)] = True)."""
    a = np.random.random((n, n))
    idx = n // 2
    positions = random.sample(range(n), num_cols)

    def operation():
        a[idx, positions] = 999.0

    return timed_loop(operation, timer, runs, warmup) / runs


def array_set_col_2d(np, n, num_indices, runs, warmup, *, timer):
    """Column-wise integer array SET: a[:, indices] = v — ZIP + scatter."""
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)
    values = np.random.random((n, num_indices))

    def operation():
        a[:, indices] = values

    return timed_loop(operation, timer, runs, warmup) / runs


def boolean_col_with_slice(np, n, runs, warmup, *, timer):
    """Boolean mask on non-first dim: a[:, bool_mask] — nonzero + ZIP + gather."""
    a = np.random.random((n, n))
    mask = a[0, :] > 0.5

    def operation():
        return a[:, mask]

    return timed_loop(operation, timer, runs, warmup) / runs


def newaxis_int_get(np, size, num_indices, runs, warmup, *, timer):
    """
    Integer array GET with trailing newaxis: a[indices, np.newaxis].
    Covers key type: newaxis; key composition: int array + newaxis (mixed).
    Path: newaxis in key prevents einsum routing → ZIP + gather.
    """
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)

    def operation():
        return a[indices, np.newaxis]  # shape: (num_indices, 1)

    return timed_loop(operation, timer, runs, warmup) / runs


def ellipsis_int_get(np, n, num_indices, runs, warmup, *, timer):
    """
    Integer array GET with Ellipsis on leading axes: a[..., indices] (2D).
    Covers key type: Ellipsis; key composition: Ellipsis + int array (mixed).
    Semantically equivalent to a[:, indices] but tests whether Ellipsis
    normalization routes to einsum or falls back to ZIP + gather.
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[..., indices]  # Ellipsis expands to slice(None) for dim 0

    return timed_loop(operation, timer, runs, warmup) / runs


def noncontiguous_get(np, n, num_indices, runs, warmup, *, timer):
    """
    Integer array GET from a non-contiguous (transposed) source array: a.T[indices].
    Covers contiguity: F-contiguous (non-C-contiguous) source requires a copy
    before gather can proceed.
    """
    a = np.random.random((n, n)).T  # F-contiguous (transposed view)
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[indices]  # gather from non-contiguous source

    return timed_loop(operation, timer, runs, warmup) / runs


def array_key_scalar_set(np, size, num_indices, runs, warmup, *, timer):
    """
    Scalar RHS assignment via integer array key: a[indices] = scalar.
    Covers RHS type: scalar with non-boolean key (no putmask path).
    Path: ZIP task → scatter (scalar broadcast, distinct from putmask).
    """
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)

    def operation():
        a[indices] = 0.0

    return timed_loop(operation, timer, runs, warmup) / runs


def take_one_per_row(np, m, n, runs, warmup, *, timer):
    """
    Take one element from each row (Legate-Boost pattern).
    Pattern: A[np.arange(M), index] where index[i] selects column for row i
    Equivalent to: np.stack([A[row, index[row]] for row in range(M)])
    """
    a = np.random.random((m, n))
    col_indices = np.random.randint(0, n, m)
    row_indices = np.arange(m)

    def operation():
        return a[row_indices, col_indices]

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run general advanced indexing benchmarks (uses Copy operations)."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_GENERAL_INDEXING_BYTES_PER_ELEMENT,
        describe_size=_describe_size,
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    ns = [math.isqrt(size) for size in sizes]

    def arg_gen_1d():
        for size in sizes:
            n = math.isqrt(size)
            num_row_idx = clamp(n // 10, 1, 1000)
            yield (np, size, num_row_idx, runs, warmup)

    def arg_gen_2d():
        for size in sizes:
            n = math.isqrt(size)
            num_row_idx = clamp(n // 10, 1, 1000)
            yield (np, n, num_row_idx, runs, warmup)

    def arg_gen_3d():
        for size in sizes:
            n_3d = int(size ** (1 / 3))
            num_indices = clamp(n_3d // 5, 1, 1000)
            yield (np, n_3d, num_indices, runs, warmup)

    def arg_gen_one_per_row():
        for n in ns:
            yield (np, n, n, runs, warmup)

    # 1. Boolean mask SET (array) — ADVANCED_INDEXING task + scatter
    suite.run_timed(boolean_set_array, np, sizes, runs, warmup, timer=timer)

    # 2. Mixed indexing: a[indices, :, slice] — ZIP + gather
    suite.run_timed_with_generator(
        None, mixed_indexing, arg_gen_3d(), timer=timer
    )

    # 3. Non-contiguous indexing (indices on non-adjacent dims — ZIP + gather)
    suite.run_timed_with_generator(
        None, non_contiguous_indexing, arg_gen_3d(), timer=timer
    )

    # 4. Boolean mask with slice: a[mask, :] — nonzero + ZIP + gather
    suite.run_timed(boolean_with_slice, np, ns, runs, warmup, timer=timer)

    # 5. Take one from each row (Legate-Boost: A[arange(M), index] — ZIP + gather)
    suite.run_timed_with_generator(
        None, take_one_per_row, arg_gen_one_per_row(), timer=timer
    )

    # 6-7. 1D integer array GET and SET (Hamiltonian pattern)
    suite.run_timed_with_generator(
        None, array_get_1d, arg_gen_1d(), timer=timer
    )
    suite.run_timed_with_generator(
        None, array_set_1d, arg_gen_1d(), timer=timer
    )

    # 8. 2D scalar+list assignment (Hamiltonian pattern)
    suite.run_timed_with_generator(
        None, scalar_list_set_2d, arg_gen_2d(), timer=timer
    )

    # 9. Column-wise integer SET: a[:, indices] = v — ZIP + scatter
    suite.run_timed_with_generator(
        None, array_set_col_2d, arg_gen_2d(), timer=timer
    )

    # 10. Boolean mask on non-first dim: a[:, bool_mask] — nonzero + ZIP + gather
    suite.run_timed(boolean_col_with_slice, np, ns, runs, warmup, timer=timer)

    # 11. newaxis key type: a[indices, np.newaxis] — ZIP + gather
    suite.run_timed_with_generator(
        None, newaxis_int_get, arg_gen_1d(), timer=timer
    )

    # 12. Ellipsis key type: a[..., indices] — tests normalization vs einsum routing
    suite.run_timed_with_generator(
        None, ellipsis_int_get, arg_gen_2d(), timer=timer
    )

    # 13. Non-contiguous GET: a.T[indices] — copy before gather
    suite.run_timed_with_generator(
        None, noncontiguous_get, arg_gen_2d(), timer=timer
    )

    # 14. Scalar RHS + integer array key: a[indices] = scalar — scatter, no putmask
    suite.run_timed_with_generator(
        None, array_key_scalar_set, arg_gen_1d(), timer=timer
    )


class GeneralIndexingSuite(MicrobenchmarkSuite):
    name = "general_indexing"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
