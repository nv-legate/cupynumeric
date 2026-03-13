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
- These operations use the ADVANCED_INDEXING task
- They trigger indirect copy operations via Legion runtime
- Performance depends on Copy operation efficiency and task launch overhead

Operations tested (all use ADVANCED_INDEXING task):
1. Boolean mask GET (1D): result = a[bool_mask]
2. Boolean mask SET (1D): a[bool_mask] = array_values
3. 2D Boolean indexing: a[mask_2d]
4. Mixed indexing (3D): a[indices, :, slice] - integer array + slices
5. Non-contiguous (3D): a[idx_row, :, idx_col] - indices on non-adjacent dims
6. Boolean with slice (2D): a[mask, :] - boolean on first dimension
7. Take one per row: a[arange(M), index] - Legate-Boost pattern (fancy 2D indexing)
8. 1D integer array GET: a[indices] - Hamiltonian pattern
9. 1D integer array SET: a[indices] = values - Hamiltonian pattern
10. 2D row selection: a[row_indices] - Hamiltonian pattern
11. 2D scalar + list SET: a[idx, list(cols)] = value - Hamiltonian pattern

For optimized paths that AVOID Copy operations (putmask, einsum, take_task),
see fast_advanced_indexing_bench.py

Usage:
    Run via main.py:
    python main.py --suite general_indexing [--size SIZE] [--runs RUNS]

    # Compare with numpy backend:
    python main.py --suite general_indexing --package numpy
"""

import random

from _benchmark import MicrobenchmarkSuite, timed_loop


# =============================================================================
# GENERAL ADVANCED INDEXING BENCHMARKS (Uses Copy Operations)
# =============================================================================


def boolean_get(np, size, runs, warmup, *, timer):
    """Boolean mask get operation."""
    a = np.random.random(size)
    mask = a > 0.5

    def operation():
        return a[mask]

    return timed_loop(operation, timer, runs, warmup)


def boolean_set_array(np, size, runs, warmup, *, timer):
    """Array assignment to boolean mask."""
    a = np.random.random(size)
    mask = a > 0.5
    num_selected = int(mask.sum())
    values = np.random.random(num_selected)

    def operation():
        a[mask] = values

    return timed_loop(operation, timer, runs, warmup)


def boolean_2d(np, n, runs, warmup, *, timer):
    """Multi-dimensional boolean indexing."""
    a = np.random.random((n, n))
    mask = a > 0.5

    def operation():
        return a[mask]

    return timed_loop(operation, timer, runs, warmup)


def mixed_indexing(np, n, num_indices, runs, warmup, *, timer):
    """Mixed indexing: integer array + slices."""
    a = np.random.random((n, n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return a[indices, :, : n // 2]

    return timed_loop(operation, timer, runs, warmup)


def non_contiguous_indexing(np, n, num_indices, runs, warmup, *, timer):
    """Non-contiguous indexing: indices on multiple non-adjacent dimensions."""
    a = np.random.random((n, n, n))
    idx_row = np.random.randint(0, n, num_indices)
    idx_col = np.random.randint(0, n, num_indices)

    def operation():
        return a[idx_row, :, idx_col]

    return timed_loop(operation, timer, runs, warmup)


def boolean_with_slice(np, n, runs, warmup, *, timer):
    """Boolean mask on first dimension with slice on remaining dimensions."""
    a = np.random.random((n, n))
    mask = a[:, 0] > 0.5

    def operation():
        return a[mask, :]

    return timed_loop(operation, timer, runs, warmup)


def array_get_1d(np, size, num_indices, runs, warmup, *, timer):
    """1D array GET with integer array (like config_ints[safe_indices])."""
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)

    def operation():
        return a[indices]

    return timed_loop(operation, timer, runs, warmup)


def array_set_1d(np, size, num_indices, runs, warmup, *, timer):
    """1D array SET with integer array (like Hv[batch_indices] = values)."""
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)
    values = np.random.random(num_indices)

    def operation():
        a[indices] = values

    return timed_loop(operation, timer, runs, warmup)


def row_select_2d(np, n, num_rows, runs, warmup, *, timer):
    """2D row selection with integer array (like alph_configs[alph_idx])."""
    a = np.random.random((n, n))
    row_indices = np.random.randint(0, n, num_rows)

    def operation():
        return a[row_indices]

    return timed_loop(operation, timer, runs, warmup)


def scalar_list_set_2d(np, n, num_cols, runs, warmup, *, timer):
    """2D assignment with scalar row + list of columns (like result[idx, list(positions)] = True)."""
    a = np.random.random((n, n))
    idx = n // 2
    positions = random.sample(range(n), num_cols)

    def operation():
        a[idx, positions] = 999.0

    return timed_loop(operation, timer, runs, warmup)


def take_one_per_row(np, m, n, runs, warmup, *, timer):
    """
    Take one element from each row (Legate-Boost pattern).
    Pattern: A[np.arange(M), index] where index[i] selects column for row i
    Equivalent to: np.stack([A[row, index[row]] for row in range(M)])
    """
    a = np.random.random((m, n))
    # Each row gets a different column index
    col_indices = np.random.randint(0, n, m)
    row_indices = np.arange(m)

    def operation():
        return a[row_indices, col_indices]

    return timed_loop(operation, timer, runs, warmup)


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size):
    """Run general advanced indexing benchmarks (uses Copy operations)."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    # Derived sizes
    n = int(size**0.5)  # For 2D arrays
    n_3d = int(size ** (1 / 3))  # For 3D arrays
    num_row_idx = min(1000, n // 10)
    num_col_idx = min(1000, n // 10)
    num_indices = min(1000, n_3d // 5)

    # 1. Boolean mask GET
    suite.run_timed(boolean_get, np, size, runs, warmup, timer=timer)

    # 2. Boolean mask SET (array)
    suite.run_timed(boolean_set_array, np, size, runs, warmup, timer=timer)

    # 3. 2D Boolean indexing
    suite.run_timed(boolean_2d, np, n, runs, warmup, timer=timer)

    # 4. Mixed indexing
    suite.run_timed(
        mixed_indexing, np, n_3d, num_indices, runs, warmup, timer=timer
    )

    # 5. Non-contiguous indexing (indices on non-adjacent dimensions)
    suite.run_timed(
        non_contiguous_indexing,
        np,
        n_3d,
        num_indices,
        runs,
        warmup,
        timer=timer,
    )

    # 6. Boolean mask with slice
    suite.run_timed(boolean_with_slice, np, n, runs, warmup, timer=timer)

    # 7. Take one from each row (Legate-Boost: A[arange(M), index])
    # Note: This also covers fancy_2d and take_pairs patterns (same code path)
    suite.run_timed(take_one_per_row, np, n, n, runs, warmup, timer=timer)

    # 8. 1D integer array GET (Hamiltonian: config_ints[safe_indices])
    suite.run_timed(
        array_get_1d, np, size, num_row_idx, runs, warmup, timer=timer
    )

    # 9. 1D integer array SET (Hamiltonian: Hv[batch_indices] = values)
    suite.run_timed(
        array_set_1d, np, size, num_row_idx, runs, warmup, timer=timer
    )

    # 10. 2D row selection (Hamiltonian: alph_configs[alph_idx])
    suite.run_timed(
        row_select_2d, np, n, num_row_idx, runs, warmup, timer=timer
    )

    # 11. 2D scalar + list assignment (Hamiltonian: result[idx, list(positions)])
    suite.run_timed(
        scalar_list_set_2d, np, n, num_col_idx, runs, warmup, timer=timer
    )


class GeneralIndexingSuite(MicrobenchmarkSuite):
    name = "general_indexing"

    def run_suite(self, size):
        run_benchmarks(self, size)
