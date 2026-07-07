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

2. TAKE task path: Row-axis integer array GET for 2D+ arrays (a[indices, :])
   - Uses TAKE task (no gather/scatter)

3. boolean GET (1D): a[bool_mask] — ADVANCED_INDEXING task, no gather
   - Single boolean array at position 0; result returned directly

4. boolean GET (2D): a[mask_2d] — ADVANCED_INDEXING task, no gather
   - Full-array boolean mask; same path as 1D case

5. row selection (2D): a[row_indices] — TAKE task path (mask_axis=0, ndim=2)
   - Single integer array with implicit trailing slice(None)

6. column GET (2D): a[:, indices] — TAKE task path (mask_axis=1, ndim=2)
   - Single integer array on non-leading axis

7. mixed indexing (3D): a[indices, :, :n//2] — TAKE task path (partial slice as view transform)
   - Integer array on axis 0; partial slice pre-applied as zero-copy store view

8. newaxis GET (1D): a[indices, np.newaxis] — TAKE task path (newaxis via promote)
   - newaxis stripped before TAKE; re-inserted via promote() after

9. Ellipsis GET (2D): a[..., indices] — TAKE task path (Ellipsis normalizes to slice(None))
   - Semantically equivalent to a[:, indices]; Ellipsis expansion handled before TAKE routing

10. Non-contiguous GET (2D): a.T[indices] — TAKE task path (transposed source)
    - F-contiguous source handled natively by TAKE task without a pre-copy

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

from typing import TYPE_CHECKING

from _benchmark import MicrobenchmarkSuite, microbenchmark, nthroot, timed_loop

if TYPE_CHECKING:
    from _benchmark import ArrayDescription


def _indexing_arrays(a, indices, values) -> list[ArrayDescription]:
    return [("a", a), ("indices", indices, "int"), ("values", values)]


def _boolean_mask_arrays(a, mask, indices, values) -> list[ArrayDescription]:
    # assumes that the implementation may materialize the indices
    # from the mask
    return [
        ("a", a),
        ("mask", mask, "bool"),
        ("indices", indices, "int"),
        ("values", values),
    ]


class FastAdvancedIndexingSuite(MicrobenchmarkSuite):
    name = "advanced_indexing"

    @microbenchmark(
        args_to_arrays=lambda size: _indexing_arrays(
            size, size // 2, size // 2
        )
    )
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

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 2),
            "num_indices": max(1, nthroot(size, 2) // 10),
        },
        args_to_arrays=lambda n, num_indices: (
            _indexing_arrays((n, n), num_indices, (num_indices, num_indices))
        ),
        args_to_work=lambda n, num_indices: n * num_indices,
    )
    def take_2d(np, n, num_indices, runs, warmup, *, timer):
        """
        Test TAKE task optimization for integer array indexing (2D case).
        Path: a[integer_indices, :]
        Optimization: Uses TAKE task (no gather)
        """
        a = np.random.random((n, n))
        indices = np.random.randint(0, n, num_indices)

        def operation():
            return a[indices, :]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        args_to_arrays=lambda size: (
            _boolean_mask_arrays(size, size, size // 2, size // 2)
        )
    )
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

    @microbenchmark(
        size_to_args=lambda size: {"n": nthroot(size, 2, lower_bound=2)},
        args_to_arrays=lambda n: (
            _boolean_mask_arrays((n, n), (n, n), (n * n) // 2, (n * n) // 2)
        ),
    )
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

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 2),
            "num_rows": max(1, nthroot(size, 2) // 10),
        },
        args_to_arrays=lambda n, num_rows: (
            _indexing_arrays((n, n), num_rows, (num_rows, n))
        ),
        args_to_work=lambda n, num_rows: n * num_rows,
    )
    def row_select_2d(np, n, num_rows, runs, warmup, *, timer):
        """
        2D row selection: a[row_indices].
        Path: computed_key=(row_indices,) → TAKE task check passes (1 array, ndim=2<5)
              → TAKE task path, no gather.
        """
        a = np.random.random((n, n))
        row_indices = np.random.randint(0, n, num_rows)

        def operation():
            return a[row_indices]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 2),
            "num_indices": max(1, nthroot(size, 2) // 10),
        },
        args_to_arrays=lambda n, num_indices: (
            _indexing_arrays((n, n), num_indices, (n, num_indices))
        ),
        args_to_work=lambda n, num_indices: n * num_indices,
    )
    def array_get_col_2d(np, n, num_indices, runs, warmup, *, timer):
        """
        Column-wise integer array GET: a[:, indices].
        Path: computed_key=(slice(None), indices) → TAKE task check passes (mask_axis=1, ndim=2<5)
              → TAKE task path, no gather.
        """
        a = np.random.random((n, n))
        indices = np.random.randint(0, n, num_indices)

        def operation():
            return a[:, indices]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 3),
            "num_indices": nthroot(size, 3),
        },
        args_to_arrays=lambda n, num_indices: [
            ("a", (n, n, n)),
            ("indices", num_indices, "int"),
            ("values", (num_indices, n, n // 2)),
        ],
        args_to_work=lambda n, num_indices: num_indices * n * (n // 2),
    )
    def mixed_indexing(np, n, num_indices, runs, warmup, *, timer):
        """Mixed indexing: integer array + partial slice → TAKE task."""
        a = np.random.random((n, n, n))
        indices = np.random.randint(0, n, num_indices)

        def operation():
            return a[indices, :, : n // 2]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {
            "n": size,
            # O(n) work: gather half the array so runtime scales with the
            # memory touched (sqrt(size) left the work sublinear in n).
            "num_indices": max(1, size // 2),
        },
        args_to_arrays=lambda n, num_indices: [
            ("a", n),
            ("indices", num_indices, "int"),
            ("values", (num_indices, 1)),
        ],
        args_to_work=lambda num_indices: num_indices,
    )
    def newaxis_int_get(np, n, num_indices, runs, warmup, *, timer):
        """
        Integer array GET with trailing newaxis: a[indices, np.newaxis].
        Path: newaxis stripped before TAKE; re-inserted via promote() after → TAKE task.
        """
        a = np.random.random(n)
        indices = np.random.randint(0, n, num_indices)

        def operation():
            return a[indices, np.newaxis]  # shape: (num_indices, 1)

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 2),
            "num_indices": nthroot(size, 2),
        },
        args_to_arrays=lambda n, num_indices: [
            ("a", (n, n)),
            ("indices", num_indices, "int"),
            ("values", (n, num_indices)),
        ],
        args_to_work=lambda n, num_indices: n * num_indices,
    )
    def ellipsis_int_get(np, n, num_indices, runs, warmup, *, timer):
        """
        Integer array GET with Ellipsis on leading axes: a[..., indices] (2D).
        Ellipsis normalizes to slice(None) via _unpack_ellipsis, then routes to
        TAKE task — semantically identical to a[:, indices] (mask_axis=1).
        """
        a = np.random.random((n, n))
        indices = np.random.randint(0, n, num_indices)

        def operation():
            return a[..., indices]  # Ellipsis expands to slice(None) for dim 0

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 2),
            "num_indices": nthroot(size, 2),
        },
        args_to_arrays=lambda n, num_indices: [
            ("a", (n, n)),
            ("indices", num_indices, "int"),
            ("values", (num_indices, n)),
        ],
        args_to_work=lambda n, num_indices: n * num_indices,
    )
    def noncontiguous_get(np, n, num_indices, runs, warmup, *, timer):
        """
        Integer array GET from a transposed (F-contiguous) source: a.T[indices].
        The transposed store is handled natively by the TAKE task (mask_axis=0).
        """
        a = np.random.random((n, n)).T  # F-contiguous (transposed view)
        indices = np.random.randint(0, n, num_indices)

        def operation():
            return a[indices]

        return timed_loop(operation, timer, runs, warmup) / runs
