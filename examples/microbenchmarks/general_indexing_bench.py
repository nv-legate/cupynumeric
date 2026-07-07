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

Operations tested:
1.  Boolean mask SET (1D): a[bool_mask] = array_values
2.  Non-contiguous (3D): a[idx_row, :, idx_col] - indices on non-adjacent dims → ZIP + gather
3.  Boolean with slice row (2D): a[mask, :] - boolean on first dim (nonzero + ZIP + gather)
4.  Take one per row: a[arange(M), index] - Legate-Boost pattern (fancy 2D indexing)
5.  1D integer array GET: a[indices] - Hamiltonian pattern
6.  1D integer array SET: a[indices] = values - Hamiltonian pattern
7.  2D scalar + list SET: a[idx, list(cols)] = value - Hamiltonian pattern
8.  Column-wise integer SET (2D): a[:, indices] = v - ZIP + scatter
9.  Boolean mask on non-first dim: a[:, bool_mask] - nonzero + ZIP + gather
10. Scalar RHS + integer array SET: a[indices] = scalar - scatter path, no putmask

For np.take, np.take_along_axis, np.put and np.put_along_axis, see take_put_bench.py.

For boolean GET without gather/scatter (ADVANCED_INDEXING task only), and
other optimized paths (putmask, TAKE task), see fast_advanced_indexing_bench.py

Usage:
    Run via main.py:
    python main.py --suite general_indexing [--size SIZE | --memory-size 64MiB]

    # Compare with numpy backend:
    python main.py --suite general_indexing --package numpy
"""

from __future__ import annotations

import random

from _benchmark import (
    SIZE,
    MicrobenchmarkSuite,
    microbenchmark,
    nthroot,
    timed_loop,
)


class GeneralIndexingSuite(MicrobenchmarkSuite):
    name = "general_indexing"

    @microbenchmark(
        args_to_arrays=lambda size: [
            ("a", size),
            ("mask", size, "bool"),
            ("indices", size // 2, "int"),
            ("values", size // 2),
        ],
        args_to_work=lambda size: size,
    )
    def boolean_set_array(np, size, runs, warmup, *, timer):
        """Array assignment to boolean mask."""
        a = np.random.random(size)
        mask = a > 0.5
        num_selected = int(mask.sum())
        values = np.random.random(num_selected)

        def operation():
            a[mask] = values

        return timed_loop(operation, timer, runs, warmup) / runs

    def _skip_non_contiguous_index(self):
        if self.package == "legate":
            from cupynumeric.runtime import runtime

            if runtime.num_gpus > 4:
                self.info(
                    f"Skipping general_indexing::non_contiguous_indexing on "
                    f"{runtime.num_gpus} GPUs: NCCL all2all crash above 4 GPUs"
                )
                return True
        return False

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 3),
            "num_indices": nthroot(size, 3) ** 2,
        },
        args_to_arrays=lambda n, num_indices: [
            ("a", (n, n, n)),
            ("idx_row", num_indices, "int"),
            ("idx_col", num_indices, "int"),
            ("values", (num_indices, n)),
        ],
        args_to_work=lambda n, num_indices: num_indices * n,
        skip=lambda suite: suite._skip_non_contiguous_index(),
    )
    def non_contiguous_indexing(np, n, num_indices, runs, warmup, *, timer):
        """Non-contiguous indexing: indices on multiple non-adjacent dimensions."""
        a = np.random.random((n, n, n))
        idx_row = np.random.randint(0, n, num_indices)
        idx_col = np.random.randint(0, n, num_indices)

        def operation():
            return a[idx_row, :, idx_col]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {"n": nthroot(size, 2)},
        args_to_arrays=lambda n: [
            ("a", (n, n)),
            ("mask", n, "bool"),
            ("indices", n // 2, "int"),
            ("values", (n // 2, n)),
        ],
        args_to_work=lambda n: n * n,
    )
    def boolean_with_slice(np, n, runs, warmup, *, timer):
        """Boolean mask on first dim with slice: a[mask, :] — nonzero + ZIP + gather."""
        a = np.random.random((n, n))
        mask = a[:, 0] > 0.5

        def operation():
            return a[mask, :]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        args_to_arrays=lambda size, num_indices: [
            ("a", size),
            ("indices", num_indices, "int"),
            ("staging buffers", 2 * num_indices),
            ("indices send/recv buffers", 2 * num_indices, "int"),
            ("values", num_indices),
        ],
        args_to_work=lambda num_indices: num_indices,
        plan={"num_indices": SIZE},
    )
    def array_get_1d(np, size, num_indices, runs, warmup, *, timer):
        """1D array GET with integer array (like config_ints[safe_indices])."""
        a = np.random.random(size)
        indices = np.random.randint(0, size, num_indices)

        def operation():
            return a[indices]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        args_to_arrays=lambda size, num_indices: [
            ("a", size),
            ("indices", num_indices, "int"),
            ("values", num_indices),
        ],
        args_to_work=lambda num_indices: num_indices,
        plan={"num_indices": SIZE},
    )
    def array_set_1d(np, size, num_indices, runs, warmup, *, timer):
        """1D array SET with integer array (like Hv[batch_indices] = values)."""
        a = np.random.random(size)
        indices = np.random.randint(0, size, num_indices)
        values = np.random.random(num_indices)

        def operation():
            a[indices] = values

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {
            "n": nthroot(size, 2),
            "num_cols": nthroot(size, 2),
        },
        args_to_arrays=lambda n, num_cols: [
            ("a", (n, n)),
            ("positions", num_cols, "int"),
        ],
        args_to_work=lambda num_cols: num_cols,
        # WARNING: work is sublinear in memory size, this should not be
        # included in --rescale-by-work sequences
        skip=True,
    )
    def scalar_list_set_2d(np, n, num_cols, runs, warmup, *, timer):
        """2D assignment with scalar row + list of columns (like result[idx, list(positions)] = True)."""
        a = np.random.random((n, n))
        idx = n // 2
        positions = random.sample(range(n), num_cols)

        def operation():
            a[idx, positions] = 999.0

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
    def array_set_col_2d(np, n, num_indices, runs, warmup, *, timer):
        """Column-wise integer array SET: a[:, indices] = v — ZIP + scatter."""
        a = np.random.random((n, n))
        indices = np.random.randint(0, n, num_indices)
        values = np.random.random((n, num_indices))

        def operation():
            a[:, indices] = values

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        size_to_args=lambda size: {"n": nthroot(size, 2)},
        args_to_arrays=lambda n: [
            ("a", (n, n)),
            ("mask", n, "bool"),
            ("indices", n // 2, "int"),
            ("values", (n, n // 2)),
        ],
        args_to_work=lambda n: n * (n // 2),
    )
    def boolean_col_with_slice(np, n, runs, warmup, *, timer):
        """Boolean mask on non-first dim: a[:, bool_mask] — nonzero + ZIP + gather."""
        a = np.random.random((n, n))
        mask = a[0, :] > 0.5

        def operation():
            return a[:, mask]

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        args_to_arrays=lambda size, num_indices: [
            ("a", size),
            ("indices", num_indices, "int"),
        ],
        args_to_work=lambda num_indices: num_indices,
        plan={"num_indices": SIZE},
    )
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

    @microbenchmark(
        size_to_args=lambda size: {"m": size, "n": 1000},
        args_to_arrays=lambda m, n: [
            ("a", (m, n)),
            ("col_indices", m, "int"),
            ("row_indices", m, "int"),
            ("values", m),
        ],
        args_to_work=lambda m: m,
    )
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
