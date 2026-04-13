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
Take/Put Benchmark Suite

Covers np.take, np.take_along_axis, np.put, and np.put_along_axis.

Take operations use the TAKE task and do not go through gather/scatter:
1. take_1d:         np.take(a, indices, axis=0)          — 1D, TAKE task
2. take_2d:         np.take(a, indices, axis=0)          — 2D, TAKE task
3. take_along_axis: np.take_along_axis(a, indices, axis) — TAKE task

Put operations use scatter (currently unoptimized):
4. np_put:          np.put(a, flat_indices, v)            — WRAP + scatter
5. put_along_axis:  np.put_along_axis(a, idx, v, axis)   — ZIP + scatter

Potential improvements for put operations:
  np.put:
    np.put(a, indices, v)  →  today: WRAP task + issue_scatter always
                               could replace with: _issue_scatter_task for
                               num_procs == 1 (avoids cross-GPU overhead)

  np.put_along_axis:
    np.put_along_axis(a, idx, v, axis)
                           →  today: _fill_fancy_index_for_along_axis_routines
                                     + N arange arrays + ZIP + issue_scatter
                               could replace with: dedicated PUT task with
                               along_axis=True (eliminates N arange temporaries)

Usage:
    python main.py --suite take_put [--size SIZE | --memory-size 64MiB]

    # Compare with numpy backend:
    python main.py --suite take_put --package numpy
"""

from __future__ import annotations

import math

from _benchmark import MicrobenchmarkSuite, timed_loop
from _benchmark.sizing import SizeRequest, clamp, resolve_linear_suite_size


# Model float64 data array plus int64 index structures.
_TAKE_PUT_BYTES_PER_ELEMENT = 12


def _describe_size(size: int) -> list[str]:
    n = max(1, math.isqrt(size))
    return [f"resolved_1d_elements: {size:,}", f"resolved_2d_shape: {n} x {n}"]


def take_1d(np, size, num_indices, runs, warmup, *, timer):
    """
    TAKE task (1D): np.take(a, indices, axis=0) → TAKE task, no gather.
    """
    a = np.random.random(size)
    indices = np.random.randint(0, size, num_indices)

    def operation():
        return np.take(a, indices, axis=0, mode="wrap")

    return timed_loop(operation, timer, runs, warmup) / runs


def take_2d(np, n, num_indices, runs, warmup, *, timer):
    """
    TAKE task (2D, row axis): np.take(a, indices, axis=0) → TAKE task, no gather.
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, num_indices)

    def operation():
        return np.take(a, indices, axis=0, mode="wrap")

    return timed_loop(operation, timer, runs, warmup) / runs


def take_along_axis(np, n, num_indices, runs, warmup, *, timer):
    """
    TAKE task (take_along_axis): np.take_along_axis(a, indices, axis=0) → TAKE task, no gather.
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, (num_indices, n))

    def operation():
        return np.take_along_axis(a, indices, axis=0)

    return timed_loop(operation, timer, runs, warmup) / runs


def np_put(np, n, num_indices, runs, warmup, *, timer):
    """
    Flat put: np.put(a, flat_indices, v).
    Current path: WRAP task + issue_scatter unconditionally.
    Could potentially be replaced with _issue_scatter_task for num_procs == 1.
    """
    a = np.random.random((n, n))
    flat_size = n * n
    indices = np.random.randint(0, flat_size, num_indices)
    values = np.random.random(num_indices)

    def operation():
        np.put(a, indices, values)

    return timed_loop(operation, timer, runs, warmup) / runs


def put_along_axis(np, n, num_indices, runs, warmup, *, timer):
    """
    Put along axis: np.put_along_axis(a, idx, v, axis=0).
    Current path: _fill_fancy_index_for_along_axis_routines
    + N arange arrays + ZIP + issue_scatter.
    Could potentially be replaced with a dedicated PUT task with along_axis=True.
    """
    a = np.random.random((n, n))
    indices = np.random.randint(0, n, (num_indices, n))
    values = np.random.random((num_indices, n))

    def operation():
        np.put_along_axis(a, indices, values, axis=0)

    return timed_loop(operation, timer, runs, warmup) / runs


# =============================================================================
# MAIN BENCHMARK SUITE
# =============================================================================


def run_benchmarks(suite, size_request):
    """Run take/put benchmarks (TAKE task for take; scatter path for put)."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_linear_suite_size(
        size_request,
        bytes_per_element=_TAKE_PUT_BYTES_PER_ELEMENT,
        describe_size=_describe_size,
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    def arg_gen_1d():
        for size in sizes:
            n = max(1, math.isqrt(size))
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, size, num_indices, runs, warmup)

    def arg_gen_2d():
        for size in sizes:
            n = max(1, math.isqrt(size))
            num_indices = clamp(n // 10, 1, 1000)
            yield (np, n, num_indices, runs, warmup)

    # TAKE task (no gather)
    suite.run_timed_with_generator(None, take_1d, arg_gen_1d(), timer=timer)
    suite.run_timed_with_generator(None, take_2d, arg_gen_2d(), timer=timer)
    suite.run_timed_with_generator(
        None, take_along_axis, arg_gen_2d(), timer=timer
    )

    # Put operations (scatter path)
    suite.run_timed_with_generator(None, np_put, arg_gen_2d(), timer=timer)
    suite.run_timed_with_generator(
        None, put_along_axis, arg_gen_2d(), timer=timer
    )


class TakePutSuite(MicrobenchmarkSuite):
    name = "take_put"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
