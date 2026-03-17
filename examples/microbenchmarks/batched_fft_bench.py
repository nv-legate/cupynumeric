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
Batched FFT microbenchmark suite.

The design doc scopes this benchmark to batched 1D, 2D, and 3D
complex-to-complex FFT coverage. The representative cases are batched
1D/2D/3D ``complex64`` transforms plus one 2D ``complex128`` case. The
shared ``--size`` budget is split across a fixed batch size and equal
transform extents.
"""

from __future__ import annotations

import math

from _benchmark import (
    MicrobenchmarkSuite,
    benchmark_info,
    get_benchmark_info,
    timed_loop,
)
from _benchmark.sizing import (
    SizeRequest,
    resolve_size_by_binary_search,
    resolve_suite_size,
)

BATCH_SIZE = 8
_CASES = (
    ("1d", 1, "complex64"),
    ("2d", 2, "complex64"),
    ("3d", 3, "complex64"),
    ("2d_double", 2, "complex128"),
)


def _case_shape(size: int, dims: int) -> tuple[int, ...]:
    work_size = max(1, size // BATCH_SIZE)
    extent = max(2, round(work_size ** (1 / dims)))
    return (BATCH_SIZE,) + (extent,) * dims


def _make_input(array_module, shape: tuple[int, ...], dtype_name: str):
    real_dtype = (
        array_module.float64
        if dtype_name == "complex128"
        else array_module.float32
    )
    return (
        array_module.arange(math.prod(shape), dtype=real_dtype)
        .reshape(shape)
        .astype(getattr(array_module, dtype_name))
    )


def _estimate_case_working_set_bytes(
    dims: int, dtype_name: str, size: int
) -> int:
    shape = _case_shape(size, dims)
    elements = math.prod(shape)
    itemsize = 16 if dtype_name == "complex128" else 8
    return 2 * elements * itemsize


def _estimate_working_set_bytes(size: int) -> int:
    return max(
        _estimate_case_working_set_bytes(dims, dtype_name, size)
        for _, dims, dtype_name in _CASES
    )


def _describe_size(size: int) -> list[str]:
    lines = []
    for case_name, dims, dtype_name in _CASES:
        shape = " x ".join(str(extent) for extent in _case_shape(size, dims))
        lines.append(f"{case_name}: shape={shape}, dtype={dtype_name}")
    return lines


def _resolve_size_from_memory_target(target_bytes: int) -> int:
    return resolve_size_by_binary_search(
        target_bytes,
        estimate_working_set_bytes=_estimate_working_set_bytes,
        initial_guess=max(1, target_bytes // (2 * 16)),
    )


@benchmark_info(
    input_names={"dims": "fft_dims", "dtype_name": "dtype"},
    formats={"dtype": str},
)
def batched_fft(np, dims, dtype_name, batch, extent, runs, warmup, *, timer):
    shape = (batch,) + (extent,) * dims
    src = _make_input(np, shape, dtype_name)
    axes = tuple(range(1, len(shape)))

    def operation():
        np.fft.fftn(src, axes=axes)

    return timed_loop(operation, timer, runs, warmup)


def run_benchmarks(suite, size_request: SizeRequest):
    """Run representative batched FFT benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    size, resolution = resolve_suite_size(
        size_request,
        resolve_from_target=_resolve_size_from_memory_target,
        estimate_working_set_bytes=_estimate_working_set_bytes,
        describe_size=_describe_size,
    )
    if resolution is not None:
        suite.print_size_resolution(resolution)

    base_info = get_benchmark_info(batched_fft)

    for case_name, dims, dtype_name in _CASES:
        shape = _case_shape(size, dims)
        batch, extent = shape[0], shape[1]
        suite.run_timed_with_info(
            base_info.replace(name=case_name),
            batched_fft,
            np,
            dims,
            dtype_name,
            batch,
            extent,
            runs,
            warmup,
            timer=timer,
        )


class BatchedFFTSuite(MicrobenchmarkSuite):
    name = "batched_fft"

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(self, size_request)
