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
1D/2D/3D ``complex64`` transforms plus one 2D ``complex128`` case. Each
case uses a fixed transform shape and resolves only the leading batch
dimension from ``--size`` or ``--memory-size``.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass

from _benchmark import (
    MicrobenchmarkSuite,
    benchmark_info,
    get_benchmark_info,
    timed_loop,
)
from _benchmark.harness import ArrayPackage
from _benchmark.sizing import SizeRequest

RAND_SCALE_FACTOR = 100


@dataclass(frozen=True)
class FFTCase:
    name: str
    dims: int
    dtype_name: str
    extent: int

    @property
    def transform_volume(self) -> int:
        return self.extent**self.dims

    @property
    def itemsize(self) -> int:
        return 16 if self.dtype_name == "complex128" else 8


@dataclass(frozen=True)
class BatchedFFTSizeResolution:
    requested_memory_target_bytes: int
    detail_lines: tuple[str, ...]

    def panel_lines(self) -> list[str]:
        return [
            (
                "requested_memory_target: "
                f"{self.requested_memory_target_bytes:,} bytes"
            ),
            *self.detail_lines,
        ]


_CASES = (
    FFTCase("1d", 1, "complex64", 262_144),
    FFTCase("2d", 2, "complex64", 512),
    FFTCase("3d", 3, "complex64", 64),
    FFTCase("2d_double", 2, "complex128", 512),
)

_SHARED_TRANSFORM_VOLUME = _CASES[0].transform_volume
if any(
    case.transform_volume != _SHARED_TRANSFORM_VOLUME for case in _CASES[1:]
):
    raise RuntimeError("batched FFT cases must use the same transform volume")


def _case_shape(case: FFTCase, batch: int) -> tuple[int, ...]:
    return (batch,) + (case.extent,) * case.dims


def _resolve_batch_from_size(size: int, case: FFTCase) -> int:
    return max(1, size // case.transform_volume)


def _estimate_case_working_set_bytes(case: FFTCase, batch: int) -> int:
    return 2 * batch * case.transform_volume * case.itemsize


def _resolve_batch_from_memory_target(target_bytes: int, case: FFTCase) -> int:
    return max(1, target_bytes // _estimate_case_working_set_bytes(case, 1))


def _describe_case(case: FFTCase, batch: int) -> str:
    shape = " x ".join(str(extent) for extent in _case_shape(case, batch))
    estimated = _estimate_case_working_set_bytes(case, batch)
    return (
        f"{case.name}: batch={batch:,}, shape={shape}, "
        f"dtype={case.dtype_name}, "
        f"estimated_working_set={estimated:,} bytes"
    )


def _resolve_case_batches(
    size_request: SizeRequest,
) -> tuple[dict[FFTCase, list[int]], list[BatchedFFTSizeResolution] | None]:
    if size_request.exact_size is not None:
        return (
            {
                case: [
                    _resolve_batch_from_size(size, case)
                    for size in size_request.exact_size
                ]
                for case in _CASES
            },
            None,
        )

    assert size_request.memory_target_bytes is not None
    batches = {case: [] for case in _CASES}
    resolutions = []
    for target_bytes in size_request.memory_target_bytes:
        detail_lines = []
        oversized = []
        for case in _CASES:
            batch = _resolve_batch_from_memory_target(target_bytes, case)
            batches[case].append(batch)
            detail_lines.append(_describe_case(case, batch))
            estimated = _estimate_case_working_set_bytes(case, batch)
            if estimated > target_bytes:
                oversized.append(f"{case.name}={estimated:,} bytes")
        if oversized:
            warnings.warn(
                "memory target is smaller than estimated working set for "
                "batched FFT case(s): " + ", ".join(oversized),
                RuntimeWarning,
                stacklevel=2,
            )
        resolutions.append(
            BatchedFFTSizeResolution(
                requested_memory_target_bytes=target_bytes,
                detail_lines=tuple(detail_lines),
            )
        )
    return batches, resolutions


def _make_input(array_module, shape: tuple[int, ...], dtype_name: str):
    real_dtype = (
        array_module.float64
        if dtype_name == "complex128"
        else array_module.float32
    )
    float_array = (
        array_module.random.rand(*shape).astype(real_dtype) * RAND_SCALE_FACTOR
    )
    return float_array.astype(getattr(array_module, dtype_name))


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

    return timed_loop(operation, timer, runs, warmup) / runs


def run_benchmarks(suite, size_request: SizeRequest):
    """Run representative batched FFT benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    case_batches, resolutions = _resolve_case_batches(size_request)
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    base_info = get_benchmark_info(batched_fft)

    for case in _CASES:

        def arg_gen(case: FFTCase = case):
            for batch in case_batches[case]:
                yield (
                    np,
                    case.dims,
                    case.dtype_name,
                    batch,
                    case.extent,
                    runs,
                    warmup,
                )

        suite.run_timed_with_generator(
            base_info.replace(name=case.name),
            batched_fft,
            arg_gen(),
            timer=timer,
        )


class BatchedFFTSuite(MicrobenchmarkSuite):
    name = "batched_fft"

    def run_suite(self, size_request: SizeRequest):
        # FFT is only supported on GPU in cuPyNumeric
        if self._config.package == ArrayPackage.LEGATE:
            from cupynumeric.runtime import runtime

            if runtime.num_gpus == 0:
                print("Skipping batched_fft suite: FFT requires GPU")
                return
        run_benchmarks(self, size_request)
