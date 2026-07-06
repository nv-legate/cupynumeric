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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from _benchmark import (
    MicrobenchmarkSuite,
    microbenchmark,
    random_array,
    timed_loop,
)
from _benchmark.harness import ArrayPackage
from _benchmark.sizing import SizeRequest

if TYPE_CHECKING:
    from typing import Callable, Any
    from _benchmark import ArrayDescription


@dataclass(frozen=True)
class FFTCase:
    name: str
    dims: int
    extent: int

    @property
    def transform_volume(self) -> int:
        return self.extent**self.dims


_CASES = (
    FFTCase("1d", 1, 262_144),
    FFTCase("2d", 2, 512),
    FFTCase("3d", 3, 64),
)


RAND_SCALE_FACTOR = 100


def _make_input(array_module, shape: tuple[int, ...], dtype_name: str):
    return random_array(
        array_module, shape, dtype_name, scale=RAND_SCALE_FACTOR
    )


def _batched_fft(np, dims, dtype, batch, extent, runs, warmup, *, timer):
    shape = (batch,) + (extent,) * dims
    src = _make_input(np, shape, dtype)
    axes = tuple(range(1, len(shape)))

    def operation():
        np.fft.fftn(src, axes=axes)

    return timed_loop(operation, timer, runs, warmup) / runs


def _args_to_arrays(dims, dtype, batch, extent) -> list[ArrayDescription]:
    shape = (batch,) + (extent,) * dims
    return [("input", shape, dtype), ("output", shape, dtype)]


def _microbenchmark(case: FFTCase) -> Callable[..., Any]:
    return microbenchmark(
        name=case.name,
        input_names={"dims": "fft_dims"},
        size_to_args=lambda size, dims, extent: {
            "batch": max(1, size // extent**dims)
        },
        args_to_arrays=_args_to_arrays,
        plan={"dims": case.dims, "extent": case.extent},
    )(_batched_fft)


class BatchedFFTSuite(MicrobenchmarkSuite):
    name = "batched_fft"

    def __init__(self, config, args) -> None:
        super().__init__(config, args)

        for case in _CASES:
            # have to prefix names because they start with numbers
            setattr(self, f"batched_fft_{case.name}", _microbenchmark(case))

    def dtypes(self) -> list[str]:
        return ["complex64", "complex128"]

    def run_suite(self, size_request: SizeRequest) -> None:
        # FFT is only supported on GPU in cuPyNumeric
        if self._config.package == ArrayPackage.LEGATE:
            from cupynumeric.runtime import runtime

            if runtime.num_gpus == 0:
                self.info("Skipping batched_fft suite: FFT requires GPU")
                return
        super().run_suite(size_request)
