# Copyright 2024 NVIDIA Corporation
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
from __future__ import annotations

from legate.util.settings import (
    EnvOnlySetting,
    PrioritizedSetting,
    Settings,
    convert_bool,
    convert_int,
)

__all__ = ("settings",)


class CupynumericRuntimeSettings(Settings):
    preload_cudalibs: PrioritizedSetting[bool] = PrioritizedSetting(
        "preload_cudalibs",
        "CUPYNUMERIC_PRELOAD_CUDALIBS",
        default=False,
        convert=convert_bool,
        help="""
        Preload and initialize handles of all CUDA libraries (cuBLAS, cuSOLVER,
        etc.) used in cuPyNumeric.
        """,
    )

    warn: PrioritizedSetting[bool] = PrioritizedSetting(
        "warn",
        "CUPYNUMERIC_WARN",
        default=False,
        convert=convert_bool,
        help="""
        Turn on warnings.
        """,
    )

    report_coverage: PrioritizedSetting[bool] = PrioritizedSetting(
        "report_coverage",
        "CUPYNUMERIC_REPORT_COVERAGE",
        default=False,
        convert=convert_bool,
        help="""
        Print an overall percentage of cupynumeric coverage.
        """,
    )

    report_dump_callstack: PrioritizedSetting[bool] = PrioritizedSetting(
        "report_dump_callstack",
        "CUPYNUMERIC_REPORT_DUMP_CALLSTACK",
        default=False,
        convert=convert_bool,
        help="""
        Print an overall percentage of cupynumeric coverage with a call stack.
        """,
    )

    report_dump_csv: PrioritizedSetting[str | None] = PrioritizedSetting(
        "report_dump_csv",
        "CUPYNUMERIC_REPORT_DUMP_CSV",
        default=None,
        help="""
        Save a coverage report to a specified CSV file.
        """,
    )

    numpy_compat: PrioritizedSetting[bool] = PrioritizedSetting(
        "numpy_compat",
        "CUPYNUMERIC_NUMPY_COMPATIBILITY",
        default=False,
        convert=convert_bool,
        help="""
        cuPyNumeric will issue additional tasks to match numpy's results
        and behavior. This is currently used in the following
        APIs: nanmin, nanmax, nanargmin, nanargmax
        """,
    )

    fast_math: EnvOnlySetting[int] = EnvOnlySetting(
        "fast_math",
        "CUPYNUMERIC_FAST_MATH",
        default=False,
        convert=convert_bool,
        help="""
        Enable certain optimized execution modes for floating-point math
        operations, that may violate strict IEEE specifications. Currently this
        flag enables the acceleration of single-precision cuBLAS routines using
        TF32 tensor cores.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_gpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_gpu_chunk",
        "CUPYNUMERIC_MIN_GPU_CHUNK",
        default=65536,  # 1 << 16
        test_default=2,
        convert=convert_int,
        help="""
        Legate will fall back to vanilla NumPy when handling arrays smaller
        than this, rather than attempt to accelerate using GPUs, as the
        offloading overhead would likely not be offset by the accelerated
        operation code.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_cpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_cpu_chunk",
        "CUPYNUMERIC_MIN_CPU_CHUNK",
        default=1024,  # 1 << 10
        test_default=2,
        convert=convert_int,
        help="""
        Legate will fall back to vanilla NumPy when handling arrays smaller
        than this, rather than attempt to accelerate using native CPU code, as
        the offloading overhead would likely not be offset by the accelerated
        operation code.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_omp_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_omp_chunk",
        "CUPYNUMERIC_MIN_OMP_CHUNK",
        default=8192,  # 1 << 13
        test_default=2,
        convert=convert_int,
        help="""
        Legate will fall back to vanilla NumPy when handling arrays smaller
        than this, rather than attempt to accelerate using OpenMP, as the
        offloading overhead would likely not be offset by the accelerated
        operation code.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    force_thunk: EnvOnlySetting[str | None] = EnvOnlySetting(
        "force_thunk",
        "CUPYNUMERIC_FORCE_THUNK",
        default=None,
        test_default="deferred",
        help="""
        Force cuPyNumeric to always use a specific strategy for backing
        ndarrays: "deferred", i.e. managed by the Legate runtime, which
        enables distribution and accelerated operations, but has some
        up-front offloading overhead, or "eager", i.e. falling back to
        using a vanilla NumPy array. By default cuPyNumeric will decide
        this on a per-array basis, based on the size of the array and
        the accelerator in use.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    matmul_cache_size: EnvOnlySetting[int] = EnvOnlySetting(
        "matmul_cache_size",
        "CUPYNUMERIC_MATMUL_CACHE_SIZE",
        default=134217728,  # 128MB
        test_default=4096,  # 4KB
        convert=convert_int,
        help="""
        Force cuPyNumeric to keep temporary task slices during matmul
        computations smaller than this threshold. Whenever the temporary
        space needed during computation would exceed this value the task
        will be batched over 'k' to fulfill the requirement.

        This is a read-only environment variable setting used by the runtime.
        """,
    )


settings = CupynumericRuntimeSettings()
