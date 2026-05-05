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

from typing import Final, Literal, cast

from legate.util.settings import (
    EnvOnlySetting,
    PrioritizedSetting,
    Settings,
    convert_bool,
    convert_int,
)

__all__ = ("settings",)

DoctorFormat = Literal["plain", "json", "csv"]

BoundsCheckOperation = Literal["indexing", "take", "take_along_axis", "put"]

_BOUNDS_CHECK_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"indexing", "take", "take_along_axis", "put"}
)
_BOUNDS_CHECK_SENTINELS: Final[frozenset[str]] = frozenset({"all", "none"})


def convert_doctor_format(value: str) -> DoctorFormat:
    """Return a DoctorFormat value."""
    VALID = {"plain", "json", "csv"}
    v = value.lower()
    if v not in VALID:
        raise ValueError(
            f"unknown cuPyNumeric Doctor format: {value}, "
            f"valid values are: {VALID}"
        )
    return cast(DoctorFormat, v)


convert_doctor_format.type = (  # type: ignore [attr-defined]
    'DoctorFormat ("plain", "csv", or "json")'
)


def _parse_bounds_checking_tokens(value: str) -> frozenset[str]:
    tokens = frozenset(
        token.strip().lower() for token in value.split(",") if token.strip()
    )
    if not tokens:
        raise ValueError(
            "cuPyNumeric disabled bounds checking selector list cannot be empty"
        )

    invalid = tokens - _BOUNDS_CHECK_SENTINELS - _BOUNDS_CHECK_OPERATIONS
    if invalid:
        raise ValueError(
            "unknown cuPyNumeric disabled bounds checking selector(s): "
            f"{sorted(invalid)}; valid values are: "
            f"{sorted(_BOUNDS_CHECK_SENTINELS | _BOUNDS_CHECK_OPERATIONS)}"
        )

    sentinels = tokens & _BOUNDS_CHECK_SENTINELS
    if sentinels and len(tokens) > 1:
        raise ValueError(
            'cuPyNumeric disabled bounds checking selectors "all" and "none" '
            "cannot be combined with operation-specific selectors"
        )

    return tokens


def parse_bounds_checking(value: str) -> str:
    tokens = _parse_bounds_checking_tokens(value)
    if tokens == {"none"}:
        return "none"
    if tokens == {"all"}:
        return "all"
    return ",".join(sorted(tokens))


parse_bounds_checking.type = (  # type: ignore [attr-defined]
    'DisableBoundsChecking ("none", "all", or comma-separated selectors: '
    "indexing, take, take_along_axis, put)"
)


class CupynumericRuntimeSettings(Settings):
    doctor: PrioritizedSetting[bool] = PrioritizedSetting(
        "doctor",
        "CUPYNUMERIC_DOCTOR",
        default=False,
        convert=convert_bool,
        help="""
        Attempt to warn about certain usage patterns that are inefficient with
        cuPyNumeric.
        """,
    )

    doctor_format: PrioritizedSetting[DoctorFormat] = PrioritizedSetting(
        "doctor_format",
        "CUPYNUMERIC_DOCTOR_FORMAT",
        default="plain",
        convert=convert_doctor_format,
        help="""
        Format for cuPyNumeric ouput: plain, json, or csv.
        """,
    )

    doctor_filename: PrioritizedSetting[str | None] = PrioritizedSetting(
        "doctor_filename",
        "CUPYNUMERIC_DOCTOR_FILENAME",
        default=None,
        help="""
        A filename for a file to dump cuPyNumeric output to, otherwise stdout.
        """,
    )

    doctor_traceback: PrioritizedSetting[bool] = PrioritizedSetting(
        "doctor_filename",
        "CUPYNUMERIC_DOCTOR_TRACEBACK",
        default=False,
        convert=convert_bool,
        help="""
        Whether cuPyNumeric Doctor output should include full tracebacks.
        """,
    )

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

    fallback_stacktrace: EnvOnlySetting[bool] = EnvOnlySetting(
        "fallback_stacktrace",
        "CUPYNUMERIC_FALLBACK_STACKTRACE",
        default=False,
        convert=convert_bool,
        help="""
        Whether to dump a full stack trace whenever cuPyNumeric emits a
        warning about falling back to Numpy routines.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    fast_math: EnvOnlySetting[bool] = EnvOnlySetting(
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
        Minimum chunk size for GPU operations.

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
        Minimum chunk size for CPU operations.

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
        Minimum chunk size for OpenMP operations.

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

    # TODO(mpapadakis): This should really be parsing the exported "test"
    # setting from Legate (which can be set with LEGATE_TEST but also other
    # methods, which we're not checking here). Or we should not be depending
    # on that setting at all.
    test: EnvOnlySetting[bool] = EnvOnlySetting(
        "test",
        "LEGATE_TEST",
        default=False,
        convert=convert_bool,
        help="""
        Enable test mode. This sets alternative defaults for various other
        settings.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    take_default: PrioritizedSetting[str] = PrioritizedSetting(
        "take_default",
        "CUPYNUMERIC_TAKE_DEFAULT",
        default="auto",
        help="""
        Default algorithm for deferred array.take():
          - 'auto':  let cuPyNumeric decide which algorithm to use
          - 'index': use advanced indexing
          - 'task':  use a task that broadcasts the indices
        """,
    )

    disable_bounds_checking: PrioritizedSetting[str] = PrioritizedSetting(
        "disable_bounds_checking",
        "CUPYNUMERIC_DISABLE_BOUNDS_CHECKING",
        default="none",
        convert=parse_bounds_checking,
        help="""
        Disables explicit bounds checking for advanced-indexing-related
        operations.

          - 'none': disable no targeted explicit bounds checks
          - 'all':  disable all targeted explicit bounds checks
          - comma-separated selectors such as:
              'indexing,take,put'
            to disable checks only for the named operations
        """,
    )

    def bounds_check_enabled(self, operation: BoundsCheckOperation) -> bool:
        if operation not in _BOUNDS_CHECK_OPERATIONS:
            raise ValueError(
                f"unknown bounds checking operation: {operation}; "
                f"valid values are: {sorted(_BOUNDS_CHECK_OPERATIONS)}"
            )

        tokens = _parse_bounds_checking_tokens(self.disable_bounds_checking())
        if tokens == {"none"}:
            return True
        if tokens == {"all"}:
            return False
        return operation not in tokens

    use_nccl_gather: PrioritizedSetting[bool] = PrioritizedSetting(
        "use_nccl_gather",
        "CUPYNUMERIC_USE_NCCL_GATHER",
        default=False,
        convert=convert_bool,
        help="""
        Enable distributed gather via the NCCL all-to-all implementation when
        multiple GPUs are available.
        """,
    )

    use_nccl_scatter: PrioritizedSetting[bool] = PrioritizedSetting(
        "use_nccl_scatter",
        "CUPYNUMERIC_USE_NCCL_SCATTER",
        default=False,
        convert=convert_bool,
        help="""
        Enable distributed scatter via the NCCL all-to-all implementation when
        multiple GPUs are available.
        """,
    )


settings = CupynumericRuntimeSettings()
