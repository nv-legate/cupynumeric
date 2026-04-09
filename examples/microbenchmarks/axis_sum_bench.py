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
Axis-wise sum microbenchmark suite.

Representative cases for distinct axis-wise reduction paths:
1. contiguous last-axis reduction on a C-order 2D array
2. contiguous first-axis reduction on a C-order 2D array
3. strided last-axis reduction on a transposed 2D view
4. middle-axis reduction on a C-order 3D array
5. multi-axis reduction on a 3D array with non-scalar output
6. representative float32 reduction on a C-order 2D array
7. integer accumulation on a C-order 2D array
"""

from __future__ import annotations

import math
from typing import Any

import numpy as host_np

from _benchmark import (
    MicrobenchmarkSuite,
    benchmark_info,
    format_dtype,
    get_benchmark_info,
    timed_loop,
)
from _benchmark.sizing import (
    SizeRequest,
    resolve_size_by_binary_search,
    resolve_suite_size,
)

# Reductions accumulate rounding error across many terms, so tolerate more
# drift than the elementwise ufunc checks in peer suites.
FLOAT_RTOL = 5e-5
FLOAT_ATOL = 1e-6

_CASES = {
    "axis_last": {
        "axis": 1,
        "dtype": "float64",
        "rank": 2,
        "transpose": False,
    },
    "axis_first": {
        "axis": 0,
        "dtype": "float64",
        "rank": 2,
        "transpose": False,
    },
    "strided_last": {
        "axis": 1,
        "dtype": "float64",
        "rank": 2,
        "transpose": True,
    },
    "axis_middle_3d": {
        "axis": 1,
        "dtype": "float64",
        "rank": 3,
        "transpose": False,
    },
    "multi_axis": {
        "axis": (0, 2),
        "dtype": "float64",
        "rank": 3,
        "transpose": False,
    },
    "float32_axis_last": {
        "axis": 1,
        "dtype": "float32",
        "rank": 2,
        "transpose": False,
    },
    "int_axis_last": {
        "axis": 1,
        "dtype": "int32",
        "rank": 2,
        "transpose": False,
    },
}

_INITIAL_BYTES_PER_SIZE = host_np.dtype("float64").itemsize


def _to_host(array: Any) -> host_np.ndarray:
    return array.get() if hasattr(array, "get") else host_np.asarray(array)


def _matrix_shape(size: int) -> tuple[int, int]:
    size = max(size, 1)
    rows = max(2, math.isqrt(size))
    return rows, max(2, math.ceil(size / rows))


def _tensor_shape(size: int) -> tuple[int, int, int]:
    size = max(size, 1)
    planes = max(2, round(size ** (1 / 3)))
    rows = max(2, planes)
    return planes, rows, max(2, math.ceil(size / (planes * rows)))


def _make_array(array_module, shape: tuple[int, ...], dtype_name: str):
    total = math.prod(shape)
    values = array_module.arange(
        total, dtype=getattr(array_module, dtype_name)
    )
    values = values.reshape(shape)
    if dtype_name == "int32":
        return values % 97 - 48
    return values / total


def _base_shape(case: dict[str, Any], size: int) -> tuple[int, ...]:
    if case["rank"] == 2:
        return _matrix_shape(size)
    return _tensor_shape(size)


def _input_shape(case: dict[str, Any], size: int) -> tuple[int, ...]:
    shape = _base_shape(case, size)
    return tuple(reversed(shape)) if case["transpose"] else shape


def _get_cases(case: str) -> list[str]:
    if case == "all":
        return [*_CASES]
    return [case]


def _case_axes(case_name: str) -> tuple[int, ...]:
    axis = _CASES[case_name]["axis"]
    return (axis,) if isinstance(axis, int) else axis


def _normalized_axes(
    axis: int | tuple[int, ...], ndim: int
) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else axis
    return tuple(axis_idx % ndim for axis_idx in axes)


def _output_shape(case_name: str, size: int) -> tuple[int, ...]:
    shape = _input_shape(_CASES[case_name], size)
    axes = _normalized_axes(_case_axes(case_name), len(shape))
    return tuple(
        extent for axis_idx, extent in enumerate(shape) if axis_idx not in axes
    )


def _output_dtype_name(dtype_name: str) -> str:
    return "int64" if dtype_name == "int32" else dtype_name


def _estimate_case_working_set_bytes(case_name: str, size: int) -> int:
    case = _CASES[case_name]
    input_shape = _input_shape(case, size)
    output_shape = _output_shape(case_name, size)
    input_bytes = (
        math.prod(input_shape) * host_np.dtype(case["dtype"]).itemsize
    )
    output_bytes = (
        math.prod(output_shape)
        * host_np.dtype(_output_dtype_name(case["dtype"])).itemsize
    )
    return input_bytes + output_bytes


def _estimate_working_set_bytes(case: str, size: int) -> int:
    return max(
        _estimate_case_working_set_bytes(case_name, size)
        for case_name in _get_cases(case)
    )


def _resolve_size_from_memory_target(case: str, target_bytes: int) -> int:
    return resolve_size_by_binary_search(
        target_bytes,
        estimate_working_set_bytes=lambda size: _estimate_working_set_bytes(
            case, size
        ),
        initial_guess=target_bytes // _INITIAL_BYTES_PER_SIZE,
    )


def _format_shape(shape: tuple[int, ...]) -> str:
    return "scalar" if not shape else " x ".join(str(dim) for dim in shape)


def _describe_size(size: int, case: str) -> list[str]:
    lines = [f"cases: {', '.join(_get_cases(case))}"]
    for case_name in _get_cases(case):
        input_shape = _input_shape(_CASES[case_name], size)
        output_shape = _output_shape(case_name, size)
        lines.append(
            f"{case_name}: input={_format_shape(input_shape)}, "
            f"output={_format_shape(output_shape)}"
        )
    return lines


def _check_result(name: str, dtype_name: str, actual, expected) -> None:
    actual_host = _to_host(actual)
    if dtype_name == "int32":
        if not host_np.array_equal(actual_host, expected):
            raise AssertionError(f"{name} mismatch")
        return

    if not host_np.allclose(
        actual_host, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL
    ):
        abs_diff = host_np.abs(actual_host - expected)
        rel_diff = abs_diff / host_np.maximum(
            host_np.abs(expected), FLOAT_ATOL
        )
        raise AssertionError(
            f"{name} mismatch: "
            f"max_abs={float(abs_diff.max())}, "
            f"max_rel={float(rel_diff.max())}, "
            f"rtol={FLOAT_RTOL}, atol={FLOAT_ATOL}"
        )


@benchmark_info(
    input_names={
        "case_name": "case",
        "dtype_name": "dtype",
        "transpose": "layout",
    },
    formats={
        "dtype": format_dtype,
        "axis": lambda axis: axis if isinstance(axis, int) else str(axis),
        "layout": lambda transpose: (
            "transposed" if transpose else "contiguous"
        ),
    },
)
def axis_sum(
    np,
    case_name: str,
    dtype_name: str,
    axis: int | tuple[int, ...],
    transpose: bool,
    shape: tuple[int, ...],
    size: int,
    runs: int,
    warmup: int,
    alloc_shape: tuple[int, ...] | None,
    *,
    timer,
    perform_check: bool = False,
):
    if alloc_shape is None:
        alloc_shape = shape
    src = _make_array(np, alloc_shape, dtype_name)
    if transpose:
        src = src.T

    axes = _normalized_axes(axis, src.ndim)
    out_shape = tuple(
        extent
        for axis_idx, extent in enumerate(src.shape)
        if axis_idx not in axes
    )
    out_dtype = np.int64 if dtype_name == "int32" else getattr(np, dtype_name)
    out = np.empty(out_shape, dtype=out_dtype)

    def operation():
        np.sum(src, axis=axis, out=out)

    total = timed_loop(operation, timer, runs, warmup) / runs
    if perform_check:
        expected = host_np.sum(_to_host(src), axis=axis)
        _check_result(case_name, dtype_name, out, expected)
    return total


def _case_args(case_name: str, size: int) -> tuple[Any, ...]:
    case = _CASES[case_name]
    return (
        case_name,
        case["dtype"],
        case["axis"],
        case["transpose"],
        _input_shape(case, size),
    )


def run_benchmarks(suite, size_request, *, case="all", perform_check=False):
    """Run representative axis-wise sum benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    info = get_benchmark_info(axis_sum)
    sizes, resolutions = resolve_suite_size(
        size_request,
        resolve_from_target=lambda target_bytes: (
            _resolve_size_from_memory_target(case, target_bytes)
        ),
        estimate_working_set_bytes=lambda resolved_size: (
            _estimate_working_set_bytes(case, resolved_size)
        ),
        describe_size=lambda resolved_size: _describe_size(
            resolved_size, case
        ),
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    for case_name in _get_cases(case):

        def arg_gen():
            for size in sizes:
                alloc_shape = _base_shape(_CASES[case_name], size)
                yield (
                    np,
                    *_case_args(case_name, size),
                    size,
                    runs,
                    warmup,
                    alloc_shape,
                )

        suite.run_timed_with_generator(
            info.replace(name=case_name),
            axis_sum,
            arg_gen(),
            timer=timer,
            perform_check=perform_check,
        )


class AxisSumSuite(MicrobenchmarkSuite):
    name = "axis_sum"

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("Axis-wise Sum Suite")
        group.add_argument(
            "--axis-sum-case",
            type=str,
            default="all",
            choices=[*_CASES, "all"],
            help="Axis-wise sum case to run",
        )
        group.add_argument(
            "--axis-sum-check",
            action="store_true",
            help="Validate axis-wise sum results after each timed sample",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.axis_sum_case = args.axis_sum_case
        self.axis_sum_check = args.axis_sum_check

    def print_config(self):
        msg = [f"case: {self.axis_sum_case}", f"check: {self.axis_sum_check}"]
        self.print_panel(msg, "Axis-wise Sum Suite")

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(
            self,
            size_request,
            case=self.axis_sum_case,
            perform_check=self.axis_sum_check,
        )
