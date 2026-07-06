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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as host_np

from _benchmark import (
    MicrobenchmarkSuite,
    microbenchmark,
    nthroot,
    random_array,
    timed_loop,
)


if TYPE_CHECKING:
    from typing import Callable
    from _benchmark import ArrayDescription

_CASES = {
    "axis_last": {"axis": 1, "rank": 2, "transpose": False},
    "axis_first": {"axis": 0, "rank": 2, "transpose": False},
    "strided_last": {"axis": 1, "rank": 2, "transpose": True},
    "axis_middle_3d": {"axis": 1, "rank": 3, "transpose": False},
    "multi_axis": {"axis": (0, 2), "rank": 3, "transpose": False},
}


def _to_host(array: Any) -> host_np.ndarray:
    return array.get() if hasattr(array, "get") else host_np.asarray(array)


def _matrix_shape(size: int) -> dict[str, tuple[int, ...]]:
    size = max(size, 1)
    n = nthroot(size, 2, lower_bound=2)
    return {"shape": (n,) * 2}


def _tensor_shape(size: int) -> dict[str, tuple[int, ...]]:
    size = max(size, 1)
    n = nthroot(size, 3, lower_bound=2)
    return {"shape": (n,) * 3}


RAND_SCALE_FACTOR = 100


def _make_array(array_module, shape: tuple[int, ...], dtype_name: str):
    return random_array(
        array_module, shape, dtype_name, shift=-0.5, scale=RAND_SCALE_FACTOR
    )


def _alloc_shape(shape: tuple[int, ...], transpose: bool) -> tuple[int, ...]:
    return tuple(reversed(shape)) if transpose else shape


def _output_shape(
    shape: tuple[int, ...], axis: int | tuple[int, ...]
) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else axis
    return tuple(
        extent for axis_idx, extent in enumerate(shape) if axis_idx not in axes
    )


def _output_dtype(dtype: str) -> str:
    return "int64" if dtype == "int32" else dtype


# Reductions accumulate rounding error across many terms, so tolerate more
# drift than the elementwise ufunc checks in peer suites.
FLOAT_RTOL = 2e-3
FLOAT_ATOL = 1e-4


def _check_result(name: str, dtype: str, actual, expected) -> None:
    actual_host = _to_host(actual)
    if dtype == "int32":
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


def _axis_sum(
    np,
    case: str,
    dtype: str,
    axis: int | tuple[int, ...],
    transpose: bool,
    shape: tuple[int, ...],
    runs: int,
    warmup: int,
    *,
    timer,
    perform_check: bool = False,
):
    alloc_shape = _alloc_shape(shape, transpose)
    src = _make_array(np, alloc_shape, dtype)
    if transpose:
        src = src.T

    out_shape = _output_shape(shape, axis)
    out_dtype = _output_dtype(dtype)
    out = np.empty(out_shape, dtype=out_dtype)

    def operation():
        np.sum(src, axis=axis, out=out)

    total = timed_loop(operation, timer, runs, warmup) / runs
    if perform_check:
        expected = host_np.sum(_to_host(src), axis=axis)
        _check_result(case, dtype, out, expected)
    return total


def _args_to_arrays(
    shape: tuple[int, ...], dtype: str, axis: int | tuple[int, ...]
) -> list[ArrayDescription]:
    out_shape = _output_shape(shape, axis)
    out_dtype = _output_dtype(dtype)
    if dtype == "float32":
        # some backends use float64 to initially create random arrays, even
        # when float32 is requested
        dtype = "float64"
    return [("input", shape, dtype), ("output", out_shape, out_dtype)]


class AxisSumSuite(MicrobenchmarkSuite):
    name = "axis_sum"

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("Axis-wise Sum Suite")
        group.add_argument(
            "--axis-sum-check",
            action="store_true",
            help="Validate axis-wise sum results after each timed sample",
        )

    def dtypes(self) -> list[str]:
        return ["int32", "float32", "float64"]

    def _microbenchmark(self, name: str) -> Callable[..., Any]:
        case = _CASES[name]
        return microbenchmark(
            name=name,
            input_names={"transpose": "layout"},
            formats={
                "layout": lambda transpose: (
                    "transposed" if transpose else "contiguous"
                )
            },
            size_to_args=_matrix_shape if case["rank"] == 2 else _tensor_shape,
            args_to_arrays=_args_to_arrays,
            args_to_work=lambda shape: float(host_np.prod(shape)),
            plan={
                "case": name,
                "axis": case["axis"],
                "transpose": case["transpose"],
            },
        )(_axis_sum)

    def default_arguments(self) -> dict[str, Any]:
        return {
            **super().default_arguments(),
            "perform_check": self.axis_sum_check,
        }

    def __init__(self, config, args):
        super().__init__(config, args)
        self.axis_sum_check = args.axis_sum_check

        for name in _CASES:
            setattr(self, name, self._microbenchmark(name))

    def print_config(self):
        msg = [f"check: {self.axis_sum_check}"]
        self.print_panel(msg, "Axis-wise Sum Suite")
