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
GEMM/GEMV microbenchmark suite.

Requested variants from issue #1534:
1. skinny_gemm: small outer dimensions with a large reduction dimension
2. square_gemm: large square matrix multiplication
3. gemv: matrix-vector multiplication

The shared ``--size`` flag is interpreted per variant:
- skinny_gemm uses ``m=n=8`` and ``k=max(1, size // 8)``
- square_gemm and gemv use ``n=max(1, floor(sqrt(size)))``
"""

from __future__ import annotations

from typing import Any

import numpy as host_np

from _benchmark import (
    MicrobenchmarkSuite,
    microbenchmark,
    nthroot,
    random_array,
    timed_loop,
)

SKINNY_OUTER_DIM = 8


def _square_dim(size):
    return nthroot(size, 2)


def _skinny_inner_dim(size):
    return max(1, size // SKINNY_OUTER_DIM)


def _make_matrix(array_module, rows, cols, dtype):
    return random_array(array_module, (rows, cols), dtype)


def _make_vector(array_module, size, dtype):
    return random_array(array_module, size, dtype)


def _get_case_dimensions(variant, size) -> tuple[int, int, int]:
    if variant == "skinny_gemm":
        m = SKINNY_OUTER_DIM
        k = _skinny_inner_dim(size)
        return (m, k, m)

    m = _square_dim(size)
    if variant == "square_gemm":
        return (m, m, m)

    assert variant == "gemv"
    return (m, m, 1)


def _args_to_arrays(variant, dtype, size):
    m, k, n = _get_case_dimensions(variant, size)
    if variant == "gemv":
        return [("A", (m, k), dtype), ("x", k, dtype), ("y", m, dtype)]
    return [("A", (m, k), dtype), ("B", (k, n), dtype), ("C", (m, n), dtype)]


def _args_to_work(variant, size):
    m, k, n = _get_case_dimensions(variant, size)
    return 2 * m * k * n


def _initialize_case(array_module, variant, size, dtype):
    m, k, n = _get_case_dimensions(variant, size)

    a = _make_matrix(array_module, m, k, dtype)

    if variant == "gemv":
        x = _make_vector(array_module, k, dtype)
        y = array_module.zeros(m, dtype=dtype)
        return (a, x, y)

    b = _make_matrix(array_module, k, n, dtype)
    c = array_module.zeros((m, n), dtype=dtype)
    return (a, b, c)


def _get_check_tolerances(dtype):
    if dtype == "float32":
        return 1e-4, 1e-6
    return 1e-8, 1e-10


def _to_host(array) -> host_np.ndarray:
    return array.get() if hasattr(array, "get") else host_np.asarray(array)


def _check_case(variant, dtype, result, expected):
    actual = _to_host(result)
    rtol, atol = _get_check_tolerances(dtype)
    if not host_np.allclose(actual, expected, rtol=rtol, atol=atol):
        abs_diff = host_np.abs(actual - expected)
        rel_diff = abs_diff / host_np.maximum(host_np.abs(expected), atol)
        raise AssertionError(
            f"{variant} result mismatch: "
            f"max_abs={float(abs_diff.max())}, "
            f"max_rel={float(rel_diff.max())}, "
            f"rtol={rtol}, atol={atol}"
        )


def run_gemm_gemv_case(
    np, variant, size, runs, warmup, dtype, timer, perform_check
):
    a, b, c = _initialize_case(np, variant, size, dtype)

    def operation():
        np.matmul(a, b, out=c)

    total = timed_loop(operation, timer, runs, warmup) / runs

    if perform_check:
        expected = host_np.matmul(_to_host(a), _to_host(b))
        _check_case(variant, dtype, c, expected)

    return ((a.shape, b.shape), total)


def _microbenchmark(variant: str):
    return microbenchmark(
        output_names=["input shapes", "time per run (ms)"],
        returns_time=1,
        args_to_arrays=lambda dtype, size: _args_to_arrays(
            variant, dtype, size
        ),
        args_to_work=lambda size: _args_to_work(variant, size),
    )


class GemmSuite(MicrobenchmarkSuite):
    name = "gemm_gemv"

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("GEMM/GEMV Suite")
        group.add_argument(
            "--gemm-gemv-check",
            action="store_true",
            help="Validate GEMM/GEMV results after each timed sample",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.gemm_gemv_check = args.gemm_gemv_check

    def print_config(self):
        msg = [f"check: {self.gemm_gemv_check}"]
        self.print_panel(msg, title="GEMM/GEMMV Suite")

    def dtypes(self) -> list[str]:
        return ["float32", "float64"]

    def default_arguments(self) -> dict[str, Any]:
        return {
            **super().default_arguments(),
            "perform_check": self.gemm_gemv_check,
        }

    @_microbenchmark("skinny_gemm")
    def skinny_gemm(np, size, runs, warmup, dtype, *, timer, perform_check):
        return run_gemm_gemv_case(
            np, "skinny_gemm", size, runs, warmup, dtype, timer, perform_check
        )

    @_microbenchmark("square_gemm")
    def square_gemm(np, size, runs, warmup, dtype, *, timer, perform_check):
        return run_gemm_gemv_case(
            np, "square_gemm", size, runs, warmup, dtype, timer, perform_check
        )

    @_microbenchmark("gemv")
    def gemv(np, size, runs, warmup, dtype, *, timer, perform_check):
        return run_gemm_gemv_case(
            np, "gemv", size, runs, warmup, dtype, timer, perform_check
        )
