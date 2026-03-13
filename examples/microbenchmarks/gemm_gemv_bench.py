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

import math

from _benchmark import MicrobenchmarkSuite

SKINNY_OUTER_DIM = 8


def _square_dim(size):
    return max(1, math.isqrt(size))


def _skinny_inner_dim(size):
    return max(1, size // SKINNY_OUTER_DIM)


def _make_matrix(array_module, rows, cols, dtype, start):
    values = array_module.arange(
        start, start + rows * cols, dtype=dtype
    ).reshape((rows, cols))
    return values / max(rows * cols, 1)


def _make_vector(array_module, size, dtype, start):
    values = array_module.arange(start, start + size, dtype=dtype)
    return values / max(size, 1)


def _get_case_dimensions(variant, size):
    if variant == "skinny_gemm":
        m = SKINNY_OUTER_DIM
        n = SKINNY_OUTER_DIM
        k = _skinny_inner_dim(size)
        return {"m": m, "n": n, "k": k}

    n = _square_dim(size)
    if variant == "square_gemm":
        return {"m": n, "n": n, "k": n}

    return {"m": n, "n": n}


def _initialize_case(array_module, variant, size, dtype):
    dimensions = _get_case_dimensions(variant, size)

    if variant == "skinny_gemm":
        m = dimensions["m"]
        n = dimensions["n"]
        k = dimensions["k"]
        a = _make_matrix(array_module, m, k, dtype, 1)
        b = _make_matrix(array_module, k, n, dtype, 2)
        c = array_module.zeros((m, n), dtype=dtype)
        return dimensions, (a, b, c)

    n = dimensions["n"]
    a = _make_matrix(array_module, n, n, dtype, 1)

    if variant == "square_gemm":
        b = _make_matrix(array_module, n, n, dtype, 2)
        c = array_module.zeros((n, n), dtype=dtype)
        return dimensions, (a, b, c)

    x = _make_vector(array_module, n, dtype, 2)
    y = array_module.zeros(n, dtype=dtype)
    return dimensions, (a, x, y)


def _get_check_tolerances(precision):
    if precision == 32:
        return 1e-4, 1e-6
    return 1e-8, 1e-10


def _check_case(variant, size, precision, result):
    import numpy as host_np

    dtype = host_np.float32 if precision == 32 else host_np.float64
    _, operands = _initialize_case(host_np, variant, size, dtype)

    if variant == "gemv":
        a, x, y = operands
        host_np.matmul(a, x, out=y)
        expected = y
    else:
        a, b, c = operands
        host_np.matmul(a, b, out=c)
        expected = c

    actual = (
        result.get() if hasattr(result, "get") else host_np.asarray(result)
    )
    rtol, atol = _get_check_tolerances(precision)
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
    np, variant, size, runs, warmup, precision, timer, perform_check
):
    dtype = np.float32 if precision == 32 else np.float64
    _, operands = _initialize_case(np, variant, size, dtype)

    if variant == "gemv":
        a, x, y = operands
    else:
        a, b, c = operands

    for idx in range(runs + warmup):
        if idx == warmup:
            timer.start()
        if variant == "gemv":
            np.matmul(a, x, out=y)
        else:
            np.matmul(a, b, out=c)
    total = timer.stop()

    if perform_check:
        _check_case(variant, size, precision, y if variant == "gemv" else c)

    return total


def _get_variants(variant):
    if variant == "all":
        return ["skinny_gemm", "square_gemm", "gemv"]
    return [variant]


def _get_precisions(precision):
    if precision == "all":
        return [32, 64]
    return [int(precision)]


def _precision_name(precision):
    return f"float{precision}"


def skinny_gemm(np, size, runs, warmup, precision, *, timer, perform_check):
    return run_gemm_gemv_case(
        np, "skinny_gemm", size, runs, warmup, precision, timer, perform_check
    )


def square_gemm(np, size, runs, warmup, precision, *, timer, perform_check):
    return run_gemm_gemv_case(
        np, "square_gemm", size, runs, warmup, precision, timer, perform_check
    )


def gemv(np, size, runs, warmup, precision, *, timer, perform_check):
    return run_gemm_gemv_case(
        np, "gemv", size, runs, warmup, precision, timer, perform_check
    )


def run_benchmarks(suite, size, variant, precision, perform_check):
    """Run GEMM/GEMV benchmarks inside the suite framework."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    variants = _get_variants(variant)
    precisions = _get_precisions(precision)
    funcs = [skinny_gemm, square_gemm, gemv]
    for func in funcs:
        if func.__name__ in variants:
            suite.run_timed(
                func,
                np,
                size,
                runs,
                warmup,
                precisions,
                timer=timer,
                perform_check=perform_check,
            )


class GemmSuite(MicrobenchmarkSuite):
    name = "gemm_gemv"

    def add_suite_parser_group(parser):
        group = parser.add_argument_group("GEMM/GEMV Suite")
        group.add_argument(
            "--gemm-gemv-variant",
            type=str,
            default="all",
            choices=["skinny_gemm", "square_gemm", "gemv", "all"],
            help="GEMM/GEMV variant to run (default: all)",
        )
        group.add_argument(
            "--gemm-gemv-precision",
            type=str,
            default="32",
            choices=["32", "64", "all"],
            help="GEMM/GEMV precision in bits (default: 32)",
        )
        group.add_argument(
            "--gemm-gemv-check",
            action="store_true",
            help="Validate GEMM/GEMV results after each timed sample",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.gemm_gemv_variant = args.gemm_gemv_variant
        self.gemm_gemv_precision = args.gemm_gemv_precision
        self.gemm_gemv_check = args.gemm_gemv_check

    def print_config(self):
        msg = [
            f"variant: {self.gemm_gemv_variant}",
            f"precision: {self.gemm_gemv_precision}",
            f"check: {self.gemm_gemv_check}",
        ]
        self.print_panel(msg, title="GEMM/GEMMV Suite")

    def run_suite(self, size):
        run_benchmarks(
            self,
            size,
            self.gemm_gemv_variant,
            self.gemm_gemv_precision,
            self.gemm_gemv_check,
        )
