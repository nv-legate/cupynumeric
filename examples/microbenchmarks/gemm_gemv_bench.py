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

import numpy as host_np

from _benchmark import MicrobenchmarkSuite, benchmark_info, timed_loop
from _benchmark.sizing import (
    SizeRequest,
    resolve_size_by_binary_search,
    resolve_suite_size,
)

SKINNY_OUTER_DIM = 8


def _dtype_bytes(dtype) -> int:
    import numpy

    return numpy.dtype(dtype).itemsize


def _square_dim(size):
    return max(1, math.isqrt(size))


def _skinny_inner_dim(size):
    return max(1, size // SKINNY_OUTER_DIM)


def _make_matrix(array_module, rows, cols, dtype):
    return array_module.random.rand(rows, cols).astype(dtype)


def _make_vector(array_module, size, dtype):
    return array_module.random.rand(size).astype(dtype)


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


def _estimate_case_working_set_bytes(variant, dtype, size):
    dimensions = _get_case_dimensions(variant, size)
    itemsize = _dtype_bytes(dtype)

    if variant == "skinny_gemm":
        m = dimensions["m"]
        n = dimensions["n"]
        k = dimensions["k"]
        return itemsize * (2 * m * k * n)

    n = dimensions["n"]
    if variant == "square_gemm":
        return itemsize * 2 * n * n * n

    # gemv
    return itemsize * 2 * n * n


def _estimate_working_set_bytes(variant, precision, size):
    variants = _get_variants(variant)
    dtypes = _get_dtypes(precision)
    return max(
        _estimate_case_working_set_bytes(case, dtype, size)
        for case in variants
        for dtype in dtypes
    )


def _resolve_size_from_memory_target(variant, precision, target_bytes):
    return resolve_size_by_binary_search(
        target_bytes,
        estimate_working_set_bytes=lambda size: _estimate_working_set_bytes(
            variant, precision, size
        ),
        initial_guess=8,
    )


def _describe_size(variant, precision, size):
    variants = _get_variants(variant)
    dtypes = _get_dtypes(precision)
    lines = [
        f"variants: {', '.join(variants)}",
        f"dtypes:   {', '.join(dtypes)}",
    ]
    for case in variants:
        dimensions = _get_case_dimensions(case, size)
        if case in {"skinny_gemm", "square_gemm"}:
            lines.append(
                f"{case}: "
                f"m={dimensions['m']}, "
                f"n={dimensions['n']}, "
                f"k={dimensions['k']}"
            )
        else:
            lines.append(f"gemv: m={dimensions['m']}, n={dimensions['n']}")
    return lines


def _initialize_case(array_module, variant, size, dtype):
    dimensions = _get_case_dimensions(variant, size)

    if variant == "skinny_gemm":
        m = dimensions["m"]
        n = dimensions["n"]
        k = dimensions["k"]
        a = _make_matrix(array_module, m, k, dtype)
        b = _make_matrix(array_module, k, n, dtype)
        c = array_module.zeros((m, n), dtype=dtype)
        return dimensions, (a, b, c)

    n = dimensions["n"]
    a = _make_matrix(array_module, n, n, dtype)

    if variant == "square_gemm":
        b = _make_matrix(array_module, n, n, dtype)
        c = array_module.zeros((n, n), dtype=dtype)
        return dimensions, (a, b, c)

    x = _make_vector(array_module, n, dtype)
    y = array_module.zeros(n, dtype=dtype)
    return dimensions, (a, x, y)


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
    _, operands = _initialize_case(np, variant, size, dtype)

    a, b, c = operands

    def operation():
        np.matmul(a, b, out=c)

    total = timed_loop(operation, timer, runs, warmup) / runs

    if perform_check:
        expected = host_np.matmul(_to_host(a), _to_host(b))
        _check_case(variant, dtype, c, expected)

    return ((a.shape, b.shape), total)


def _get_variants(variant):
    if variant == "all":
        return ["skinny_gemm", "square_gemm", "gemv"]
    return [variant]


def _get_dtypes(precision):
    if precision == "all":
        return ["float32", "float64"]
    else:
        assert precision in ["32", "64"]
        return [f"float{precision}"]


_INFO = {
    "output_names": ["input shapes", "time per run (ms)"],
    "returns_time": 1,
}


@benchmark_info(**_INFO)
def skinny_gemm(np, size, runs, warmup, dtype, *, timer, perform_check):
    return run_gemm_gemv_case(
        np, "skinny_gemm", size, runs, warmup, dtype, timer, perform_check
    )


@benchmark_info(**_INFO)
def square_gemm(np, size, runs, warmup, dtype, *, timer, perform_check):
    return run_gemm_gemv_case(
        np, "square_gemm", size, runs, warmup, dtype, timer, perform_check
    )


@benchmark_info(**_INFO)
def gemv(np, size, runs, warmup, dtype, *, timer, perform_check):
    return run_gemm_gemv_case(
        np, "gemv", size, runs, warmup, dtype, timer, perform_check
    )


def run_benchmarks(suite, size_request, variant, precision, perform_check):
    """Run GEMM/GEMV benchmarks inside the suite framework."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    sizes, resolutions = resolve_suite_size(
        size_request,
        resolve_from_target=lambda target_bytes: (
            _resolve_size_from_memory_target(variant, precision, target_bytes)
        ),
        estimate_working_set_bytes=lambda resolved_size: (
            _estimate_working_set_bytes(variant, precision, resolved_size)
        ),
        describe_size=lambda resolved_size: _describe_size(
            variant, precision, resolved_size
        ),
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    variants = _get_variants(variant)
    dtypes = _get_dtypes(precision)
    funcs = [skinny_gemm, square_gemm, gemv]
    for func in funcs:
        if func.__name__ in variants:
            suite.run_timed(
                func,
                np,
                sizes,
                runs,
                warmup,
                dtypes,
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
            help="GEMM/GEMV variant to run",
        )
        group.add_argument(
            "--gemm-gemv-precision",
            type=str,
            default="32",
            choices=["32", "64", "all"],
            help="GEMM/GEMV precision in bits",
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

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(
            self,
            size_request,
            self.gemm_gemv_variant,
            self.gemm_gemv_precision,
            self.gemm_gemv_check,
        )
