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
Ufunc microbenchmark suite.

Representative cases for distinct ufunc execution paths:
1. unary math: ``exp(x)``
2. binary math: ``add(a, b)``
3. broadcasted binary math: ``add(a_2d, b_1d)``
4. mixed-signature binary math: ``ldexp(x, exp)``
5. comparison: ``greater(a, b)``
6. logical: ``logical_and(mask_a, mask_b)``
7. multi-output unary: ``frexp(x)``

All cases use preallocated ``out=`` buffers when possible so the timing focuses
on ufunc execution rather than output allocation.
"""

from __future__ import annotations

import math

import numpy as host_np

from microbenchmark_utilities import create_benchmark_function


def _float_input(array_module, size: int, *, start: float = 1.0):
    values = array_module.arange(size, dtype=array_module.float32)
    return (values + start) / max(size, 1)


def _bool_input(array_module, size: int, *, offset: int):
    values = array_module.arange(size, dtype=array_module.int32) + offset
    return values % 2 == 0


def _broadcast_shape(size: int) -> tuple[int, int]:
    cols = max(1, math.isqrt(size))
    rows = max(1, size // cols)
    return rows, cols


def _to_host(array):
    return array.get() if hasattr(array, "get") else host_np.asarray(array)


def _check_allclose(name: str, actual, expected) -> None:
    actual_host = _to_host(actual)
    if not host_np.allclose(actual_host, expected, rtol=1e-6, atol=1e-7):
        abs_diff = host_np.abs(actual_host - expected)
        rel_diff = abs_diff / host_np.maximum(host_np.abs(expected), 1e-7)
        raise AssertionError(
            f"{name} mismatch: "
            f"max_abs={float(abs_diff.max())}, "
            f"max_rel={float(rel_diff.max())}"
        )


def _check_equal(name: str, actual, expected) -> None:
    actual_host = _to_host(actual)
    if not host_np.array_equal(actual_host, expected):
        raise AssertionError(f"{name} mismatch")


def run_unary_exp(np, timer, size, runs, warmup, *, perform_check=False):
    x = _float_input(np, size)
    out = np.empty_like(x)

    def operation():
        np.exp(x, out=out)

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected = host_np.exp(_to_host(x))
        _check_allclose("exp", out, expected)
    return total


def run_binary_add(np, timer, size, runs, warmup, *, perform_check=False):
    lhs = _float_input(np, size)
    rhs = _float_input(np, size, start=2.0)
    out = np.empty_like(lhs)

    def operation():
        np.add(lhs, rhs, out=out)

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected = _to_host(lhs) + _to_host(rhs)
        _check_allclose("add", out, expected)
    return total


def run_binary_add_broadcast(
    np, timer, size, runs, warmup, *, perform_check=False
):
    rows, cols = _broadcast_shape(size)
    lhs = _float_input(np, rows * cols).reshape((rows, cols))
    rhs = _float_input(np, cols, start=3.0)
    out = np.empty_like(lhs)

    def operation():
        np.add(lhs, rhs, out=out)

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected = _to_host(lhs) + _to_host(rhs)
        _check_allclose("add_broadcast", out, expected)
    return total


def run_comparison_greater(
    np, timer, size, runs, warmup, *, perform_check=False
):
    lhs = _float_input(np, size, start=2.0)
    rhs = _float_input(np, size, start=1.0)
    out = np.empty(lhs.shape, dtype=np.bool_)

    def operation():
        np.greater(lhs, rhs, out=out)

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected = _to_host(lhs) > _to_host(rhs)
        _check_equal("greater", out, expected)
    return total


def run_binary_ldexp(np, timer, size, runs, warmup, *, perform_check=False):
    lhs = _float_input(np, size, start=2.0)
    rhs = np.arange(size, dtype=np.int32) % 6
    out = np.empty_like(lhs)

    def operation():
        np.ldexp(lhs, rhs, out=out)

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected = host_np.ldexp(_to_host(lhs), _to_host(rhs))
        _check_allclose("ldexp", out, expected)
    return total


def run_logical_and(np, timer, size, runs, warmup, *, perform_check=False):
    lhs = _bool_input(np, size, offset=0)
    rhs = _bool_input(np, size, offset=1)
    out = np.empty(lhs.shape, dtype=np.bool_)

    def operation():
        np.logical_and(lhs, rhs, out=out)

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected = host_np.logical_and(_to_host(lhs), _to_host(rhs))
        _check_equal("logical_and", out, expected)
    return total


def run_multiout_frexp(np, timer, size, runs, warmup, *, perform_check=False):
    x = _float_input(np, size, start=4.0)
    mantissa = np.empty_like(x)
    exponent = np.empty(x.shape, dtype=np.int32)

    def operation():
        np.frexp(x, out=(mantissa, exponent))

    total = create_benchmark_function(np, timer, operation, runs, warmup)()
    if perform_check:
        expected_mantissa, expected_exponent = host_np.frexp(_to_host(x))
        _check_allclose("frexp_mantissa", mantissa, expected_mantissa)
        _check_equal("frexp_exponent", exponent, expected_exponent)
    return total


def run_benchmarks(suite, size, *, perform_check=False):
    """Run representative ufunc benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    suite.run_single_benchmark(
        name="unary_exp",
        bench_func=lambda: run_unary_exp(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={"size": size, "category": "unary", "op": "exp"},
    )
    suite.run_single_benchmark(
        name="binary_add",
        bench_func=lambda: run_binary_add(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={"size": size, "category": "binary", "op": "add"},
    )

    rows, cols = _broadcast_shape(size)
    suite.run_single_benchmark(
        name="binary_add_broadcast",
        bench_func=lambda: run_binary_add_broadcast(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={
            "size": size,
            "category": "binary_broadcast",
            "op": "add",
            "rows": rows,
            "cols": cols,
        },
    )
    suite.run_single_benchmark(
        name="binary_ldexp",
        bench_func=lambda: run_binary_ldexp(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={
            "size": size,
            "category": "binary_non_common_type",
            "op": "ldexp",
        },
    )
    suite.run_single_benchmark(
        name="comparison_greater",
        bench_func=lambda: run_comparison_greater(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={"size": size, "category": "comparison", "op": "greater"},
    )
    suite.run_single_benchmark(
        name="logical_and",
        bench_func=lambda: run_logical_and(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={"size": size, "category": "logical", "op": "logical_and"},
    )
    suite.run_single_benchmark(
        name="multiout_frexp",
        bench_func=lambda: run_multiout_frexp(
            np, timer, size, runs, warmup, perform_check=perform_check
        ),
        size_params={
            "size": size,
            "category": "multiout_unary",
            "op": "frexp",
        },
    )
