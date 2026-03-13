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

from _benchmark import MicrobenchmarkSuite, timed_loop


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


def unary_exp(np, size, runs, warmup, *, timer, perform_check):
    x = _float_input(np, size)
    out = np.empty_like(x)

    def operation():
        np.exp(x, out=out)

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected = host_np.exp(_to_host(x))
        _check_allclose("exp", out, expected)
    return total


def binary_add(np, size, runs, warmup, *, timer, perform_check):
    lhs = _float_input(np, size)
    rhs = _float_input(np, size, start=2.0)
    out = np.empty_like(lhs)

    def operation():
        np.add(lhs, rhs, out=out)

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected = _to_host(lhs) + _to_host(rhs)
        _check_allclose("add", out, expected)
    return total


def binary_add_broadcast(
    np, rows, cols, runs, warmup, *, timer, perform_check
):
    lhs = _float_input(np, rows * cols).reshape((rows, cols))
    rhs = _float_input(np, cols, start=3.0)
    out = np.empty_like(lhs)

    def operation():
        np.add(lhs, rhs, out=out)

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected = _to_host(lhs) + _to_host(rhs)
        _check_allclose("add_broadcast", out, expected)
    return total


def comparison_greater(np, size, runs, warmup, *, timer, perform_check):
    lhs = _float_input(np, size, start=2.0)
    rhs = _float_input(np, size, start=1.0)
    out = np.empty(lhs.shape, dtype=np.bool_)

    def operation():
        np.greater(lhs, rhs, out=out)

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected = _to_host(lhs) > _to_host(rhs)
        _check_equal("greater", out, expected)
    return total


def binary_ldexp(np, size, runs, warmup, *, timer, perform_check):
    lhs = _float_input(np, size, start=2.0)
    rhs = np.arange(size, dtype=np.int32) % 6
    out = np.empty_like(lhs)

    def operation():
        np.ldexp(lhs, rhs, out=out)

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected = host_np.ldexp(_to_host(lhs), _to_host(rhs))
        _check_allclose("ldexp", out, expected)
    return total


def logical_and(np, size, runs, warmup, *, timer, perform_check):
    lhs = _bool_input(np, size, offset=0)
    rhs = _bool_input(np, size, offset=1)
    out = np.empty(lhs.shape, dtype=np.bool_)

    def operation():
        np.logical_and(lhs, rhs, out=out)

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected = host_np.logical_and(_to_host(lhs), _to_host(rhs))
        _check_equal("logical_and", out, expected)
    return total


def multiout_frexp(np, size, runs, warmup, *, timer, perform_check):
    x = _float_input(np, size, start=4.0)
    mantissa = np.empty_like(x)
    exponent = np.empty(x.shape, dtype=np.int32)

    def operation():
        np.frexp(x, out=(mantissa, exponent))

    total = timed_loop(operation, timer, runs, warmup)
    if perform_check:
        expected_mantissa, expected_exponent = host_np.frexp(_to_host(x))
        _check_allclose("frexp_mantissa", mantissa, expected_mantissa)
        _check_equal("frexp_exponent", exponent, expected_exponent)
    return total


def run_benchmarks(suite, size, perform_check):
    """Run representative ufunc benchmarks."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    args = (np, size, runs, warmup)
    kwargs = {"timer": timer, "perform_check": perform_check}

    basic_ufuncs = [
        unary_exp,
        binary_add,
        comparison_greater,
        binary_ldexp,
        logical_and,
        multiout_frexp,
    ]

    for f in basic_ufuncs:
        suite.run_timed(f, *args, **kwargs)

    rows, cols = _broadcast_shape(size)
    suite.run_timed(
        binary_add_broadcast, np, rows, cols, runs, warmup, **kwargs
    )


class UfuncSuite(MicrobenchmarkSuite):
    name = "ufunc"

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("Ufunc Suite")
        group.add_argument(
            "--ufunc-check",
            action="store_true",
            help="Validate ufunc benchmark results after each timed sample",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.ufunc_check = args.ufunc_check

    def print_config(self):
        msg = [f"check: {self.ufunc_check}"]
        self.print_panel(msg, "Ufunc Suite")

    def run_suite(self, size):
        run_benchmarks(self, size, self.ufunc_check)
