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

from __future__ import annotations

from _benchmark import MicrobenchmarkSuite

"""
STREAM microbenchmark suite.

Operations tested:
1. copy: c[...] = a
2. mul: np.multiply(c, scalar, out=b)
3. add: np.add(a, b, out=c)

The classical STREAM triad is intentionally omitted because cuPyNumeric does
not currently lower `a[...] = b + scalar * c` as one fused backend kernel.
"""


SCALAR = 3.0


def get_noncontiguous_shape(size):
    """Find a 2D factorization for transpose-based non-contiguous views."""
    rows = int(size**0.5)
    while rows > 1 and size % rows != 0:
        rows -= 1
    if rows == 1:
        raise ValueError(
            "non-contiguous STREAM requires size to have a non-trivial "
            "2D factorization"
        )
    return rows, size // rows


def initialize(array_module, size, dtype, contiguous):
    """Allocate contiguous or transpose-based non-contiguous arrays."""
    if contiguous:
        a = array_module.arange(1, size + 1, dtype=dtype)
        b = array_module.full(size, 2, dtype=dtype)
        c = array_module.full(size, 1, dtype=dtype)
        return a, b, c

    shape = get_noncontiguous_shape(size)
    a = array_module.arange(1, size + 1, dtype=dtype).reshape(shape)
    b = array_module.full(shape, 2, dtype=dtype)
    c = array_module.full(shape, 1, dtype=dtype)
    return a.T, b.T, c.T


def check_stream(operation, size, precision, contiguous, result):
    import numpy as host_np

    dtype = host_np.float32 if precision == 32 else host_np.float64
    a, b, c = initialize(host_np, size, dtype, contiguous)

    if operation == "copy":
        c[...] = a
        expected = c
    elif operation == "mul":
        host_np.multiply(c, SCALAR, out=b)
        expected = b
    else:
        host_np.add(a, b, out=c)
        expected = c

    actual = (
        result.get() if hasattr(result, "get") else host_np.asarray(result)
    )

    if not host_np.allclose(actual, expected):
        raise AssertionError("stream result mismatch")


def stream(
    np,
    operation,
    contiguous,
    precision,
    size,
    runs,
    warmup,
    *,
    timer,
    perform_check,
):
    dtype = np.float32 if precision == 32 else np.float64
    a, b, c = initialize(np, size, dtype, contiguous)

    for idx in range(runs + warmup):
        if idx == warmup:
            timer.start()
        if operation == "copy":
            c[...] = a
        elif operation == "mul":
            np.multiply(c, SCALAR, out=b)
        else:
            np.add(a, b, out=c)
    total = timer.stop()

    if perform_check:
        result = b if operation == "mul" else c
        check_stream(operation, size, precision, contiguous, result)

    return total


def _normalize_operation(operation):
    return "mul" if operation == "scale" else operation


def _get_operations(operation):
    operation = _normalize_operation(operation)
    if operation == "all":
        return ["copy", "mul", "add"]
    return [operation]


def _get_precisions(precision):
    if precision == "all":
        return [32, 64]
    return [int(precision)]


def _get_contiguous_modes(contiguous):
    if contiguous == "all":
        return [True, False]
    return [contiguous == "true"]


def _layout_name(contiguous):
    return "contiguous" if contiguous else "noncontiguous"


def _precision_name(precision):
    return f"float{precision}"


def run_benchmarks(
    suite, size, operation, precision, contiguous, perform_check
):
    """Run STREAM benchmarks inside the suite framework."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    precisions = _get_precisions(precision)
    contigs = _get_contiguous_modes(contiguous)
    ops = _get_operations(operation)

    suite.run_timed(
        stream,
        np,
        ops,
        contigs,
        precisions,
        size,
        runs,
        warmup,
        timer=timer,
        perform_check=perform_check,
    )


class StreamSuite(MicrobenchmarkSuite):
    name = "stream"

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("STREAM Suite")
        group.add_argument(
            "--stream-operation",
            type=str,
            default="all",
            choices=["copy", "mul", "scale", "add", "all"],
            help="STREAM operation to run (default: all)",
        )
        group.add_argument(
            "--stream-precision",
            type=str,
            default="32",
            choices=["32", "64", "all"],
            help="STREAM precision in bits (default: 32)",
        )
        group.add_argument(
            "--stream-contiguous",
            type=str,
            default="all",
            choices=["true", "false", "all"],
            help=(
                "STREAM layout to run; 'false' uses transpose-based "
                "non-contiguous views (default: all)"
            ),
        )
        group.add_argument(
            "--stream-check",
            action="store_true",
            help="Validate STREAM results after each timed sample",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.stream_operation = args.stream_operation
        self.stream_precision = args.stream_precision
        self.stream_contiguous = args.stream_contiguous
        self.stream_check = args.stream_check

    def print_config(self):
        msg = [
            f"operation: {self.stream_operation}",
            f"precision: {self.stream_precision}",
            f"contiguous: {self.stream_contiguous}",
            f"check: {self.stream_check}",
        ]
        self.print_panel(msg, "STREAM Suite")

    def run_suite(self, size):
        run_benchmarks(
            self,
            size,
            self.stream_operation,
            self.stream_precision,
            self.stream_contiguous,
            self.stream_check,
        )
