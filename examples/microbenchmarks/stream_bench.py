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

import itertools
import math
from typing import Any

from _benchmark import (
    ArrayDescription,
    MicrobenchmarkSuite,
    microbenchmark,
    random_array,
    timed_loop,
)

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


def _args_to_arrays(
    size: int, dtype: str, contiguous: bool
) -> list[ArrayDescription]:
    shape = size
    if not contiguous:
        n = math.isqrt(size)
        shape = (n, n)
    return [("a", shape, dtype), ("b", shape, dtype), ("c", shape, dtype)]


def _to_host(array):
    import numpy as host_np

    return array.get() if hasattr(array, "get") else host_np.asarray(array)


def _initialize(array_module, size, dtype, contiguous):
    """Allocate contiguous or transpose-based non-contiguous arrays."""
    if contiguous:
        a = array_module.arange(1, size + 1, dtype=dtype)
        b = array_module.full(size, 2, dtype=dtype)
        c = array_module.full(size, 1, dtype=dtype)
        return a, b, c

    n = math.isqrt(size)
    shape = (n, n)
    a = random_array(array_module, shape, dtype)
    b = array_module.full(shape, 2, dtype=dtype)
    c = array_module.full(shape, 1, dtype=dtype)
    return a.T, b.T, c.T


def _check_stream(operation, a, b, c, result):
    import numpy as host_np

    if operation == "copy":
        expected = _to_host(a)
    elif operation == "mul":
        expected = _to_host(c) * SCALAR
    else:
        expected = _to_host(a) + _to_host(b)

    actual = _to_host(result)

    if not host_np.allclose(actual, expected):
        raise AssertionError("stream result mismatch")


_OPS = ["copy", "mul", "add"]


class StreamSuite(MicrobenchmarkSuite):
    name = "stream"

    def dtypes(self) -> list[str]:
        return ["float32", "float64"]

    def default_arguments(self) -> dict[str, Any]:
        return {
            **super().default_arguments(),
            "perform_check": self.stream_check,
        }

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("STREAM Suite")
        group.add_argument(
            "--stream-check",
            action="store_true",
            help="Validate STREAM results after each timed sample",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.stream_check = args.stream_check

    def print_config(self):
        msg = [f"check: {self.stream_check}"]
        self.print_panel(msg, "STREAM Suite")

    @microbenchmark(
        args_to_arrays=_args_to_arrays,
        plan=[
            {"operation": op, "contiguous": contig}
            for op, contig in itertools.product(_OPS, [True, False])
        ],
    )
    def stream(
        np,
        operation,
        contiguous,
        dtype,
        size,
        runs,
        warmup,
        *,
        timer,
        perform_check,
    ):
        a, b, c = _initialize(np, size, dtype, contiguous)
        check_inputs = None

        if perform_check:
            # Preserve the pre-op operands so validation still catches unexpected
            # source-array mutations while using the actual benchmark inputs.
            check_inputs = tuple(_to_host(array).copy() for array in (a, b, c))

        def op():
            if operation == "copy":
                c[...] = a
            elif operation == "mul":
                np.multiply(c, SCALAR, out=b)
            else:
                np.add(a, b, out=c)

        total = timed_loop(op, timer, runs, warmup) / runs

        if perform_check:
            result = b if operation == "mul" else c
            assert check_inputs is not None
            _check_stream(operation, *check_inputs, result)

        return total
