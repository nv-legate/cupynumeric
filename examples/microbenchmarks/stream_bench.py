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


def run_stream_case(
    np,
    timer,
    size,
    runs,
    warmup,
    operation,
    precision,
    contiguous,
    *,
    perform_check=False,
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
    suite,
    size,
    *,
    operation="all",
    precision="32",
    contiguous="all",
    perform_check=False,
):
    """Run STREAM benchmarks inside the suite framework."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    for stream_operation in _get_operations(operation):
        for stream_precision in _get_precisions(precision):
            for stream_contiguous in _get_contiguous_modes(contiguous):
                benchmark_name = (
                    f"{stream_operation}_"
                    f"{_precision_name(stream_precision)}_"
                    f"{_layout_name(stream_contiguous)}"
                )
                suite.run_single_benchmark(
                    benchmark_name,
                    lambda op=stream_operation,
                    prec=stream_precision,
                    cont=stream_contiguous: run_stream_case(
                        np,
                        timer,
                        size,
                        runs,
                        warmup,
                        op,
                        prec,
                        cont,
                        perform_check=perform_check,
                    ),
                    size_params={
                        "size": size,
                        "operation": stream_operation,
                        "precision": stream_precision,
                        "contiguous": stream_contiguous,
                    },
                )
