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
Solve microbenchmark suite.

Solve variants:
1. solve-1-rhs: solve with 1 right hand side
2. solve-n-rhs: solve with n right hand sides
3. batched-solve-1-rhs: batched solve with 1 right hand side
4. batched-solve-n-rhs: batched solve with n right hand sides

The shared ``--size`` flag is interpreted per variant:
- solve-1-rhs uses ``matrix_size=(ceil(sqrt(size)), ceil(sqrt(size)))``, ``rhs_size=(ceil(sqrt(size)),)``
- solve-n-rhs uses ``matrix_size=(ceil(sqrt(size)), ceil(sqrt(size)))``, ``rhs_size=(ceil(sqrt(size)), ceil(sqrt(size))//2)``
- batched-solve-1-rhs uses ``matrix_size=(ceil(size^(1/3)), ceil(size^(1/3)), ceil(size^(1/3)))``, ``rhs_size=(ceil(size^(1/3)), ceil(size^(1/3)), 1)``
- batched-solve-n-rhs uses ``matrix_size=(ceil(size^(1/3)), ceil(size^(1/3)), ceil(size^(1/3)))``, ``rhs_size=(ceil(size^(1/3)), ceil(size^(1/3)), ceil(size^(1/3))//2)``
"""

from __future__ import annotations

import math

from _benchmark import (
    MicrobenchmarkSuite,
    microbenchmark,
    random_array,
    timed_loop,
)


_VARIANTS = [
    "solve-1-rhs",
    "solve-n-rhs",
    "batched-solve-1-rhs",
    "batched-solve-n-rhs",
]


def _get_case_dimensions(variant, size):
    dimensions = {}
    if variant.startswith("solve"):
        n = math.ceil(math.pow(size, 1 / 2))
        dimensions["matrix_size"] = (n, n)
        if variant.endswith("-1-rhs"):
            dimensions["rhs_size"] = (n,)
        elif variant.endswith("-n-rhs"):
            dimensions["rhs_size"] = (n, n // 2)
        else:
            raise ValueError(f"Invalid variant: {variant}")
    elif variant.startswith("batched-solve"):
        n = math.ceil(math.pow(size, 1 / 3))
        dimensions["matrix_size"] = (n, n, n)
        if variant.endswith("-1-rhs"):
            dimensions["rhs_size"] = (n, n, 1)
        elif variant.endswith("-n-rhs"):
            dimensions["rhs_size"] = (n, n, n // 2)
        else:
            raise ValueError(f"Invalid variant: {variant}")
    else:
        raise ValueError(f"Invalid variant: {variant}")

    return dimensions


def _args_to_arrays(variant, dtype, size):
    dimensions = _get_case_dimensions(variant, size)
    matrix_dims = dimensions["matrix_size"]
    rhs_dims = dimensions["rhs_size"]
    return [
        ("matrix", matrix_dims, dtype),
        ("rhs", rhs_dims, dtype),
        ("solution", rhs_dims, dtype),
    ]


def _args_to_work(variant, size):
    dimensions = _get_case_dimensions(variant, size)
    matrix_dims = dimensions["matrix_size"]
    rhs_dims = dimensions["rhs_size"]
    n = matrix_dims[-1]
    b = 1
    k = rhs_dims[-1]
    if variant.endswith("-1-rhs"):
        k = 1
    if variant.startswith("batched-solve"):
        b = matrix_dims[0]

    return b * (n * n * n + n * n * k)


def _initialize_case(array_module, variant, size, dtype):
    dimensions = _get_case_dimensions(variant, size)

    matrix_shape = dimensions["matrix_size"]
    rhs_shape = dimensions["rhs_size"]

    matrix_array = random_array(array_module, matrix_shape, dtype)
    rhs_array = random_array(array_module, rhs_shape, dtype)

    return matrix_array, rhs_array


class SolveSuite(MicrobenchmarkSuite):
    name = "solve"

    def dtypes(self) -> list[str]:
        return ["float32", "float64"]

    @microbenchmark(
        output_names=["matrix shape", "rhs shape", "time per run (ms)"],
        returns_time=2,
        args_to_arrays=_args_to_arrays,
        args_to_work=_args_to_work,
        plan=[{"variant": variant} for variant in _VARIANTS],
    )
    def solve(np, variant, size, runs, warmup, dtype, *, timer):
        matrix_array, rhs_array = _initialize_case(np, variant, size, dtype)

        def operation():
            return np.linalg.solve(matrix_array, rhs_array)

        avg = timed_loop(operation, timer, runs, warmup) / runs
        return (matrix_array.shape, rhs_array.shape, avg)
