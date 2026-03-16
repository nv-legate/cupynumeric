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
SOLVE microbenchmark suite.

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

from _benchmark import MicrobenchmarkSuite, timed_loop


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


def _initialize_case(array_module, variant, size, dtype):
    dimensions = _get_case_dimensions(variant, size)

    matrix_shape = dimensions["matrix_size"]
    rhs_shape = dimensions["rhs_size"]

    matrix_array = array_module.random.random(matrix_shape).astype(dtype)
    rhs_array = array_module.random.random(rhs_shape).astype(dtype)

    return dimensions, matrix_array, rhs_array


def solve(np, variant, size, runs, warmup, precision, *, timer):
    dtype = np.float32 if precision == 32 else np.float64
    _, matrix_array, rhs_array = _initialize_case(np, variant, size, dtype)

    def operation():
        return np.linalg.solve(matrix_array, rhs_array)

    return timed_loop(operation, timer, runs, warmup)


def _get_variants(variant):
    if variant == "all":
        return [
            "solve-1-rhs",
            "solve-n-rhs",
            "batched-solve-1-rhs",
            "batched-solve-n-rhs",
        ]
    return [variant]


def _get_precisions(precision):
    if precision == "all":
        return [32, 64]
    return [int(precision)]


def run_benchmarks(suite, size, *, variant="all", precision="64"):
    """Run SOLVE benchmarks inside the suite framework."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup
    precisions = _get_precisions(precision)
    variants = _get_variants(variant)
    suite.run_timed(
        solve, np, variants, size, runs, warmup, precisions, timer=timer
    )


class SolveSuite(MicrobenchmarkSuite):
    name = "solve"

    def add_suite_parser_group(parser):
        group = parser.add_argument_group("SOLVE Suite")
        group.add_argument(
            "--solve-variant",
            type=str,
            default="all",
            choices=[
                "solve-1-rhs",
                "solve-n-rhs",
                "batched-solve-1-rhs",
                "batched-solve-n-rhs",
                "all",
            ],
            help="SOLVE variant to run (default: all)",
        )
        group.add_argument(
            "--solve-precision",
            type=str,
            default="64",
            choices=["32", "64", "all"],
            help="SOLVE precision in bits (default: 64)",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.solve_variant = args.solve_variant
        self.solve_precision = args.solve_precision

    def print_config(self):
        msg = [
            f"variant: {self.solve_variant}",
            f"precision: {self.solve_precision}",
        ]
        self.print_panel(msg, title="SOLVE Suite")

    def run_suite(self, size):
        run_benchmarks(
            self,
            size,
            variant=self.solve_variant,
            precision=self.solve_precision,
        )
