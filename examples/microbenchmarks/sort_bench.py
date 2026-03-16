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
SORT microbenchmark suite.

Sort variants:
1. sort-1D: single dimension sort
2. argsort-1D: single dimension argsort
3. sort-2D-flat: two dimensional sort (large sort dimension)
4. argsort-2D-flat: two dimensional argsort (large sort dimension)
5. sort-2D-skinny: two dimensional sort (small sort dimension)
6. argsort-2D-skinny: two dimensional argsort (small sort dimension)

The shared ``--size`` flag is interpreted per variant:
- sort-1D and argsort-1D use ``size=(size)``
- sort-2D-flat and argsort-2D-flat use ``size=(ceil(size^(1/4)), ceil(size^(3/4)))`` and ``sort_dim=1``
- sort-2D-skinny and argsort-2D-skinny use ``size=(ceil(size^(3/4)), ceil(size^(1/4)))`` and ``sort_dim=1``
"""

from __future__ import annotations

from _benchmark import MicrobenchmarkSuite, timed_loop

import math


def _get_case_dimension(variant, size):
    if variant.endswith("-1D"):
        return {"size": (size,)}
    elif variant.endswith("-2D-flat"):
        return {
            "size": (
                math.ceil(math.pow(size, 1 / 4)),
                math.ceil(math.pow(size, 3 / 4)),
            )
        }
    elif variant.endswith("-2D-skinny"):
        return {
            "size": (
                math.ceil(math.pow(size, 3 / 4)),
                math.ceil(math.pow(size, 1 / 4)),
            )
        }
    else:
        raise ValueError(f"Invalid variant: {variant}")


def _initialize_case(array_module, variant, size, dtype):
    dimensions = _get_case_dimension(variant, size)

    shape = dimensions["size"]

    input_array = array_module.random.random(shape).astype(dtype)

    return dimensions, input_array


def sort(np, variant, size, runs, warmup, precision, *, timer):
    dtype = np.float32 if precision == 32 else np.float64
    _, input_array = _initialize_case(np, variant, size, dtype)

    if variant.startswith("sort"):

        def operation():
            return np.sort(input_array, -1)

    elif variant.startswith("argsort"):

        def operation():
            return np.argsort(input_array, -1)
    else:
        raise ValueError(f"Invalid variant: {variant}")

    return timed_loop(operation, timer, runs, warmup)


def _get_variants(variant):
    if variant == "all":
        return [
            "sort-1D",
            "argsort-1D",
            "sort-2D-flat",
            "argsort-2D-flat",
            "sort-2D-skinny",
            "argsort-2D-skinny",
        ]
    return [variant]


def _get_precisions(precision):
    if precision == "all":
        return [32, 64]
    return [int(precision)]


def run_benchmarks(suite, size, *, variant="all", precision="32"):
    """Run SORT benchmarks inside the suite framework."""
    np = suite.np
    timer = suite.timer
    runs = suite.runs
    warmup = suite.warmup

    variants = _get_variants(variant)
    precisions = _get_precisions(precision)
    suite.run_timed(
        sort, np, variants, size, runs, warmup, precisions, timer=timer
    )


class SortSuite(MicrobenchmarkSuite):
    name = "sort"

    def add_suite_parser_group(parser):
        group = parser.add_argument_group("SORT Suite")
        group.add_argument(
            "--sort-variant",
            type=str,
            default="all",
            choices=[
                "sort-1D",
                "argsort-1D",
                "sort-2D-flat",
                "argsort-2D-flat",
                "sort-2D-skinny",
                "argsort-2D-skinny",
                "all",
            ],
            help="SORT variant to run (default: all)",
        )
        group.add_argument(
            "--sort-precision",
            type=str,
            default="32",
            choices=["32", "64", "all"],
            help="SOLVE precision in bits (default: 32)",
        )

    def __init__(self, config, args):
        super().__init__(config, args)
        self.sort_variant = args.sort_variant
        self.sort_precision = args.sort_precision

    def print_config(self):
        msg = [
            f"variant: {self.sort_variant}",
            f"precision: {self.sort_precision}",
        ]
        self.print_panel(msg, title="SORT Suite")

    def run_suite(self, size):
        run_benchmarks(
            self,
            size,
            variant=self.sort_variant,
            precision=self.sort_precision,
        )
