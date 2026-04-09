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
Sort microbenchmark suite.

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

import math

from _benchmark import MicrobenchmarkSuite, benchmark_info, timed_loop
from _benchmark.sizing import (
    SizeRequest,
    resolve_size_by_binary_search,
    resolve_suite_size,
)


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


def _get_dtypes(precision):
    if precision == "all":
        return ["float32", "float64"]
    else:
        assert precision in ["32", "64"]
        return [f"float{precision}"]


def _dtype_bytes(dtype) -> int:
    import numpy

    return numpy.dtype(dtype).itemsize


def _initial_bytes_per_size(precision) -> int:
    # Seed the search with a cheap upper-bound guess before the full estimate.
    return 8 + max(_dtype_bytes(d) for d in _get_dtypes(precision))


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


def _estimate_case_working_set_bytes(variant, dtype, size):
    elements = math.prod(_get_case_dimension(variant, size)["size"])
    return elements * (8 + _dtype_bytes(dtype))


def _estimate_working_set_bytes(variant, precision, size):
    return max(
        _estimate_case_working_set_bytes(case, dtype, size)
        for case in _get_variants(variant)
        for dtype in _get_dtypes(precision)
    )


def _resolve_size_from_memory_target(variant, precision, target_bytes):
    return resolve_size_by_binary_search(
        target_bytes,
        estimate_working_set_bytes=lambda size: _estimate_working_set_bytes(
            variant, precision, size
        ),
        initial_guess=target_bytes // _initial_bytes_per_size(precision),
    )


def _describe_size(size, variant, precision):
    lines = [
        f"variants: {', '.join(_get_variants(variant))}",
        f"dtypes:   {', '.join(_get_dtypes(precision))}",
    ]
    for case in _get_variants(variant):
        shape = _get_case_dimension(case, size)["size"]
        lines.append(f"{case}: {' x '.join(str(dim) for dim in shape)}")
    return lines


def _initialize_case(array_module, variant, size, dtype):
    dimensions = _get_case_dimension(variant, size)

    shape = dimensions["size"]

    input_array = array_module.random.random(shape).astype(dtype)

    return dimensions, input_array


@benchmark_info(
    output_names=["array shape", "time per run (ms)"], returns_time=1
)
def sort(np, variant, size, runs, warmup, dtype, *, timer):
    _, input_array = _initialize_case(np, variant, size, dtype)

    if variant.startswith("sort"):

        def operation():
            return np.sort(input_array, -1)

    elif variant.startswith("argsort"):

        def operation():
            return np.argsort(input_array, -1)
    else:
        raise ValueError(f"Invalid variant: {variant}")

    avg = timed_loop(operation, timer, runs, warmup) / runs
    return (input_array.shape, avg)


def run_benchmarks(suite, size_request, *, variant="all", precision="32"):
    """Run Sort benchmarks inside the suite framework."""
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
            resolved_size, variant, precision
        ),
    )
    if resolutions is not None:
        suite.print_size_resolution(resolutions)

    variants = _get_variants(variant)
    dtypes = _get_dtypes(precision)
    suite.run_timed(
        sort, np, variants, sizes, runs, warmup, dtypes, timer=timer
    )


class SortSuite(MicrobenchmarkSuite):
    name = "sort"

    @staticmethod
    def add_suite_parser_group(parser):
        group = parser.add_argument_group("Sort Suite")
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
            help="Sort variant to run",
        )
        group.add_argument(
            "--sort-precision",
            type=str,
            default="32",
            choices=["32", "64", "all"],
            help="Sort precision in bits",
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
        self.print_panel(msg, title="Sort Suite")

    def run_suite(self, size_request: SizeRequest):
        run_benchmarks(
            self,
            size_request,
            variant=self.sort_variant,
            precision=self.sort_precision,
        )
