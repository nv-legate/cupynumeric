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

from _benchmark import (
    MicrobenchmarkSuite,
    microbenchmark,
    random_array,
    timed_loop,
)


_VARIANTS = [
    "sort-1D",
    "argsort-1D",
    "sort-2D-flat",
    "argsort-2D-flat",
    "sort-2D-skinny",
    "argsort-2D-skinny",
]


def _get_case_dimension(variant, size):
    if variant.endswith("-1D"):
        return (size,)
    elif variant.endswith("-2D-flat"):
        return (
            math.ceil(math.pow(size, 1 / 4)),
            math.ceil(math.pow(size, 3 / 4)),
        )
    elif variant.endswith("-2D-skinny"):
        return (
            math.ceil(math.pow(size, 3 / 4)),
            math.ceil(math.pow(size, 1 / 4)),
        )
    else:
        raise ValueError(f"Invalid variant: {variant}")


def _args_to_arrays(variant, dtype, size):
    shape = _get_case_dimension(variant, size)
    if variant.startswith("sort"):
        return [
            ("input", shape, dtype),
            ("work", shape, dtype),
            ("output", shape, dtype),
        ]
    else:
        return [
            ("input", shape, dtype),
            ("work", shape, dtype),
            ("work indices", shape, "int64"),
            ("output", shape, "int64"),
        ]


def _args_to_work(variant, size):
    shape = _get_case_dimension(variant, size)
    return math.prod(shape) * math.log2(max(2, shape[-1]))


def _initialize_case(array_module, variant, size, dtype):
    shape = _get_case_dimension(variant, size)

    input_array = random_array(array_module, shape, dtype)

    return input_array


class SortSuite(MicrobenchmarkSuite):
    name = "sort"

    def dtypes(self) -> list[str]:
        return ["float32", "float64"]

    @microbenchmark(
        output_names=["array shape", "time per run (ms)"],
        returns_time=1,
        args_to_arrays=_args_to_arrays,
        args_to_work=_args_to_work,
        plan=[{"variant": variant} for variant in _VARIANTS],
    )
    def sort(np, variant, size, runs, warmup, dtype, *, timer):
        input_array = _initialize_case(np, variant, size, dtype)

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
