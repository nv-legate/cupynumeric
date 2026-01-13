#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation
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

import argparse

import numpy
from benchmark import parse_args, run_benchmark


def check_result(a, b, x):
    print("Checking result...")

    res = b - num.matmul(a, x)
    max_res = num.linalg.norm(res, numpy.inf)
    if max_res < 1e-04:
        print(f"PASS! max-res = {max_res}")
    else:
        print(f"FAIL! max-res = {max_res}")
        assert False


def solve(n, nrhs, dtype, *, perform_check=False, print_timing=False):
    a = num.random.rand(n, n).astype(dtype=dtype)
    b = num.random.rand(n, nrhs).astype(dtype=dtype)

    timer.start()
    x = num.linalg.solve(a, b)
    total = timer.stop()

    if perform_check:
        check_result(a, b, x)

    if print_timing:
        print(f"Elapsed Time: {total} ms")

        if dtype in ["complex64", "complex128"]:
            getrf_flops = (n**3) * 8 / 3
            getrs_flops = (n**2) * nrhs * 8 / 3
        else:
            getrf_flops = (n**3) * 2 / 3
            getrs_flops = (n**2) * nrhs * 2 / 3
        flops = getrf_flops + getrs_flops
        print(f"{flops / total / 1000000} GOP/s")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="n",
        help="number of rows/columns in the matrix",
    )
    parser.add_argument(
        "-s",
        "--nrhs",
        type=int,
        default=1,
        dest="nrhs",
        help="number of right hand sides",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="float64",
        choices=["float32", "float64", "complex64", "complex128"],
        dest="dtype",
        help="data type",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="compare result to numpy",
    )
    args, num, timer = parse_args(parser)

    run_benchmark(
        solve,
        args.benchmark,
        "Solve",
        [
            ("problem size", args.n),
            ("righthand sides", args.nrhs),
            ("datatype", args.dtype),
        ],
        ["time (milliseconds)"],
        perform_check=args.check,
        print_timing=args.timing,
    )
