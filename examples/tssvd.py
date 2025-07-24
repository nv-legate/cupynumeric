#!/usr/bin/env python

# Copyright 2025 NVIDIA Corporation
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

import numpy as np
from benchmark import parse_args, run_benchmark

import cupynumeric as num


def check_result(a, u, s, vh):
    print("Checking result...")

    # (u * s) @ vh
    a2 = num.matmul(u * s, vh)
    print("PASS!" if num.allclose(a, a2) else "FAIL!")


# make random real full column rank mxn, m>n matrix:
#
def make_random_matrix(
    m: int, n: int, scale: float = 10.0, dtype_=np.dtype("float64")
) -> np.ndarray:
    num.random.seed(6174)

    mat = scale * num.random.rand(m, n)

    mat = mat.astype(dtype_)

    # strictly diagonally dominant:
    #
    for i in range(n):
        mat[i, i] = 1.0 + num.sum(num.abs(mat[i, :]))

    return mat


def run_tssvd(m, n, perform_check, timing):
    A = make_random_matrix(m, n)

    timer.start()
    u, s, vh = num.linalg.tssvd(A)
    total = timer.stop()

    if perform_check:
        check_result(A, u, s, vh)

    if timing:
        print(f"TSSVD elapsed Time: {total:.3f} ms")


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
        "-m",
        "--rows",
        type=int,
        default=10,
        dest="m",
        help="number of rows in the matrix",
    )
    parser.add_argument(
        "-n",
        "--cols",
        type=int,
        default=10,
        dest="n",
        help="number of cols in the matrix",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="compare result to numpy",
    )

    global num

    args, num, timer = parse_args(parser)

    run_benchmark(
        run_tssvd,
        args.benchmark,
        "TSSVD",
        (args.m, args.n, args.check, args.timing),
    )
