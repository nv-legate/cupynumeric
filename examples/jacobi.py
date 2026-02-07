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
import math

from benchmark import parse_args, run_benchmark


def generate_random(N):
    print("Generating %dx%d system..." % (N, N))
    # Generate a random matrix
    A = np.random.rand(N, N)
    # Make sure that it is diagonally dominate
    A = A + N * np.eye(N)
    # Generate a random vector
    b = np.random.rand(N)
    return A, b


def check(A, x, b):
    print("Checking result...")
    return np.allclose(A.dot(x), b)


def run_jacobi(N, iters, warmup, perform_check, print_timing, verbose):
    A, b = generate_random(N)

    print("Solving system...")
    x = np.zeros(A.shape[1])
    d = np.diag(A)
    R = A - np.diag(d)

    timer.start()
    for i in range(iters + warmup):
        if i == warmup:
            timer.start()
        x = (b - np.dot(R, x)) / d
    total = timer.stop()

    if perform_check:
        assert check(A, x, b)
    else:
        assert not math.isnan(np.sum(x)), (
            f"{np.count_nonzero(np.isnan(x))} NaNs, "
            f"{np.count_nonzero(np.isinf(x))} infs in x"
        )

    if print_timing:
        print(f"Elapsed Time: {total} ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=1000,
        dest="iters",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        nargs="+",
        default=[100],
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_jacobi,
        args.benchmark,
        "Jacobi",
        [
            ("problem size", args.N),
            ("iterations", args.iters),
            ("warmup iterations", args.warmup),
        ],
        ["time (milliseconds)"],
        perform_check=args.check,
        print_timing=args.timing,
        verbose=args.verbose,
    )
