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

from _benchmark import benchmark_info, parse_with_harness


def check_result(np, a, u, s, vh):
    print("Checking result...")

    # (u * s) @ vh
    a2 = np.matmul(u * s, vh)
    print("PASS!" if np.allclose(a, a2) else "FAIL!")


# make random real full column rank mxn, m>n matrix:
#
def make_random_matrix(np, m, n):
    np.random.seed(6174)

    mat = np.random.rand(m, n)

    # strictly diagonally dominant:
    #
    for i in range(n):
        mat[i, i] = 1.0 + np.sum(np.abs(mat[i, :]))

    return mat


@benchmark_info(name="TSSVD")
def run_tssvd(np, m, n, *, timer, perform_check=False, print_timing=False):
    A = make_random_matrix(np, m, n)

    timer.start()
    u, s, vh = np.linalg.tssvd(A)
    total = timer.stop()

    if perform_check:
        check_result(np, A, u, s, vh)

    if print_timing:
        print(f"TSSVD elapsed Time: {total:.3f} ms")

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
        "-m",
        "--rows",
        type=int,
        nargs="+",
        default=[10],
        dest="m",
        help="number of rows in the matrix",
    )
    parser.add_argument(
        "-n",
        "--cols",
        type=int,
        nargs="+",
        default=[10],
        dest="n",
        help="number of cols in the matrix",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="compare result to numpy",
    )

    args, harness = parse_with_harness(parser)

    harness.run_timed(
        run_tssvd,
        harness.np,
        args.m,
        args.n,
        timer=harness.timer,
        perform_check=args.check,
        print_timing=args.timing,
    )
