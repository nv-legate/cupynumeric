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

from _benchmark import benchmark_info, parse_with_harness


def check_result(np, a, q, r):
    print("Checking result...")

    if np.allclose(a, np.matmul(q, r)):
        print("PASS!")
    else:
        print("FAIL!")

        qtq = np.matmul(q.T, q)
        qtq_err = np.eye(q.shape[1]) - qtq
        q_max = np.max(qtq_err)
        print(f"cunumeric I-qTq max: {q_max}")


@benchmark_info(name="QR")
def qr(np, m, n, dtype, *, timer, perform_check=False, print_timing=False):
    a = np.random.rand(m, n).astype(dtype=dtype)

    timer.start()
    q, r = np.linalg.qr(a)
    total = timer.stop()

    if perform_check:
        check_result(np, a, q, r)

    if print_timing:
        print(f"Elapsed Time: {total} ms")

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
        default=[1000],
        dest="m",
        help="number of rows in the matrix",
    )
    parser.add_argument(
        "-n",
        "--cols",
        type=int,
        nargs="+",
        default=[1000],
        dest="n",
        help="number of cols in the matrix",
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
    args, harness = parse_with_harness(parser)

    harness.run_timed(
        qr,
        harness.np,
        args.m,
        args.n,
        args.dtype,
        timer=harness.timer,
        perform_check=args.check,
        print_timing=args.timing,
    )
