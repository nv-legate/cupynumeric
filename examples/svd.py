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
from _benchmark import benchmark_info, parse_with_harness


def check_result(np, a, u, s, vh):
    print("Checking result...")

    # (u * s) @ vh
    m = a.shape[0]
    n = a.shape[1]
    k = min(m, n)

    u = u[:, :k] if k < m else u
    vh = vh[:k, :] if k < m else vh
    a2 = np.matmul(u * s, vh)
    print("PASS!" if np.allclose(a, a2) else "FAIL!")


@benchmark_info(name="SVD")
def svd(
    np,
    m,
    n,
    full_matrices,
    dtype,
    *,
    timer,
    perform_check=False,
    print_timing=False,
):
    if numpy.issubdtype(dtype, numpy.integer):
        a = np.random.randint(0, 1000, size=m * n).astype(dtype)
        a = a.reshape((m, n))
    elif numpy.issubdtype(dtype, numpy.floating):
        a = np.random.random((m, n)).astype(dtype)
    elif numpy.issubdtype(dtype, numpy.complexfloating):
        a = np.array(
            np.random.random((m, n)) + np.random.random((m, n)) * 1j
        ).astype(dtype)
    else:
        print("unsupported type " + str(dtype))
        assert False

    timer.start()
    u, s, vh = np.linalg.svd(a, full_matrices)
    total = timer.stop()

    if perform_check:
        check_result(np, a, u, s, vh)

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
        "--no-full-matrices",
        dest="full_matrices",
        action="store_false",
        help="return u as (m,k)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="float64",
        choices=[
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ],
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
        svd,
        harness.np,
        args.m,
        args.n,
        args.full_matrices,
        args.dtype,
        timer=harness.timer,
        perform_check=args.check,
        print_timing=args.timing,
    )
