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

from _benchmark import (
    benchmark_info,
    format_dtype,
    parse_with_harness,
    get_benchmark_info,
)


def initialize(np, M, N, K, ft):
    A = np.random.uniform(size=(N, N)).astype(ft)
    B = np.random.uniform(size=(N, N)).astype(ft)
    C = np.zeros((N, N), dtype=ft)
    return A, B, C


def total_flops(M, N, K):
    return M * N * (2 * K - 1)


def total_space(np, M, N, K, ft):
    return (M * N + M * K + K * N) * np.dtype(ft).itemsize


@benchmark_info(formats={"ft": format_dtype})
def run_gemm(np, N, I, warmup, ft, *, timer):  # noqa: E741
    print("Problem Size:     M=" + str(N) + " N=" + str(N) + " K=" + str(N))
    print("Total Iterations: " + str(I))
    flops = total_flops(N, N, N)
    print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
    space = total_space(np, N, N, N, ft)
    print("Total Size:       " + str(space / 1e6) + " MB")
    A, B, C = initialize(np, N, N, N, ft)

    timer.start()
    # Run for as many iterations as was requested
    for idx in range(I + warmup):
        if idx == warmup:
            timer.start()
        np.dot(A, B, out=C)
        # We need to rotate the matrices to keep Legate honest
        # about moving data so it can't just duplicate A and B
        # on the first iteration and reuse them, this means
        # that A, B, C all need to be square
        A, B, C = B, C, A
    total = timer.stop()

    print("Elapsed Time:     " + str(total) + " ms")
    average = total / I
    print("Average GEMM:     " + str(average) + " ms")
    print("FLOPS/s:          " + str(flops / (average * 1e6)) + " GFLOPS/s")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        dest="I",
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
        "--precision",
        type=int,
        default=32,
        dest="P",
        help="number of bits of precision to use for the gemm computation "
        "(16,32,64)",
    )

    args, harness = parse_with_harness(parser)
    np = harness.np

    name = None
    dtype = None
    if args.P == 16:
        name = "HGEMM"
        dtype = np.float16
    elif args.P == 32:
        name = "SGEMM"
        dtype = np.float32
    elif args.P == 64:
        name = "DGEMM"
        dtype = np.float64
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")
    info = get_benchmark_info(run_gemm).replace(name=name)
    harness.run_timed_with_info(
        info,
        run_gemm,
        np,
        args.N,
        args.I,
        args.warmup,
        dtype,
        timer=harness.timer,
    )
